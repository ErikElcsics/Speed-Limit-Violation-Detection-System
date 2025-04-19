import streamlit as st
import cv2
import tempfile
import numpy as np
import time
import pandas as pd
from ultralytics import YOLO
import easyocr
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from PIL import Image
import torch

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'], gpu=True)


def read_license_plate(frame, box):
    x1, y1, x2, y2 = box
    plate_img = frame[y1:y2, x1:x2]

    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    results = ocr_reader.readtext(plate_img)
    if results:
        return results[0][1]
    return "Unclear"


def estimate_speed(prev_pos, curr_pos, time_elapsed, ppm):
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    dist_pixels = np.sqrt(dx ** 2 + dy ** 2)
    dist_meters = dist_pixels / ppm
    speed_mps = dist_meters / time_elapsed
    speed_mph = speed_mps * 2.237
    return speed_mph


def send_email_with_attachment(to_email, df):
    try:
        sender_email = "your_email@example.com"
        sender_password = "your_email_password"
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        subject = "Speed Limit Violation Report"
        body = "Attached is the CSV file with the list of speed limit violations."
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(df.to_csv(index=False).encode())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', "attachment; filename=violators.csv")
        msg.attach(attachment)

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        st.success(f"Email sent successfully to {to_email}")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🚗 Speed Limit Violation Detection System")
st.markdown("""
Upload a video. The app will track cars, estimate their speeds, read their license plates, and highlight any vehicle that goes over the speed limit (default 20 mph). Red boxes = speeding. Green = under the limit.
""")

col1, col2 = st.columns([3, 2])

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**💻 Running on:** `{device.upper()}`")

model = YOLO('yolov8n.pt').to(device)
license_plate_detector = YOLO('license_plate_detector.pt').to(device)

ppm = 8
placeholder_chart = st.empty()
violators_data = []
stframe = st.empty()
prev_positions = {}
car_speeds = {}
speeding_car_images = []
speeding_cars = set()  # Track which cars are currently speeding


def process_frame_for_video(frame):
    global prev_positions, car_speeds, speeding_car_images, violators_data, speeding_cars
    results = model.track(frame, persist=True)
    license_plate_results = license_plate_detector(frame)[0]
    annotated_frame = frame.copy()
    curr_time = time.time()

    for lp_box in license_plate_results.boxes:
        x1, y1, x2, y2 = map(int, lp_box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plate_text = read_license_plate(frame, (x1, y1, x2, y2))
        cv2.putText(annotated_frame, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if model.names[cls] != 'car':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_id = int(box.id[0]) if box.id is not None else None
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Default color is green (not speeding)
        box_color = (0, 255, 0)
        label_text = ""

        if car_id in prev_positions:
            prev_pos, prev_time = prev_positions[car_id]
            time_diff = curr_time - prev_time
            if time_diff > 0:  # Only calculate speed if time has passed
                speed = estimate_speed(prev_pos, (center_x, center_y), time_diff, ppm)
                car_speeds[car_id] = speed

                if speed > speed_limit:
                    speeding_cars.add(car_id)  # Add to speeding cars set
                    box_color = (0, 0, 255)  # Red for speeding
                    label_text = f"Speeding: {round(speed, 1)} mph"

                    if car_id not in [v["Car ID"] for v in violators_data]:
                        plate_number = read_license_plate(frame, (x1, y1, x2, y2))
                        timestamp = time.strftime("%H:%M:%S", time.localtime(curr_time))

                        violators_data.append({
                            "Car ID": car_id,
                            "Speed (mph)": round(speed, 2),
                            "License Plate": plate_number,
                            "Timestamp": timestamp
                        })

                        car_img = frame[y1:y2, x1:x2]
                        if car_img.size > 0:
                            car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
                            car_img_resized = cv2.resize(car_img, (600, 400))
                            speeding_car_images.append((car_id, car_img_resized, speed, timestamp))
                else:
                    # Only remove from speeding set if speed is below threshold with some buffer
                    if car_id in speeding_cars and speed < speed_limit * 0.9:  # 10% buffer to avoid flickering
                        speeding_cars.discard(car_id)

            prev_positions[car_id] = ((center_x, center_y), curr_time)
        else:
            prev_positions[car_id] = ((center_x, center_y), curr_time)

        # If car was previously detected as speeding, keep it red
        if car_id in speeding_cars:
            box_color = (0, 0, 255)
            if not label_text:  # If we don't have a current speed reading, use the last known speed
                last_speed = car_speeds.get(car_id, 0)
                label_text = f"Speeding: {round(last_speed, 1)} mph"

        # Draw the bounding box and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        if label_text:
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    if violators_data:
        df = pd.DataFrame(violators_data).drop_duplicates(subset='Car ID', keep='last')
        placeholder_chart.dataframe(df)

    return annotated_frame


# Sidebar
with st.sidebar:
    st.header("📋 Settings")
    use_live_cam = st.checkbox("📷 Use Live Camera")
    uploaded_video = None
    if not use_live_cam:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    speed_limit = st.slider("Set Speed Limit (mph)", min_value=1, max_value=100, value=20)

    show_gallery = st.checkbox("📸 Show Speeding Cars Gallery")

    st.markdown("---")
    st.subheader("🛠️ Calibration Tips")
    st.info("""
    **For best OCR results, ensure plates are:**
    - Well-lit  
    - At least 100px wide  
    - Facing the camera
    """)

    st.markdown("---")
    st.subheader("📧 Send Violator Data via Email")
    recipient_email = st.text_input("Recipient's Email", placeholder="Enter recipient email")
    send_email_button = st.button("📤 Send Violator Data via Email")

    st.markdown("---")
    if st.button("🔄 Reset System"):
        prev_positions.clear()
        car_speeds.clear()
        speeding_car_images.clear()
        violators_data.clear()
        speeding_cars.clear()
        placeholder_chart.empty()
        stframe.empty()
        st.success("System reset! All data cleared.")

# Video Upload Mode
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and get the annotated frame
        annotated_frame = process_frame_for_video(frame)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the frame in Streamlit
        with col1:
            stframe.image(annotated_frame, channels="BGR", width=800)

    cap.release()
    out.release()

# Live Camera Mode
elif use_live_cam:
    cam = cv2.VideoCapture(0)
    stop_button = st.button("❌ Stop Camera")
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret or stop_button:
            break

        # Process the frame and get the annotated frame
        annotated_frame = process_frame_for_video(frame)

        # Display the frame in Streamlit
        with col1:
            stframe.image(annotated_frame, channels="BGR", width=800)
    cam.release()

# Summary
if violators_data:
    st.subheader("Speed Limit Violations Summary")
    final_df = pd.DataFrame(violators_data).drop_duplicates(subset='Car ID', keep='last')
    st.dataframe(final_df)

    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Violator Data as CSV", data=csv, file_name="violators.csv", mime="text/csv")

    if not use_live_cam and uploaded_video:
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.download_button("⬇️ Download Processed Video", data=video_bytes,
                           file_name="processed_output.mp4", mime="video/mp4")

    if recipient_email and send_email_button:
        send_email_with_attachment(recipient_email, final_df)
else:
    st.success("No speed limit violations detected yet!")

if show_gallery:
    st.subheader("🚨 Speeding Cars Gallery")

    # Get the unique car IDs from the violators data
    if violators_data:
        # Create a dictionary to map car IDs to their latest image and info
        car_gallery_data = {}
        final_df_ids = set(final_df['Car ID'].unique())

        # Filter and keep only the latest image for each car in the final_df
        for car_id, img, speed, timestamp in speeding_car_images:
            if car_id in final_df_ids:
                # Only keep if this is a newer timestamp than what we have
                if car_id not in car_gallery_data:
                    car_gallery_data[car_id] = (img, speed, timestamp)
                else:
                    existing_timestamp = car_gallery_data[car_id][2]
                    if timestamp > existing_timestamp:
                        car_gallery_data[car_id] = (img, speed, timestamp)

        # Ensure we have exactly the same cars as in the summary
        if car_gallery_data:
            # Create list sorted by car ID to match the summary order
            gallery_items = sorted([(car_id, data[0], data[1], data[2])
                                    for car_id, data in car_gallery_data.items()],
                                   key=lambda x: x[0])

            cols = st.columns(3)
            for i, (car_id, img, speed, timestamp) in enumerate(gallery_items):
                with cols[i % 3]:
                    st.image(img, caption=f"Car {car_id}\n{round(speed, 1)} mph @ {timestamp}", width=300)
        else:
            st.warning("No speeding cars images available for the violators in the summary.")
    else:
        st.warning("No speeding cars detected yet.")