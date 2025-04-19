# 🚗 Speed Limit Violation Detection System - Speed Limit Violation Detection System - Live or Upload Videos, Estimates Speed, Extract License plate number, Reports Violation Summary, Vehicle Gallery in Violation, and Email reporting


A powerful Speed Limit Violation Detection Streamlit-based computer vision app that detects speeding vehicles from uploaded videos or live webcam feed. The system tracks vehicles using YOLOv8, estimates their speeds, extracts license plate numbers using OCR, and provides a detailed dashboard with downloadable violation data and optional email reporting.



## Features

✅ Live webcam feed or use Upload video
✅ Detects and tracks vehicles using YOLOv8  
✅ Estimates vehicle speed using position and time data  
✅ OCR-based license plate recognition with EasyOCR  
✅ Visual annotations:  
  - 🔴 Red box for speeding vehicles  
  - 🟢 Green box for vehicles under the limit  
✅ Adjustable speed limit slider   
✅ CSV export of violators  
✅ Optional email report sending with violations  
✅ Gallery view for each speeding vehicle (1 image per car)  
✅ Fully interactive Streamlit UI with side panel configuration  



## Technologies & Libraries Used

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) - UI framework
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) - Vehicle detection and tracking
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - License plate recognition
- [OpenCV](https://opencv.org/) - Frame processing and speed calculation
- [NumPy](https://numpy.org/) - Math utilities
- [Pandas](https://pandas.pydata.org/) - Data handling
- [smtplib](https://docs.python.org/3/library/smtplib.html) - Email integration
- [Pillow (PIL)](https://pillow.readthedocs.io/) - Image handling



## Installation

> Python 3.8+ recommended

1. Clone the repository:

- git clone https://github.com/yourusername/speed-violation-detector.git

- cd speed-violation-detector


2. Create and activate a virtual environment:

- python -m venv venv
- source venv/bin/activate  # or venv\Scripts\activate on Windows


3. Install dependencies:

- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
- pip install ultralytics easyocr opencv-python-headless numpy pandas streamlit Pillow 
- pip install secure-smtplib


## ▶️ Running the App


streamlit run Speed_Violation_Detector.py


## How It Works

1. Object Detection: YOLOv8 detects cars in each video frame.
2. Tracking: Vehicles are tracked with unique IDs frame-by-frame.
3. Speed Estimation: Speed is calculated using the change in position over time.
4. Violation Detection: If a car exceeds the speed limit, it is marked as a violator.
5. OCR & Logging: EasyOCR attempts to extract the license plate number.
6. Display: Violations are displayed in a summary table and image gallery.
7. Email: You can optionally send a CSV of the violations via email.



## Email Functionality

To use the email feature:

1. Update the `sender_email` and `sender_password` in the code.
2. Ensure less secure apps are enabled in your email settings (Gmail may require an app password if 2FA is on).
3. Enter the recipient's email in the sidebar and click Send.



## Example Use Cases

- School zones & urban speed monitoring  
- Parking lot enforcement  
- Residential speed zone analysis  
- Surveillance analytics in private/public roads  



## 📝 To-Do / Future Improvements

- Improved OCR with Tesseract 
- Or use a faster/more accurate license plate detection



## License

This project is licensed under the MIT License. Feel free to use and modify it.



## 🤝 Contributing

PRs are welcome! If you have an idea or spot a bug, feel free to open an issue or submit a pull request.

