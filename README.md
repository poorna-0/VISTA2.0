# VISTA 2.0 - Vision Integrated Surveillance for Threat Analysis

VISTA 2.0 is a real-time smart video surveillance application built with Python and Flask. It uses Ultralytics YOLO models to process video feeds (via webcam, RTSP streams, or uploaded files) to conduct parallel threat analysis.

## Features Supported
1. **Crowd Detection (CD)**: Detects clusters of people and triggers an alert if the local crowd limit exceeds the threshold.
2. **Violence Detection (VD)**: Highlights instances of violence in the frame.
3. **Abandoned Baggage Detection (UBD)**: Detects and counts unattended or abandoned items.
4. **Suspicious Activity Detection (SAD)**: Monitors the frame for abnormal or suspicious activities.

---

## Project Structure
```text
VISTA2.0-main/
│
├── app.py                   # Main Flask application and video processing thread
├── models/                  # YOLO Model weights directory
│   ├── best (CD).pt         # Crowd Detection Model
│   ├── best (VD).pt         # Violence Detection Model
│   ├── best (UBD).pt        # Abandoned Baggage Model
│   └── best (SAD).pt        # Suspicious Activity Model
│
├── templates/               # HTML templates for Flask UI
│   └── index.html           
│
├── uploads/                 # Storage for user-uploaded videos/images
├── uploads_test/            # Storage testing directory
│
├── detection_log.csv        # Log output containing detection states over time
│
├── run_tests.py             # Script to execute project tests
├── test_unit.py             # Unit testing configurations
├── test_integration.py      # Integration testing configurations
├── test_system_perf.py      # System performance monitoring tests
│
└── README.md                # Project documentation
```

---

## Prerequisites & Requirements

This project runs on **Python 3.8+**. Because it evaluates multiple deep learning models concurrently, a machine with a dedicated GPU (CUDA supported) is highly recommended for smooth real-time video processing, though it will run functionally on a CPU.

The primary Python packages required are:
- `Flask` (Web framework and backend)
- `ultralytics` (For running the YOLO inference engine)
- `opencv-python` (Image and video frame manipulation)
- `numpy` (Mathematical and array operations for centroid calculations)

*(Note: The system utilizes standard built-in Python libraries like `os`, `csv`, `threading`, and heavily relies on `winsound` for Windows-specific audio alerts).*

---

## Installation Guide

**1. Navigate to the project directory**
```bash
cd VISTA2.0-main
```

**2. Create a Virtual Environment (Highly Recommended)**
```bash
# Generate the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (MacOS/Linux)
source venv/bin/activate
```

**3. Install Dependencies**
You can install all necessary packages via pip directly:
```bash
pip install Flask ultralytics opencv-python numpy
```

**4. Ensure Models are Present**
Verify that your YOLO weights are physically located inside the `models/` directory exactly with these names:
- `models/best (CD).pt`
- `models/best (VD).pt`
- `models/best (UBD).pt`
- `models/best (SAD).pt`

---

## How To Run

**1. Start the Flask Application**
Run the core python script. The server will launch the backend, load the ML models into memory, and begin intercepting camera feeds.
```bash
python app.py
```
*(Wait until your terminal prints `Models loaded.` usually takes ~5 seconds).*

**2. Open the Dashboard**
Open your web browser and navigate to the local host address: 
[http://localhost:5000](http://localhost:5000)

**3. Interacting with the Feed**
By default, the application attempts to capture video from your primary webcam (`Webcam 0`). Using the web interface features, you can:
- Switch the target camera index.
- Provide an RTSP stream URL to intercept CCTV traffic.
- Upload local images or video fragments to test model inferencing directly.

## Logs
Every status shift within the visual feed is dynamically logged into `detection_log.csv` located at your system root. Feel free to download this file through the UI or access it directly inside the directory to view historical system records.
