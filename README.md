# VISTA 2.0 - Vision Integrated Surveillance for Threat Analysis

## Project Structure

```
VISTA2.O/
├── app.py                          # Original integrated Flask app
├── pipeline/
│   ├── __init__.py
│   ├── detection_pipeline.py       # Main pipeline class for parallel processing
│   ├── demo.py                     # Standalone demo script
│   └── integrated_app.py           # Flask app using the pipeline
├── modules/
│   ├── crowd_detection/
│   │   ├── __init__.py
│   │   └── crowd_detector.py       # Crowd detection module
│   ├── violence_detection/
│   │   ├── __init__.py
│   │   └── violence_detector.py    # Violence detection module
│   ├── abandoned_detection/
│   │   ├── __init__.py
│   │   └── abandoned_detector.py   # Abandoned baggage detection module
│   └── suspicious_detection/
│       ├── __init__.py
│       └── suspicious_detector.py  # Suspicious activity detection module
├── saved_models/                   # YOLO model files
├── templates/                      # Flask templates
├── uploads/                        # Uploaded files
├── detection_log.csv               # Detection logs
├── TODO.md                         # Task list
├── TODO_SAD.md                     # SAD-specific tasks
└── README.md                       # This file
```

## Features

- **Crowd Detection (CD)**: Detects people and determines if area is crowded based on proximity
- **Violence Detection (VD)**: Identifies violent activities
- **Abandoned Baggage Detection (UBD)**: Counts abandoned items
- **Suspicious Activity Detection (SAD)**: Flags suspicious behaviors

## Pipeline Architecture

The project uses a modular pipeline architecture for efficient parallel processing:

1. **Individual Modules**: Each detection type has its own module with dedicated detector class
2. **Detection Pipeline**: Orchestrates parallel execution of all detectors using ThreadPoolExecutor
3. **Integrated App**: Flask web application that uses the pipeline for real-time processing

## Usage

### Running the Integrated App
```bash
python pipeline/integrated_app.py
```
Access at http://localhost:5000

### Running the Demo
```bash
python pipeline/demo.py
```

### Using Individual Modules
```python
from modules.crowd_detection.crowd_detector import CrowdDetector
detector = CrowdDetector()
boxes, centroids = detector.detect(frame)
```

## Pipeline Benefits

- **Parallel Processing**: All models run simultaneously using multi-threading
- **Modular Design**: Easy to add/remove/modify detection modules
- **Scalable**: Can handle additional detection types without major refactoring
- **Efficient**: Reduces overall processing time by parallelizing inference tasks
