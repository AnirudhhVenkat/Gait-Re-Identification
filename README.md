# Gait Re-Identification System

A real-time pose detection system using TensorFlow and MoveNet for gait analysis and re-identification.

## Features

- Real-time pose detection using MoveNet
- Joint angle calculation and visualization
- Pose data recording and saving to JSON files
- Support for both webcam inputs
- FPS monitoring

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- TensorFlow Hub
- NumPy

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python pose_detection.py
```

### Controls
- Press 'r' to toggle pose recording (saves to 'pose_data' directory)
- Press 'q' to quit the application

### Output
The system displays:
- Real-time skeleton overlay
- Joint angles for arms and legs
- Current FPS
- Recording status (when active)

Recorded poses are saved in JSON format with:
- Timestamp
- Keypoint coordinates
- Joint angles
- Confidence scores

## Data Format

Saved pose data is structured as follows:
```json
{
    "timestamp": 1234567890.123,
    "keypoints": {
        "nose": {"x": 0.5, "y": 0.5, "confidence": 0.9},
        ...
    },
    "angles": {
        "left_elbow_angle": 90.5,
        "right_elbow_angle": 85.3,
        ...
    }
}
```