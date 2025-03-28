# Gait Re-Identification Project

This project implements human pose detection and video processing using MediaPipe for real-time and batch processing of videos. It includes functionality for detecting body poses, hand movements, and facial landmarks.

## Features

- Real-time pose detection using MediaPipe
- Hand tracking with left/right hand identification
- Facial landmark detection with iris tracking
- Video processing capabilities
- Support for COCO-VID dataset integration
- Configurable camera orientation handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Gait-Re-Identification.git
cd Gait-Re-Identification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Real-time Pose Detection

Run the real-time pose detection script:
```bash
python pose_detection.py
```

Controls:
- Press 'r' to start/stop recording
- Press 'i' to toggle camera inversion
- Press 'q' to quit

### Video Processing

1. Place your videos in the `videos` directory
2. Run the video processing script:
```bash
python process_videos.py
```

The processed videos will be saved in the `outputs` directory.

### COCO-VID Dataset Integration

1. Download the COCO-VID dataset:
```bash
python download_coco_vid.py
```

2. Extract human videos from the dataset:
```bash
python extract_human_videos.py
```

The extracted human videos will be saved in the `human_videos` directory.

## Project Structure

```
Gait-Re-Identification/
├── pose_detection.py      # Real-time pose detection
├── process_videos.py      # Video processing script
├── download_coco_vid.py   # COCO dataset downloader
├── extract_human_videos.py # Human video extractor
├── requirements.txt       # Project dependencies
├── videos/               # Input videos directory
├── outputs/              # Processed videos directory
├── pose_data/           # Pose detection data
├── human_videos/        # Extracted human videos
└── coco_vid_dataset/    # COCO dataset directory
```

## Dependencies

- opencv-python
- mediapipe
- numpy
- tqdm
- requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for providing the pose detection and tracking models
- COCO dataset for providing the video dataset
- OpenCV for video processing capabilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/Gait-Re-Identification