# Gait Re-Identification System

A system for identifying individuals based on their gait patterns using 3D pose data.

## Overview

This project implements a gait recognition system that:
1. Processes 3D pose data from JSON files
2. Extracts meaningful gait features
3. Creates a vector database for efficient similarity search
4. Evaluates matching accuracy using weighted cosine similarity

## Features

The system extracts 8 key gait features:
1. Physical Features:
   - Step width
   - Hip width
   - Left hip angle
   - Right hip angle
2. Dynamic Features:
   - Mean step length
   - Step length standard deviation
   - Mean stride length
   - Stride length standard deviation

## Optimal Window Size

After extensive analysis, the optimal window size was determined to be 60 frames (0.6 seconds at 100 FPS) because:
- Captures one complete step cycle
- Provides best balance between:
  - Number of sequences (67,803)
  - Feature importance scores
  - Feature stability
- Most computationally efficient while maintaining accuracy

## Feature Importance

Based on mutual information analysis, the most important features are:
1. Left hip angle (0.0622)
2. Right hip angle (0.0516)
3. Step width (0.0254)
4. Mean stride length (0.0218)
5. Stride length standard deviation (0.0176)
6. Hip width (0.0086)
7. Mean step length (0.0083)
8. Step length standard deviation (0.0076)

## Directory Structure

```
.
├── converted_poses/         # Input pose data in JSON format
├── vector_db/              # Vector database and analysis results
│   ├── window_60/          # Results for optimal 60-frame window
│   │   ├── vector_database.pt
│   │   ├── metrics.txt
│   │   └── feature_analysis/
│   └── window_analysis/    # Analysis of different window sizes
├── create_vector_db.py     # Script to create vector database
├── evaluate_similarity.py  # Script to evaluate matching accuracy
└── analyze_features.py     # Script for feature analysis
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Create vector database:
```bash
python3 create_vector_db.py
```

2. Evaluate similarity:
```bash
python3 evaluate_similarity.py
```

## Results

The system achieves high accuracy in gait recognition by:
- Using weighted cosine similarity based on feature importance
- Optimizing window size for feature extraction
- Leveraging GPU acceleration for efficient processing
- Implementing robust feature normalization

## Future Improvements

1. Implement temporal features for better gait cycle analysis
2. Add more sophisticated feature weighting schemes
3. Explore deep learning approaches for feature extraction
4. Improve handling of varying walking speeds
5. Add support for real-time gait recognition

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