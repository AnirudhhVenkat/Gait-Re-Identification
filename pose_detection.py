import cv2
import numpy as np
import json
import time
from datetime import datetime
import os
import mediapipe as mp

class PoseDetector:
    def __init__(self, camera_id=0, invert_camera=False):
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # Initialize segmentation for person detection
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0=general, 1=landscape (faster)
        )
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.7,
            model_complexity=1,  # Using the most accurate model
            enable_segmentation=True,  # Enable segmentation for better occlusion handling
            smooth_landmarks=True,  # Enable smoothing for better tracking
            static_image_mode=False  # Video mode for better performance
        )
        
        print(f"Initializing camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            available_ports = []
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_ports.append(i)
                    cap.release()
            if available_ports:
                raise ValueError(f"Could not open camera {camera_id}. Available camera ports are: {available_ports}")
            else:
                raise ValueError("No cameras found. Please check your camera connection.")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
        
        # Initialize video writer
        self.output_dir = "output_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_writer = None
        self.is_recording = False
        
        # Initialize frame counter and time tracking
        self.frame_count = 0
        self.start_time = None
        
        # Camera inversion flag
        self.invert_camera = invert_camera
        
        # Define keypoints for tracking (MediaPipe format)
        self.keypoints = {
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_heel': self.mp_pose.PoseLandmark.LEFT_HEEL,
            'right_heel': self.mp_pose.PoseLandmark.RIGHT_HEEL,
            'left_foot_index': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            'right_foot_index': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        }
        
        # Colors for visualization
        self.colors = {
            'keypoints': (0, 255, 0),  # Green for keypoints
            'connections': (255, 0, 0),  # Red for connections
            'foot_points': (0, 0, 255),  # Blue for foot points
            'text': (255, 255, 255),  # White for text
            'background': (0, 0, 0)  # Black for background
        }
        
        # Initialize tracking variables
        self.last_positions = {key: None for key in self.keypoints.keys()}
        self.trajectories = {key: [] for key in self.keypoints.keys()}
    
        
        # Initialize occlusion handling
        self.occlusion_history = {key: [] for key in self.keypoints.keys()}
        self.occlusion_window = 5  # Number of frames to consider for occlusion detection
        self.position_history = {key: [] for key in self.keypoints.keys()}
        self.history_window = 10  # Number of frames to keep in position history
        
        # Initialize person detection
        self.person_mask = None
        self.person_detected = False
        self.person_confidence_threshold = 0.5  # Threshold for person detection
        
        print("Initialization complete. Press 'q' to quit, 'r' to start/stop recording.")

    def detect_person(self, frame_rgb):
        """Detect person using segmentation"""
        # Get segmentation mask
        results = self.segmentation.process(frame_rgb)
        if results.segmentation_mask is not None:
            # Convert mask to binary
            mask = results.segmentation_mask > self.person_confidence_threshold
            mask = mask.astype(np.uint8) * 255
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Check if person is detected
            self.person_detected = np.any(mask > 0)
            self.person_mask = mask
            
            return self.person_detected
        return False

    def handle_occlusion(self, landmarks, keypoint_name):
        """Handle occlusion by using historical data and interpolation"""
        current_landmark = landmarks[self.keypoints[keypoint_name]]
        current_visibility = current_landmark.visibility
        
        # Get keypoint position
        x, y = current_landmark.x, current_landmark.y
        
        # Check if keypoint is in person mask
        if self.person_mask is not None:
            pixel_x = int(x * self.width)
            pixel_y = int(y * self.height)
            if 0 <= pixel_x < self.width and 0 <= pixel_y < self.height:
                if self.person_mask[pixel_y, pixel_x] == 0:
                    # Point is outside person mask, consider it occluded
                    current_visibility = 0
        
        # Update occlusion history
        self.occlusion_history[keypoint_name].append(current_visibility)
        if len(self.occlusion_history[keypoint_name]) > self.occlusion_window:
            self.occlusion_history[keypoint_name].pop(0)
        
        # Update position history
        if current_visibility > 0.5:  # Only update history with good visibility
            self.position_history[keypoint_name].append((x, y))
            if len(self.position_history[keypoint_name]) > self.history_window:
                self.position_history[keypoint_name].pop(0)
        
        # Check if we're in an occlusion
        is_occluded = current_visibility < 0.5
        if is_occluded and len(self.position_history[keypoint_name]) >= 2:
            # Use the last known good position
            last_good_pos = self.position_history[keypoint_name][-1]
            return True, last_good_pos
        
        return False, (x, y)

    def process_frame(self, frame):
        if self.invert_camera:
            frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect person
        person_detected = self.detect_person(frame_rgb)
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # If no person detected, return frame with message
        if not person_detected:
            cv2.putText(annotated_frame, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
            return annotated_frame
        
        # Process the frame and detect pose
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Process and draw keypoints with trajectories
            landmarks = results.pose_landmarks.landmark
            
            # Draw connections between keypoints first (so they appear behind the points)
            connections = [
                ('left_hip', 'left_knee'),
                ('right_hip', 'right_knee'),
                ('left_knee', 'left_ankle'),
                ('right_knee', 'right_ankle'),
                ('left_hip', 'right_hip'),
                ('left_ankle', 'left_heel'),
                ('right_ankle', 'right_heel'),
                ('left_heel', 'left_foot_index'),
                ('right_heel', 'right_foot_index')
            ]
            
            for start_point, end_point in connections:
                start_landmark = landmarks[self.keypoints[start_point]]
                end_landmark = landmarks[self.keypoints[end_point]]
                
                # Handle potential occlusions for both points
                start_occluded, start_pos = self.handle_occlusion(landmarks, start_point)
                end_occluded, end_pos = self.handle_occlusion(landmarks, end_point)
                
                # Only draw if at least one point is visible
                if not (start_occluded and end_occluded):
                    start_x = int(start_pos[0] * self.width)
                    start_y = int(start_pos[1] * self.height)
                    end_x = int(end_pos[0] * self.width)
                    end_y = int(end_pos[1] * self.height)
                    cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), self.colors['connections'], 2)
            
            # Draw keypoints and trajectories
            for name, idx in self.keypoints.items():
                landmark = landmarks[idx]
                is_occluded, pos = self.handle_occlusion(landmarks, name)
                
                if not is_occluded or len(self.position_history[name]) >= 2:
                    x = int(pos[0] * self.width)
                    y = int(pos[1] * self.height)
                    
     
                    # Draw keypoint
                    color = self.colors['foot_points'] if 'foot' in name or 'heel' in name else self.colors['keypoints']
                    cv2.circle(annotated_frame, (x, y), 6, color, -1)
        
        # Add frame counter
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        if self.is_recording:
            cv2.putText(annotated_frame, "Recording", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return annotated_frame

    def save_keypoints(self, keypoints_data):
        """Save keypoints data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"keypoints_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(keypoints_data, f, indent=4)
        
        print(f"Keypoints saved to {filename}")

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Record if active
                if self.is_recording:
                    if self.video_writer is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(self.output_dir, f"pose_{timestamp}.mp4")
                        self.video_writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (self.width, self.height)
                        )
                    self.video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Pose Detection', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not self.is_recording:
                        self.is_recording = True
                        self.start_time = time.time()
                        print("Recording started")
                    else:
                        self.is_recording = False
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        print("Recording stopped")
                
                self.frame_count += 1
                
        finally:
            # Cleanup
            self.pose.close()
            if self.video_writer:
                self.video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pose Detection with MediaPipe')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--invert', action='store_true', help='Invert camera horizontally')
    
    args = parser.parse_args()
    
    detector = PoseDetector(camera_id=args.camera, invert_camera=args.invert)
    detector.run() 