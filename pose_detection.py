import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
import time

class PoseDetector:
    def __init__(self, camera_inverted=False):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
        )
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0  # 0=Lite, 1=Full
        )
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True  # Enable iris landmarks
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera orientation setting
        self.camera_inverted = camera_inverted
        
        # Create output directory if it doesn't exist
        self.output_dir = "pose_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize recording variables
        self.is_recording = False
        self.recorded_frames = []
        self.frame_count = 0
        self.last_save_time = 0
        self.save_interval = 0.033  # 30 FPS
        
        # COCO keypoint mapping for MediaPipe
        self.coco_keypoint_indices = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        
        # MediaPipe to COCO keypoint mapping
        self.mp_to_coco = {
            0: 0,   # nose
            1: 1,   # left_eye
            2: 2,   # right_eye
            3: 3,   # left_ear
            4: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16  # right_ankle
        }

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for pose
        pose_results = self.pose.process(rgb_frame)
        
        # Process the frame for hands
        hands_results = self.hands.process(rgb_frame)
        
        # Process the frame for face mesh
        face_mesh_results = self.face_mesh.process(rgb_frame)
        
        # Draw the pose landmarks (upper body only)
        if pose_results.pose_landmarks:
            # Define body connections (excluding face)
            body_connections = [
                # Shoulders to elbows
                (11, 13), (12, 14),
                # Elbows to wrists
                (13, 15), (14, 16)
            ]
            
            # Draw body landmarks and connections
            for connection in body_connections:
                start_idx, end_idx = connection
                start_point = pose_results.pose_landmarks.landmark[start_idx]
                end_point = pose_results.pose_landmarks.landmark[end_idx]
                
                # Convert normalized coordinates to pixel coordinates
                h, w, c = frame.shape
                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)
                
                # Draw connection
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
                # Draw keypoints
                cv2.circle(frame, (start_x, start_y), 4, (0, 0, 255), -1)
                cv2.circle(frame, (end_x, end_y), 4, (0, 0, 255), -1)
            
            # Convert MediaPipe landmarks to COCO format
            coco_keypoints = self.convert_to_coco_format(pose_results.pose_landmarks, frame.shape)
            
            # Save pose data if recording
            if self.is_recording:
                self.save_pose_data(coco_keypoints, frame.shape)
        
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # Get hand type from MediaPipe's handedness detection
                hand_type = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Invert hand type if camera is inverted
                if self.camera_inverted:
                    hand_type = "Right" if hand_type == "Left" else "Left"
                
                # Draw hand landmarks with different colors for left/right
                color = (255, 0, 0) if hand_type == "Left" else (0, 0, 255)  # Blue for left, red for right
                
                # Draw hand connections
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=color, thickness=2)
                )
                
                # Draw hand type and confidence
                h, w, c = frame.shape
                x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * w)
                y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * h)
                text = f"{hand_type} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw face mesh landmarks
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Draw face mesh landmarks
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                
                # Draw iris landmarks
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )
                
                # Draw face contour
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )
        
        return frame

    def convert_to_coco_format(self, landmarks, image_shape):
        height, width = image_shape[:2]
        coco_keypoints = [0] * 51  # 17 keypoints * 3 (x, y, visibility)
        
        for mp_idx, coco_idx in self.mp_to_coco.items():
            if mp_idx < len(landmarks.landmark):
                landmark = landmarks.landmark[mp_idx]
                x = landmark.x * width
                y = landmark.y * height
                visibility = 2.0 if landmark.visibility > 0.5 else 0.0
                
                # Store in COCO format (x, y, visibility)
                base_idx = coco_idx * 3
                coco_keypoints[base_idx] = x
                coco_keypoints[base_idx + 1] = y
                coco_keypoints[base_idx + 2] = visibility
        
        return coco_keypoints

    def save_pose_data(self, keypoints, image_shape):
        current_time = time.time()
        
        # Save at specified interval (30 FPS)
        if current_time - self.last_save_time >= self.save_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"pose_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Calculate bounding box
            visible_points = [(keypoints[i], keypoints[i+1]) 
                            for i in range(0, len(keypoints), 3) 
                            if keypoints[i+2] > 0]
            
            if visible_points:
                x_coords = [x for x, _ in visible_points]
                y_coords = [y for _, y in visible_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                
                # Create COCO format annotation
                annotation = {
                    "image_id": int(time.time() * 1000),
                    "annotation_id": int(time.time() * 1000),
                    "keypoints": keypoints,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "iscrowd": 0,
                    "num_keypoints": sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0),
                    "visible_keypoints": sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
                }
                
                with open(filepath, 'w') as f:
                    json.dump(annotation, f, indent=4)
                
                self.last_save_time = current_time
                self.frame_count += 1

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            print("Started recording pose data...")
            self.frame_count = 0
            self.last_save_time = time.time()
        else:
            print(f"Stopped recording. Saved {self.frame_count} frames.")

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize pose detector with camera orientation setting
    detector = PoseDetector(camera_inverted=True)  # Set this based on your camera orientation
    
    print("Press 'r' to start/stop recording, 'q' to quit")
    print("Press 'i' to toggle camera inversion")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Display recording status
        status = "Recording..." if detector.is_recording else "Not Recording"
        cv2.putText(processed_frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame count
        cv2.putText(processed_frame, f"Frames: {detector.frame_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display camera inversion status
        inversion_status = "Camera Inverted" if detector.camera_inverted else "Camera Normal"
        cv2.putText(processed_frame, inversion_status, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Pose, Hand, and Face Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.toggle_recording()
        elif key == ord('i'):
            detector.camera_inverted = not detector.camera_inverted
            print(f"Camera inversion {'enabled' if detector.camera_inverted else 'disabled'}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 