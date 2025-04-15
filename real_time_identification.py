import cv2
import numpy as np
import torch
import pickle
import os
from collections import deque
from typing import Dict, Any, List, Tuple
import time
from dataclasses import dataclass
import mediapipe as mp
from extract_gait_cycles import detect_gait_cycles, analyze_cycle, detect_gait_events, GaitEvent
from create_vector_db import extract_gait_features, calculate_physical_features, calculate_angles, calculate_step_length, calculate_stride_length

@dataclass
class GaitEvent:
    type: str  # 'heel_strike' or 'mid_swing'
    foot: str  # 'right' or 'left'
    frame: int
    height: float

class RealTimeGaitIdentification:
    def __init__(self, vector_db_path: str, window_size: int = 120):
        """Initialize the real-time gait identification system."""
        # Load vector database
        self.train_features = torch.load(os.path.join(vector_db_path, 'train_features.pt'))
        with open(os.path.join(vector_db_path, 'train_metadata.pkl'), 'rb') as f:
            self.train_metadata = pickle.load(f)
        with open(os.path.join(vector_db_path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose buffer
        self.pose_buffer = deque(maxlen=window_size)
        self.window_size = window_size
        
        # Initialize results buffer
        self.results_buffer = deque(maxlen=30)  # Keep last 30 predictions
        
        # Initialize gait cycle tracking
        self.current_cycle = []
        self.cycle_events = []
        self.right_heights = []
        self.left_heights = []
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
    
    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate the angle between three points in degrees."""
        # Convert points to numpy arrays if they aren't already
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate dot product
        dot_product = np.dot(ba, bc)
        
        # Calculate magnitudes
        ba_mag = np.linalg.norm(ba)
        bc_mag = np.linalg.norm(bc)
        
        # Calculate angle in radians
        cos_angle = dot_product / (ba_mag * bc_mag)
        # Ensure cos_angle is within valid range [-1, 1]
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """Process a single frame and return pose data and annotated frame."""
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, annotated_frame
        
        # Get the pose landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Extract only the keypoints we need for gait analysis
        pose_data = {
            'right_hip': {
                'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * self.width,
                'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * self.height
            },
            'left_hip': {
                'x': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * self.width,
                'y': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * self.height
            },
            'right_knee': {
                'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * self.width,
                'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * self.height
            },
            'left_knee': {
                'x': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x * self.width,
                'y': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y * self.height
            },
            'right_ankle': {
                'x': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * self.width,
                'y': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * self.height
            },
            'left_ankle': {
                'x': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * self.width,
                'y': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * self.height
            }
        }
        
        # Store ankle heights for gait cycle detection
        self.right_heights.append(pose_data['right_ankle']['y'])
        self.left_heights.append(pose_data['left_ankle']['y'])
        
        # Draw only the keypoints we're using
        connections = [
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        ]
        
        # Draw connections
        for connection in connections:
            start_point = landmarks[connection[0]]
            end_point = landmarks[connection[1]]
            cv2.line(annotated_frame,
                    (int(start_point.x * self.width), int(start_point.y * self.height)),
                    (int(end_point.x * self.width), int(end_point.y * self.height)),
                    (0, 255, 0), 2)
        
        # Draw keypoints
        for landmark in [self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE,
                        self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HIP,
                        self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE]:
            point = landmarks[landmark]
            cv2.circle(annotated_frame,
                      (int(point.x * self.width), int(point.y * self.height)),
                      5, (0, 0, 255), -1)
        
        return pose_data, annotated_frame
    
    def detect_current_cycle(self) -> Tuple[List[Dict[str, Any]], List[GaitEvent]]:
        """Detect gait cycles in the current buffer."""
        if len(self.pose_buffer) < 30:  # Minimum frames for a cycle
            return [], []
        
        # Convert buffer to list for processing
        pose_list = list(self.pose_buffer)
        
        # Detect gait events
        events, right_heights, left_heights = detect_gait_events(pose_list)
        
        if not events:
            return [], []
        
        # Find right heel strikes to use as cycle boundaries
        right_heel_strikes = [e for e in events if e.type == 'heel_strike' and e.foot == 'right']
        
        if len(right_heel_strikes) < 2:
            return [], []
        
        # Get the most recent complete cycle
        start_idx = right_heel_strikes[-2].frame
        end_idx = right_heel_strikes[-1].frame
        
        cycle = pose_list[start_idx:end_idx]
        cycle_events = [e for e in events if start_idx <= e.frame < end_idx]
        
        return cycle, cycle_events
    
    def analyze_current_cycle(self, cycle: List[Dict[str, Any]], events: List[GaitEvent]) -> Dict[str, float]:
        """Analyze the current gait cycle."""
        if not cycle or not events:
            return None
        
        # Calculate cycle statistics
        stats = analyze_cycle(cycle, events)
        
        # Extract features
        features = self.extract_gait_features(cycle, events)
        
        return {
            'stats': stats,
            'features': features
        }
    
    def extract_gait_features(self, cycle: List[Dict[str, Any]], events: List[GaitEvent]) -> Dict[str, float]:
        """Extract gait features from a complete cycle."""
        if not cycle:
            return {}
        
        # Calculate temporal features
        cycle_length = len(cycle)
        
        # Calculate spatial features
        right_ankle_positions = np.array([[p['right_ankle']['x'], p['right_ankle']['y']] for p in cycle])
        left_ankle_positions = np.array([[p['left_ankle']['x'], p['left_ankle']['y']] for p in cycle])
        right_hip_positions = np.array([[p['right_hip']['x'], p['right_hip']['y']] for p in cycle])
        left_hip_positions = np.array([[p['left_hip']['x'], p['left_hip']['y']] for p in cycle])
        
        # Calculate step width (horizontal distance between ankles)
        step_width = np.mean(np.abs(right_ankle_positions[:, 0] - left_ankle_positions[:, 0]))
        
        # Calculate hip width
        hip_width = np.mean(np.abs(right_hip_positions[:, 0] - left_hip_positions[:, 0]))
        
        # Calculate stride length (distance between consecutive heel strikes)
        stride_length = 0
        if len(events) >= 2:
            heel_strikes = [e for e in events if e.type == 'heel_strike']
            if len(heel_strikes) >= 2:
                # Get the first two heel strikes
                hs1, hs2 = heel_strikes[0], heel_strikes[1]
                # Ensure frame indices are within cycle bounds
                if 0 <= hs1.frame < len(cycle) and 0 <= hs2.frame < len(cycle):
                    pos1 = np.array([cycle[hs1.frame]['right_ankle']['x'], 
                                   cycle[hs1.frame]['right_ankle']['y']])
                    pos2 = np.array([cycle[hs2.frame]['right_ankle']['x'], 
                                   cycle[hs2.frame]['right_ankle']['y']])
                    stride_length = np.linalg.norm(pos2 - pos1)
        
        # Calculate foot clearance (maximum height difference during swing)
        right_clearance = np.max(right_ankle_positions[:, 1]) - np.min(right_ankle_positions[:, 1])
        left_clearance = np.max(left_ankle_positions[:, 1]) - np.min(left_ankle_positions[:, 1])
        
        # Calculate knee angles
        right_knee_angles = []
        left_knee_angles = []
        for pose in cycle:
            # Right knee angle
            hip = np.array([pose['right_hip']['x'], pose['right_hip']['y']])
            knee = np.array([pose['right_knee']['x'], pose['right_knee']['y']])
            ankle = np.array([pose['right_ankle']['x'], pose['right_ankle']['y']])
            right_knee_angles.append(self.calculate_angle(hip, knee, ankle))
            
            # Left knee angle
            hip = np.array([pose['left_hip']['x'], pose['left_hip']['y']])
            knee = np.array([pose['left_knee']['x'], pose['left_knee']['y']])
            ankle = np.array([pose['left_ankle']['x'], pose['left_ankle']['y']])
            left_knee_angles.append(self.calculate_angle(hip, knee, ankle))
        
        # Calculate average knee angles
        avg_right_knee_angle = np.mean(right_knee_angles)
        avg_left_knee_angle = np.mean(left_knee_angles)
        
        return {
            'cycle_length': cycle_length,
            'step_width': step_width,
            'hip_width': hip_width,
            'stride_length': stride_length,
            'right_foot_clearance': right_clearance,
            'left_foot_clearance': left_clearance,
            'avg_right_knee_angle': avg_right_knee_angle,
            'avg_left_knee_angle': avg_left_knee_angle
        }
    
    def identify_gait(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Identify gait by comparing features with the vector database."""
        # Convert features dictionary to numpy array
        feature_names = [
            'cycle_length', 'step_width', 'hip_width', 'stride_length',
            'right_foot_clearance', 'left_foot_clearance',
            'avg_right_knee_angle', 'avg_left_knee_angle'
        ]
        features_array = np.array([features.get(name, 0.0) for name in feature_names])
        
        # Normalize features
        features_norm = self.scaler.transform(features_array.reshape(1, -1))
        features_tensor = torch.from_numpy(features_norm).float()
        
        # Calculate similarities
        similarities = torch.mm(features_tensor, self.train_features.t())
        
        # Get top matches
        top_k = 5
        _, top_indices = torch.topk(similarities, top_k)
        
        # Get subject IDs and confidence scores
        subject_ids = [self.train_metadata['subject_ids'][i] for i in top_indices[0]]
        scores = similarities[0][top_indices[0]].numpy()
        
        # Calculate confidence (normalize scores)
        confidence = scores / scores.sum()
        
        return {
            'subjects': subject_ids,
            'confidence': confidence
        }
    
    def draw_results(self, frame: np.ndarray, results: Dict[str, Any], cycle_stats: Dict[str, float]) -> np.ndarray:
        """Draw identification results and gait cycle information on the frame."""
        if not results:
            return frame
        
        # Draw top matches
        y = 30
        for subject, conf in zip(results['subjects'], results['confidence']):
            text = f"Subject {subject}: {conf:.2f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
        
        # Draw gait cycle information
        if cycle_stats:
            y += 20
            cv2.putText(frame, "Gait Cycle Stats:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 30
            
            stats = cycle_stats['stats']
            features = cycle_stats['features']
            cv2.putText(frame, f"Cycle Length: {stats['cycle_length']} frames", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y += 30
            cv2.putText(frame, f"Step Width: {features['step_width']:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y += 30
            cv2.putText(frame, f"Hip Width: {features['hip_width']:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Run the real-time identification system."""
        print("Starting real-time gait identification...")
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            pose_data, annotated_frame = self.process_frame(frame)
            
            if pose_data:
                # Add to buffer
                self.pose_buffer.append(pose_data)
                
                # Detect and analyze current gait cycle
                cycle, events = self.detect_current_cycle()
                if cycle and events:
                    # Analyze cycle
                    cycle_analysis = self.analyze_current_cycle(cycle, events)
                    
                    if cycle_analysis:
                        # Identify gait
                        results = self.identify_gait(cycle_analysis['features'])
                        
                        # Add to results buffer
                        self.results_buffer.append((results, cycle_analysis))
                        
                        # Get most common prediction
                        if self.results_buffer:
                            all_subjects = [r[0]['subjects'][0] for r in self.results_buffer]
                            most_common = max(set(all_subjects), key=all_subjects.count)
                            confidence = np.mean([r[0]['confidence'][0] for r in self.results_buffer])
                            print(f"Identified as Subject {most_common} with confidence {confidence:.2f}")
                            print(f"Cycle Length: {cycle_analysis['stats']['cycle_length']} frames")
            
            # Draw results
            if self.results_buffer:
                results, cycle_analysis = self.results_buffer[-1]
                annotated_frame = self.draw_results(annotated_frame, results, cycle_analysis)
            
            # Show frame
            cv2.imshow('Gait Identification', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Set up paths
    vector_db_path = "vector_db"
    
    # Initialize and run the system
    identifier = RealTimeGaitIdentification(vector_db_path)
    identifier.run()

if __name__ == "__main__":
    main() 