import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points in 3D space."""
    # Convert points to numpy arrays
    p1 = np.array([p1['x'], p1['y'], p1['z']])
    p2 = np.array([p2['x'], p2['y'], p2['z']])
    p3 = np.array([p3['x'], p3['y'], p3['z']])
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculate the distance between two points in 3D space."""
    p1 = np.array([p1['x'], p1['y'], p1['z']])
    p2 = np.array([p2['x'], p2['y'], p2['z']])
    return np.linalg.norm(p1 - p2)

def validate_pose(landmarks):
    """Validate pose data based on anatomical constraints."""
    if not landmarks:
        return False, "No landmarks detected"
    
    # Check visibility of essential points
    essential_points = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    for point in essential_points:
        if point not in landmarks or landmarks[point]['visibility'] < 0.5:
            return False, f"Essential point {point} not visible enough"
    
    # Check hip width (should be reasonable)
    hip_width = calculate_distance(landmarks['left_hip'], landmarks['right_hip'])
    if hip_width > 0.5:  # Normalized coordinates, so 0.5 is half the frame width
        return False, "Hip width too large"
    
    # Check knee angles (should be between 0 and 180 degrees)
    left_knee_angle = calculate_angle(landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle'])
    right_knee_angle = calculate_angle(landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle'])
    
    if not (0 <= left_knee_angle <= 180 and 0 <= right_knee_angle <= 180):
        return False, "Invalid knee angles"
    
    # Check hip angles (should be between 0 and 180 degrees)
    left_hip_angle = calculate_angle(landmarks['left_shoulder'], landmarks['left_hip'], landmarks['left_knee'])
    right_hip_angle = calculate_angle(landmarks['right_shoulder'], landmarks['right_hip'], landmarks['right_knee'])
    
    if not (0 <= left_hip_angle <= 180 and 0 <= right_hip_angle <= 180):
        return False, "Invalid hip angles"
    
    # Check ankle angles (should be between 0 and 180 degrees)
    left_ankle_angle = calculate_angle(landmarks['left_knee'], landmarks['left_ankle'], landmarks['left_heel'])
    right_ankle_angle = calculate_angle(landmarks['right_knee'], landmarks['right_ankle'], landmarks['right_heel'])
    
    if not (0 <= left_ankle_angle <= 180 and 0 <= right_ankle_angle <= 180):
        return False, "Invalid ankle angles"
    
    # Check limb length ratios (should be roughly equal on both sides)
    left_leg_length = calculate_distance(landmarks['left_hip'], landmarks['left_ankle'])
    right_leg_length = calculate_distance(landmarks['right_hip'], landmarks['right_ankle'])
    
    if abs(left_leg_length - right_leg_length) > 0.1:  # Allow 10% difference
        return False, "Leg length mismatch"
    
    # Check if knees are below hips
    if (landmarks['left_knee']['y'] < landmarks['left_hip']['y'] or 
        landmarks['right_knee']['y'] < landmarks['right_hip']['y']):
        return False, "Knees above hips"
    
    # Check if ankles are below knees
    if (landmarks['left_ankle']['y'] < landmarks['left_knee']['y'] or 
        landmarks['right_ankle']['y'] < landmarks['right_knee']['y']):
        return False, "Ankles above knees"
    
    return True, "Valid pose"

def process_video(video_path, output_dir):
    """Process a video file to extract pose data using MediaPipe Pose."""
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Invalid frame count for video {video_path}")
            return
            
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Initialize lists to store pose data
        pose_data = []
        validation_data = []
        frame_count = 0
        
        # Define the keypoints we want to track
        # MediaPipe Pose has 33 landmarks, we'll focus on the ones we need
        keypoint_indices = {
            # Arms
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            
            # Hips
            'left_hip': 23,
            'right_hip': 24,
            
            # Legs
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        # Process frames with progress bar
        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe Pose
                results = pose.process(frame_rgb)
                
                # Extract pose landmarks
                if results.pose_landmarks:
                    frame_landmarks = {}
                    for name, idx in keypoint_indices.items():
                        landmark = results.pose_landmarks.landmark[idx]
                        frame_landmarks[name] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        }
                    
                    # Validate pose
                    is_valid, validation_message = validate_pose(frame_landmarks)
                    
                    if is_valid:
                        pose_data.append(frame_landmarks)
                    else:
                        pose_data.append(None)
                    
                    validation_data.append({
                        'frame': frame_count,
                        'is_valid': is_valid,
                        'message': validation_message
                    })
                else:
                    pose_data.append(None)
                    validation_data.append({
                        'frame': frame_count,
                        'is_valid': False,
                        'message': 'No pose detected'
                    })
                
                pbar.update(1)
        
        # Calculate detection and validation rates
        total_frames = len(pose_data)
        pose_detection_rate = sum(1 for x in pose_data if x is not None) / total_frames * 100
        valid_pose_rate = sum(1 for x in validation_data if x['is_valid']) / total_frames * 100
        
        print(f"\nDetection rates for {os.path.basename(video_path)}:")
        print(f"Pose detection rate: {pose_detection_rate:.2f}%")
        print(f"Valid pose rate: {valid_pose_rate:.2f}%")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save pose data
        output_file = os.path.join(output_dir, f"pose_data_{os.path.splitext(os.path.basename(video_path))[0]}.json")
        with open(output_file, 'w') as f:
            json.dump({
                'fps': fps,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'total_frames': total_frames,
                'processed_frames': frame_count,
                'pose_landmarks': pose_data,
                'validation_data': validation_data,
                'detection_rate': pose_detection_rate,
                'valid_pose_rate': valid_pose_rate,
                'keypoint_indices': keypoint_indices,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        print(f"\nPose data saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        if 'pose' in locals():
            pose.close()

def main():
    # Directory containing videos
    video_dir = "videos"
    output_dir = "pose_data"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all videos in the directory
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mpg', '.mpeg', '.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in the 'videos' directory: {os.path.abspath(video_dir)}")
        print("Supported formats: .mpg, .mpeg, .mp4, .avi, .mov")
        return
    
    print(f"Found {len(video_files)} video files:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {video}")
    
    for filename in video_files:
        video_path = os.path.join(video_dir, filename)
        process_video(video_path, output_dir)
    
    print("\nAll videos processed!")

if __name__ == "__main__":
    main() 