import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import json
from datetime import datetime
import os

# Check if GPU is available
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device Name:", tf.test.gpu_device_name())

# Load the MoveNet model
try:
    model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
    movenet = model.signatures['serving_default']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

KEYPOINT_DICT = {
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

# Dictionary to define the skeleton connections
SKELETON_CONNECTIONS = [
    ('left_ear', 'left_eye'), ('left_eye', 'nose'), ('nose', 'right_eye'),
    ('right_eye', 'right_ear'), ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'), ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]

def process_image(image):
    # Resize and pad the image to keep the aspect ratio and fit the expected size
    input_size = 192  # MoveNet requires 192x192 input
    input_img = cv2.resize(image, (input_size, input_size))
    input_img = tf.cast(input_img, dtype=tf.int32)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points.
    Points are in [y, x, confidence] format.
    """
    if point1[2] < 0.3 or point2[2] < 0.3 or point3[2] < 0.3:
        return None
    
    # Convert to vectors
    vector1 = [point1[1] - point2[1], point1[0] - point2[0]]  # Using x,y format for vectors
    vector2 = [point3[1] - point2[1], point3[0] - point2[0]]
    
    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    
    # Calculate magnitudes
    magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
    
    # Calculate angle in degrees
    try:
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        return np.degrees(angle)
    except:
        return None

def draw_joint_angles(frame, keypoints, confidence_threshold=0.3):
    """
    Calculate and draw angles for major joints.
    """
    height, width = frame.shape[:2]
    shaped_keypoints = np.squeeze(keypoints)
    
    # Define joints to calculate angles for
    joints_to_measure = [
        # Left arm
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        # Right arm
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        # Left leg
        ('left_hip', 'left_knee', 'left_ankle'),
        # Right leg
        ('right_hip', 'right_knee', 'right_ankle'),
    ]
    
    for joint_set in joints_to_measure:
        p1_idx = KEYPOINT_DICT[joint_set[0]]
        p2_idx = KEYPOINT_DICT[joint_set[1]]
        p3_idx = KEYPOINT_DICT[joint_set[2]]
        
        p1 = shaped_keypoints[p1_idx]
        p2 = shaped_keypoints[p2_idx]
        p3 = shaped_keypoints[p3_idx]
        
        angle = calculate_angle(p1, p2, p3)
        
        if angle is not None:
            # Convert normalized coordinates to pixel coordinates
            x = int(p2[1] * width)
            y = int(p2[0] * height)
            
            # Draw the angle
            cv2.putText(frame, f'{angle:.1f}°', (x - 30, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def draw_keypoints_and_skeleton(frame, keypoints, confidence_threshold=0.3):
    height, width = frame.shape[:2]
    shaped_keypoints = np.squeeze(keypoints)

    # Draw skeleton connections first
    for connection in SKELETON_CONNECTIONS:
        start_point_idx = KEYPOINT_DICT[connection[0]]
        end_point_idx = KEYPOINT_DICT[connection[1]]
        
        start_point = shaped_keypoints[start_point_idx]
        end_point = shaped_keypoints[end_point_idx]
        
        if start_point[2] > confidence_threshold and end_point[2] > confidence_threshold:
            start_x = int(start_point[1] * width)
            start_y = int(start_point[0] * height)
            end_x = int(end_point[1] * width)
            end_y = int(end_point[0] * height)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Draw keypoints on top
    for idx, kp in enumerate(shaped_keypoints):
        y, x, confidence = kp
        if confidence > confidence_threshold:
            x_px = int(x * width)
            y_px = int(y * height)
            cv2.circle(frame, (x_px, y_px), 4, (0, 0, 255), -1)
    
    # Draw joint angles
    draw_joint_angles(frame, keypoints, confidence_threshold)

def save_pose_data(keypoints, timestamp, output_dir="pose_data"):
    """
    Save the detected pose keypoints to a JSON file.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format the data
    pose_data = {
        "timestamp": timestamp,
        "keypoints": {},
        "angles": {}
    }
    
    shaped_keypoints = np.squeeze(keypoints)
    
    # Save keypoints
    for name, idx in KEYPOINT_DICT.items():
        y, x, confidence = shaped_keypoints[idx]
        pose_data["keypoints"][name] = {
            "x": float(x),
            "y": float(y),
            "confidence": float(confidence)
        }
    
    # Calculate and save angles
    joints_to_measure = [
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('right_hip', 'right_knee', 'right_ankle'),
    ]
    
    for joint_set in joints_to_measure:
        p1_idx = KEYPOINT_DICT[joint_set[0]]
        p2_idx = KEYPOINT_DICT[joint_set[1]]
        p3_idx = KEYPOINT_DICT[joint_set[2]]
        
        p1 = shaped_keypoints[p1_idx]
        p2 = shaped_keypoints[p2_idx]
        p3 = shaped_keypoints[p3_idx]
        
        angle = calculate_angle(p1, p2, p3)
        if angle is not None:
            pose_data["angles"][f"{joint_set[1]}_angle"] = float(angle)
    
    # Generate filename with timestamp
    filename = f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(pose_data, f, indent=4)
    
    return filepath

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(2)  # Try 0 first
    if not cap.isOpened():
        print("Trying alternative webcam index...")
        cap = cv2.VideoCapture(0)  # Try index 2 if 0 doesn't work
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
    
    # Initialize FPS calculation
    prev_time = 0
    fps = 0
    recording = False
    
    print("Starting pose detection. Press 'q' to quit, 'r' to toggle recording.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        try:
            # Process the image for the model
            input_tensor = process_image(frame)
            
            # Run inference
            results = movenet(input=input_tensor)
            keypoints = results['output_0']
            
            # Draw the keypoints and skeleton
            draw_keypoints_and_skeleton(frame, keypoints)
            
            # Save pose data if recording is enabled
            if recording:
                save_pose_data(keypoints, current_time)
                # Display recording indicator
                cv2.putText(frame, 'Recording', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display FPS
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('MoveNet Pose Detection', frame)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            break
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 