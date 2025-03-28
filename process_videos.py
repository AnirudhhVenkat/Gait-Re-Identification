import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import subprocess

class VideoProcessor:
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

def process_video(input_path, output_path, camera_inverted=False):
    # Initialize video processor with camera orientation setting
    processor = VideoProcessor(camera_inverted=camera_inverted)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize ffmpeg process for writing output video
    command = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        
        # Process frames with progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = processor.process_frame(frame)
                
                # Write frame to ffmpeg process
                process.stdin.write(processed_frame.tobytes())
                pbar.update(1)
        
        # Close ffmpeg process
        process.stdin.close()
        process.wait()
        
    except Exception as e:
        print(f"Error processing video: {e}")
        if 'process' in locals():
            process.stdin.close()
            process.wait()
    
    finally:
        # Clean up
        cap.release()

def main():
    # Create videos directory if it doesn't exist
    if not os.path.exists("videos"):
        os.makedirs("videos")
        print("Created 'videos' directory. Please add your videos to this directory.")
        return
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # Get list of video files
    video_files = [f for f in os.listdir("videos") if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print("No video files found in the 'videos' directory.")
        return
    
    print(f"Found {len(video_files)} video files to process.")
    
    # Ask user about camera orientation
    while True:
        response = input("Is the camera view inverted? (y/n): ").lower()
        if response in ['y', 'n']:
            camera_inverted = response == 'y'
            break
        print("Please enter 'y' for yes or 'n' for no.")
    
    # Process each video
    for video_file in video_files:
        input_path = os.path.join("videos", video_file)
        output_path = os.path.join("outputs", f"processed_{video_file}")
        
        print(f"\nProcessing {video_file}...")
        process_video(input_path, output_path, camera_inverted)
        print(f"Completed processing {video_file}")

if __name__ == "__main__":
    main() 