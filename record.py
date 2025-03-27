import cv2
import os
import torch
import numpy as np
from datetime import datetime
import sys
import os.path as osp
import signal

# Add GAST-Net paths
sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
from tools.utils import get_path
from model.gast_net import SpatioTemporalModelOptimized1f
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from tools.preprocess import h36m_coco_format, revise_kpts
from tools.inference import gen_pose
from tools.vis_h36m import render_animation

# Get paths
cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
model_dir = chk_root + 'gastnet/'

sys.path.insert(1, lib_root)
from lib.pose import gen_video_kpts as hrnet_pose
sys.path.pop(1)
sys.path.pop(0)

# Global flag for graceful exit
running = True

def signal_handler(signum, frame):
    global running
    running = False

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Check GPU availability and print detailed info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check for multiple GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
    print(f"\nCUDA Version: {torch.version.cuda}")
    
    # Select GPU with most available memory
    max_memory = 0
    selected_gpu = 0
    for i in range(num_gpus):
        memory = torch.cuda.get_device_properties(i).total_memory
        if memory > max_memory:
            max_memory = memory
            selected_gpu = i
    
    print(f"\nSelected GPU {selected_gpu} with {max_memory/1024**3:.2f} GB memory")
    torch.cuda.set_device(selected_gpu)
else:
    print("No CUDA GPUs found. Using CPU.")

# Initialize skeleton
skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
adj = adj_mx_from_skeleton(skeleton)

# Initialize keypoints metadata
keypoints_metadata = {
    'keypoints_symmetry': ([4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]),
    'layout_name': 'Human3.6M',
    'num_joints': 17
}

def draw_skeleton(frame, keypoints_2d, color=(0, 255, 0), thickness=2):
    """Draw 2D skeleton on frame"""
    # Define connections between joints
    connections = [
        (0, 1), (1, 2), (2, 3),  # Right arm
        (0, 4), (4, 5), (5, 6),  # Left arm
        (0, 7), (7, 8), (8, 9),  # Spine
        (9, 10),  # Head
        (8, 11), (11, 12), (12, 13),  # Right leg
        (8, 14), (14, 15), (15, 16)   # Left leg
    ]
    
    # Draw connections
    for connection in connections:
        start_point = tuple(map(int, keypoints_2d[connection[0]]))
        end_point = tuple(map(int, keypoints_2d[connection[1]]))
        cv2.line(frame, start_point, end_point, color, thickness)
    
    # Draw joints
    for joint in keypoints_2d:
        point = tuple(map(int, joint))
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
    
    return frame

# Load GAST-Net model
def load_model_realtime(rf=27):
    print('Loading GAST-Net...')
    chk = model_dir + '27_frame_model.bin'  # Using the available model file
    filters_width = [3, 3, 3]
    channels = 128  # Back to original size to match checkpoint
    
    model_pos = SpatioTemporalModelOptimized1f(adj, 17, 2, 17, filter_widths=filters_width, 
                                              causal=True, channels=channels, dropout=0.25)
    
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        print("GAST-Net model moved to GPU")
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Set CUDA device to use all available memory
        torch.cuda.set_per_process_memory_fraction(1.0)
        # Pre-allocate a large chunk of GPU memory
        torch.cuda.empty_cache()
        # Allocate multiple smaller tensors to reserve memory (1GB total)
        large_tensor1 = torch.zeros((1, 512, 512, 512), dtype=torch.float32, device='cuda')
        large_tensor2 = torch.zeros((1, 512, 512, 512), dtype=torch.float32, device='cuda')
        print(f"Pre-allocated GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    model_pos.eval()
    print('GAST-Net loaded successfully')
    return model_pos

# Create output directory if it doesn't exist
output_dir = 'data/video'
os.makedirs(output_dir, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = os.path.join(output_dir, f'recording_{timestamp}.mp4')

# Initialize frame buffer for temporal processing
frame_buffer = []
max_buffer_size = 27  # Back to original size

# Load the model
model_pos = load_model_realtime(rf=27)

# List all available cameras
print("\nChecking available cameras:")
for i in range(10):  # Check first 10 camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i} is available:")
            print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
            print(f"  Backend: {cap.getBackendName()}")
            print("---")
        cap.release()

# Use the first available camera
for cam_id in [2]:
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Camera {cam_id} failed to open.")
        continue

    # Initialize video writer with better codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))

    print(f"Recording from Camera {cam_id}")
    print(f"Saving to: {output_file}")
    print("Press 'q' to stop recording or Ctrl+C to exit")

    # Initialize frame counter for processing
    frame_counter = 0
    process_every_n_frames = 2  # Process every 2nd frame to maintain FPS
    
    # Store the last processed skeleton
    last_skeleton = None
    last_frame = None
    
    # FPS monitoring
    fps_start_time = datetime.now()
    fps_counter = 0
    current_fps = 0
    
    # Pre-allocate GPU memory for frames and processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache before starting
        print(f"Initial GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB")
        # Pre-allocate GPU tensors for frame processing (500MB)
        frame_tensor = torch.zeros((1, 3, 480, 640), dtype=torch.float32, device='cuda')
        keypoints_tensor = torch.zeros((1, 27, 17, 2), dtype=torch.float32, device='cuda')
        scores_tensor = torch.zeros((1, 27, 17), dtype=torch.float32, device='cuda')
        print(f"Pre-allocated Frame Tensor Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        fps_counter += 1
        
        # Calculate FPS
        if (datetime.now() - fps_start_time).seconds >= 1:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = datetime.now()
            print(f"Current FPS: {current_fps}")
            if torch.cuda.is_available():
                print(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        
        # Add frame to buffer
        frame_buffer.append(frame)
        if len(frame_buffer) > max_buffer_size:
            frame_buffer.pop(0)
        
        # Process frames when buffer is full and it's time to process
        if len(frame_buffer) == max_buffer_size and frame_counter % process_every_n_frames == 0:
            # Convert frame to GPU tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor[0] = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            
            # Save current frame temporarily
            temp_frame_path = os.path.join(output_dir, 'temp_frame.jpg')
            cv2.imwrite(temp_frame_path, frame)
            
            # Detect keypoints using HRNet with optimized settings
            keypoints, scores = hrnet_pose(temp_frame_path, det_dim=416, num_peroson=1, gen_output=True)
            
            # Remove temporary file
            os.remove(temp_frame_path)
            
            # Ensure keypoints has correct shape (T, M, N, 2)
            if len(keypoints.shape) == 3:
                keypoints = keypoints[np.newaxis, ...]  # Add time dimension if missing
            
            # Convert to H36M format
            keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
            re_kpts = revise_kpts(keypoints, scores, valid_frames)
            
            # Generate 3D poses with GPU optimization
            prediction = gen_pose(re_kpts, valid_frames, 640, 480, model_pos, pad=13, causal_shift=0)
            
            # Normalize height
            prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])
            
            # Store the last processed frame and skeleton
            last_frame = frame.copy()
            if len(re_kpts) > 0 and len(re_kpts[0]) > 0:
                last_skeleton = re_kpts[0][-1]
                frame_with_skeleton = draw_skeleton(last_frame, last_skeleton)
            else:
                frame_with_skeleton = last_frame
            
            # Write frame with skeleton to video
            out.write(frame_with_skeleton)
            
            # Display the frame with skeleton and FPS
            cv2.putText(frame_with_skeleton, f"FPS: {current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("3D Pose Estimation", frame_with_skeleton)
        else:
            # Display frame with last known skeleton if available
            display_frame = frame.copy()
            if last_skeleton is not None:
                display_frame = draw_skeleton(display_frame, last_skeleton)
            cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("3D Pose Estimation", display_frame)
        
        # Check for 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording saved to: {output_file}")
    break  # Exit the camera loop
