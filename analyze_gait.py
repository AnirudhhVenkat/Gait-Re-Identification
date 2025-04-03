import json
import numpy as np
import os
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import math

def load_pose_data(file_path: str) -> Dict:
    """Load pose data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Print metadata for verification
        print(f"\nLoaded pose data from {os.path.basename(file_path)}:")
        print(f"FPS: {data.get('fps', 'N/A')}")
        print(f"Total frames: {data.get('total_frames', 'N/A')}")
        print(f"Processed frames: {data.get('processed_frames', 'N/A')}")
        print(f"Detection rate: {data.get('detection_rate', 'N/A')}%")
        return data

def calculate_joint_angles(pose_data: Dict) -> List[Dict]:
    """Calculate joint angles for each frame."""
    frames = pose_data['pose_landmarks']
    angles = []
    
    for frame_idx, frame in enumerate(frames):
        # Get joint positions using new key names
        left_hip = np.array([frame['left_hip']['x'], frame['left_hip']['y'], frame['left_hip']['z']])
        right_hip = np.array([frame['right_hip']['x'], frame['right_hip']['y'], frame['right_hip']['z']])
        left_knee = np.array([frame['left_knee']['x'], frame['left_knee']['y'], frame['left_knee']['z']])
        right_knee = np.array([frame['right_knee']['x'], frame['right_knee']['y'], frame['right_knee']['z']])
        left_ankle = np.array([frame['left_ankle']['x'], frame['left_ankle']['y'], frame['left_ankle']['z']])
        right_ankle = np.array([frame['right_ankle']['x'], frame['right_ankle']['y'], frame['right_ankle']['z']])
        
        # Calculate hip angles
        left_hip_angle = calculate_angle(left_knee, left_hip, left_ankle)
        right_hip_angle = calculate_angle(right_knee, right_hip, right_ankle)
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate ankle angles
        left_ankle_angle = calculate_angle(left_knee, left_ankle, np.array([left_ankle[0], left_ankle[1], 0]))
        right_ankle_angle = calculate_angle(right_knee, right_ankle, np.array([right_ankle[0], right_ankle[1], 0]))
        
        angles.append({
            'frame': frame_idx,
            'left_hip_angle': left_hip_angle,
            'right_hip_angle': right_hip_angle,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'left_ankle_angle': left_ankle_angle,
            'right_ankle_angle': right_ankle_angle
        })
    
    return angles

def calculate_angle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """Calculate angle between three points in 3D space."""
    # Calculate vectors
    v1 = v1 - v2
    v3 = v3 - v2
    
    # Calculate norms
    norm_v1 = np.linalg.norm(v1)
    norm_v3 = np.linalg.norm(v3)
    
    # Handle edge cases
    if norm_v1 < 1e-10 or norm_v3 < 1e-10:
        return 0.0  # Return 0 degrees if vectors are too small
    
    # Calculate angle
    cos_angle = np.dot(v1, v3) / (norm_v1 * norm_v3)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is in [-1, 1]
    
    try:
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    except:
        return 0.0  # Return 0 degrees if calculation fails

def analyze_gait_cycle(angles: List[Dict]) -> Dict:
    """Analyze gait cycle from joint angles."""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(angles)
    
    # Find gait cycles (from left heel strike to next left heel strike)
    # Using hip angle peaks as indicators
    left_hip_peaks = find_peaks(df['left_hip_angle'])
    
    # Calculate gait cycle metrics
    cycles = []
    for i in range(len(left_hip_peaks) - 1):
        start_idx = left_hip_peaks[i]
        end_idx = left_hip_peaks[i + 1]
        
        cycle = {
            'start_frame': start_idx,
            'end_frame': end_idx,
            'duration': end_idx - start_idx,
            'angles': df.iloc[start_idx:end_idx].to_dict('records')
        }
        cycles.append(cycle)
    
    return {
        'total_cycles': len(cycles),
        'average_cycle_duration': np.mean([c['duration'] for c in cycles]) if cycles else 0,
        'cycles': cycles
    }

def find_peaks(series: pd.Series, threshold: float = 0.1) -> List[int]:
    """Find peaks in a time series."""
    peaks = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i-1] and series[i] > series[i+1]:
            peaks.append(i)
    return peaks

def calculate_physical_features(frame: Dict) -> Dict:
    """Calculate physical/structural features from joint positions."""
    # Get joint positions using new key names
    left_hip = np.array([frame['left_hip']['x'], frame['left_hip']['y'], frame['left_hip']['z']])
    right_hip = np.array([frame['right_hip']['x'], frame['right_hip']['y'], frame['right_hip']['z']])
    left_knee = np.array([frame['left_knee']['x'], frame['left_knee']['y'], frame['left_knee']['z']])
    right_knee = np.array([frame['right_knee']['x'], frame['right_knee']['y'], frame['right_knee']['z']])
    left_ankle = np.array([frame['left_ankle']['x'], frame['left_ankle']['y'], frame['left_ankle']['z']])
    right_ankle = np.array([frame['right_ankle']['x'], frame['right_ankle']['y'], frame['right_ankle']['z']])
    
    # Calculate joint angles
    left_hip_angle = calculate_angle(left_knee, left_hip, left_ankle)
    right_hip_angle = calculate_angle(right_knee, right_hip, right_ankle)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_ankle_angle = calculate_angle(left_knee, left_ankle, np.array([left_ankle[0], left_ankle[1], 0]))
    right_ankle_angle = calculate_angle(right_knee, right_ankle, np.array([right_ankle[0], right_ankle[1], 0]))
    
    # Calculate step width (distance between ankles)
    step_width = euclidean(left_ankle, right_ankle)
    
    # Calculate hip width (distance between hips)
    hip_width = euclidean(left_hip, right_hip)
    
    return {
        'joint_angles': {
            'left_hip': left_hip_angle,
            'right_hip': right_hip_angle,
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle,
            'left_ankle': left_ankle_angle,
            'right_ankle': right_ankle_angle
        },
        'step_width': step_width,
        'hip_width': hip_width
    }

def calculate_dynamic_features(frames: List[Dict]) -> Dict:
    """Calculate dynamic/behavioral features from frame sequence."""
    # Calculate step lengths
    step_lengths = []
    for i in range(1, len(frames)):
        prev_left_ankle = np.array([frames[i-1]['left_ankle']['x'], frames[i-1]['left_ankle']['y'], frames[i-1]['left_ankle']['z']])
        curr_right_ankle = np.array([frames[i]['right_ankle']['x'], frames[i]['right_ankle']['y'], frames[i]['right_ankle']['z']])
        step_length = euclidean(prev_left_ankle, curr_right_ankle)
        step_lengths.append(step_length)
    
    # Calculate stride lengths (distance between same foot positions)
    stride_lengths = []
    for i in range(15, len(frames)):  # Using 15 frames per stride as requested
        prev_left_ankle = np.array([frames[i-15]['left_ankle']['x'], frames[i-15]['left_ankle']['y'], frames[i-15]['left_ankle']['z']])
        curr_left_ankle = np.array([frames[i]['left_ankle']['x'], frames[i]['left_ankle']['y'], frames[i]['left_ankle']['z']])
        stride_length = euclidean(prev_left_ankle, curr_left_ankle)
        stride_lengths.append(stride_length)
    
    # Calculate cadence (steps per minute)
    fps = 100  # Default to 100 fps if not specified
    steps_per_2sec = []
    for i in range(0, len(frames)-2*fps, 2*fps):
        steps = sum(1 for j in range(i, min(i+2*fps, len(frames)-1))
                   if euclidean(np.array([frames[j]['left_ankle']['x'], frames[j]['left_ankle']['y'], frames[j]['left_ankle']['z']]),
                              np.array([frames[j+1]['right_ankle']['x'], frames[j+1]['right_ankle']['y'], frames[j+1]['right_ankle']['z']])) > 0.1)
        steps_per_2sec.append(steps)
    
    # Handle empty arrays
    if not step_lengths:
        step_lengths = [0.0]
    if not stride_lengths:
        stride_lengths = [0.0]
    if not steps_per_2sec:
        steps_per_2sec = [0]
    
    cadence = np.mean(steps_per_2sec) * (fps/2) if steps_per_2sec else 0.0
    
    return {
        'step_lengths': {
            'mean': float(np.mean(step_lengths)),
            'std': float(np.std(step_lengths)) if len(step_lengths) > 1 else 0.0
        },
        'stride_lengths': {
            'mean': float(np.mean(stride_lengths)),
            'std': float(np.std(stride_lengths)) if len(stride_lengths) > 1 else 0.0
        },
        'cadence': float(cadence)
    }

def calculate_temporal_features(physical_features: List[Dict]) -> Dict:
    """Calculate temporal features from joint angles."""
    if not physical_features:
        return {
            'cycle_durations': {
                'left': {'mean': 0.0, 'std': 0.0},
                'right': {'mean': 0.0, 'std': 0.0}
            },
            'symmetry': {'mean': 0.0, 'std': 0.0}
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            'frame': i,
            'left_hip_angle': f['joint_angles']['left_hip'],
            'right_hip_angle': f['joint_angles']['right_hip']
        }
        for i, f in enumerate(physical_features)
    ])
    
    # Find gait cycles using hip angle peaks
    left_hip_peaks = find_peaks(df['left_hip_angle'])
    right_hip_peaks = find_peaks(df['right_hip_angle'])
    
    # Handle empty peak arrays
    if len(left_hip_peaks) < 2 or len(right_hip_peaks) < 2:
        return {
            'cycle_durations': {
                'left': {'mean': 0.0, 'std': 0.0},
                'right': {'mean': 0.0, 'std': 0.0}
            },
            'symmetry': {'mean': 0.0, 'std': 0.0}
        }
    
    # Calculate cycle durations
    left_cycle_durations = []
    right_cycle_durations = []
    
    for i in range(len(left_hip_peaks) - 1):
        left_cycle_durations.append(left_hip_peaks[i + 1] - left_hip_peaks[i])
    
    for i in range(len(right_hip_peaks) - 1):
        right_cycle_durations.append(right_hip_peaks[i + 1] - right_hip_peaks[i])
    
    # Handle empty cycle durations
    if not left_cycle_durations:
        left_cycle_durations = [0.0]
    if not right_cycle_durations:
        right_cycle_durations = [0.0]
    
    # Calculate symmetry metrics
    left_right_symmetry = []
    for i in range(min(len(left_hip_peaks), len(right_hip_peaks))):
        if i + 1 >= len(left_hip_peaks) or i + 1 >= len(right_hip_peaks):
            break
            
        left_angles = df.iloc[left_hip_peaks[i]:left_hip_peaks[i+1]]['left_hip_angle'].values
        right_angles = df.iloc[right_hip_peaks[i]:right_hip_peaks[i+1]]['right_hip_angle'].values
        
        # Pad the shorter sequence
        max_len = max(len(left_angles), len(right_angles))
        left_angles = np.pad(left_angles, (0, max_len - len(left_angles)))
        right_angles = np.pad(right_angles, (0, max_len - len(right_angles)))
        
        # Calculate correlation between left and right angles
        try:
            symmetry = np.corrcoef(left_angles, right_angles)[0, 1]
            if not np.isnan(symmetry):  # Only add valid correlation values
                left_right_symmetry.append(symmetry)
        except:
            continue
    
    # Handle empty symmetry array
    if not left_right_symmetry:
        left_right_symmetry = [0.0]
    
    return {
        'cycle_durations': {
            'left': {
                'mean': float(np.mean(left_cycle_durations)),
                'std': float(np.std(left_cycle_durations)) if len(left_cycle_durations) > 1 else 0.0
            },
            'right': {
                'mean': float(np.mean(right_cycle_durations)),
                'std': float(np.std(right_cycle_durations)) if len(right_cycle_durations) > 1 else 0.0
            }
        },
        'symmetry': {
            'mean': float(np.mean(left_right_symmetry)),
            'std': float(np.std(left_right_symmetry)) if len(left_right_symmetry) > 1 else 0.0
        }
    }

def calculate_real_time_features(frames: List[Dict], window_size: int = 150, stride: int = 15) -> List[Dict]:
    """Calculate real-time features using sliding window approach."""
    if not frames:
        return []
        
    features = []
    
    # Calculate features for each window
    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i + window_size]
        
        # Calculate physical features for the window
        physical_features = [calculate_physical_features(frame) for frame in window]
        
        # Calculate dynamic features for the window
        dynamic_features = calculate_dynamic_features(window)
        
        # Calculate temporal features for the window
        temporal_features = calculate_temporal_features(physical_features)
        
        # Calculate walking speed (using ankle positions)
        if len(window) > 1:
            first_left_ankle = np.array([window[0]['left_ankle']['x'], window[0]['left_ankle']['y'], window[0]['left_ankle']['z']])
            last_left_ankle = np.array([window[-1]['left_ankle']['x'], window[-1]['left_ankle']['y'], window[-1]['left_ankle']['z']])
            distance = euclidean(last_left_ankle, first_left_ankle)
            time = (window_size - 1) / 100  # Assuming 100 fps
            walking_speed = distance / time
        else:
            walking_speed = 0.0
        
        # Calculate symmetry score
        left_angles = [f['joint_angles']['left_hip'] for f in physical_features]
        right_angles = [f['joint_angles']['right_hip'] for f in physical_features]
        
        # Pad the shorter sequence
        max_len = max(len(left_angles), len(right_angles))
        left_angles = np.pad(left_angles, (0, max_len - len(left_angles)))
        right_angles = np.pad(right_angles, (0, max_len - len(right_angles)))
        
        # Calculate correlation between left and right angles
        try:
            symmetry = np.corrcoef(left_angles, right_angles)[0, 1]
            if np.isnan(symmetry):
                symmetry = 0.0
        except:
            symmetry = 0.0
        
        features.append({
            'frame': i,
            'joint_angles': physical_features[-1]['joint_angles'],
            'step_width': physical_features[-1]['step_width'],
            'hip_width': physical_features[-1]['hip_width'],
            'step_length': dynamic_features['step_lengths']['mean'],
            'walking_speed': walking_speed,
            'cadence': dynamic_features['cadence'],
            'symmetry': symmetry
        })
    
    return features

def analyze_gait_patterns(pose_file: str, output_dir: str) -> Dict:
    """Analyze gait patterns from pose data and save results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pose data
    pose_data = load_pose_data(pose_file)
    
    # Calculate features
    angles = calculate_joint_angles(pose_data)
    physical_features = [calculate_physical_features(frame) for frame in pose_data['pose_landmarks']]
    dynamic_features = calculate_dynamic_features(pose_data['pose_landmarks'])
    temporal_features = calculate_temporal_features(physical_features)
    real_time_features = calculate_real_time_features(pose_data['pose_landmarks'])
    
    # Combine all features
    analysis_results = {
        'metadata': {
            'fps': pose_data.get('fps', 100),
            'total_frames': pose_data.get('total_frames', len(pose_data['pose_landmarks'])),
            'processed_frames': pose_data.get('processed_frames', len(pose_data['pose_landmarks'])),
            'detection_rate': pose_data.get('detection_rate', 100.0),
            'timestamp': pose_data.get('timestamp', datetime.now().isoformat())
        },
        'joint_angles': angles,
        'physical_features': physical_features,
        'dynamic_features': dynamic_features,
        'temporal_features': temporal_features,
        'real_time_features': real_time_features
    }
    
    # Save results
    output_file = os.path.join(output_dir, f"analysis_{os.path.basename(pose_file)}")
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    return analysis_results

def main():
    """Main function to process all converted pose files."""
    # Process all converted pose files
    input_dir = "converted_poses"
    output_dir = "gait_analysis"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    pose_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    print(f"Found {len(pose_files)} pose files to analyze")
    
    # Process each file
    for pose_file in tqdm(pose_files, desc="Analyzing gait patterns"):
        input_path = os.path.join(input_dir, pose_file)
        try:
            analysis_results = analyze_gait_patterns(input_path, output_dir)
            print(f"\nSuccessfully analyzed {pose_file}")
        except Exception as e:
            print(f"\nError analyzing {pose_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 