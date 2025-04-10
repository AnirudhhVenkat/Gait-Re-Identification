import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Tuple
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class GaitEvent:
    type: str  # 'heel_strike' or 'mid_swing'
    foot: str  # 'right' or 'left'
    frame: int
    height: float

def load_pose_data(file_path: str) -> Dict[str, Any]:
    """Load pose data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing pose data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_cycle_parameters(ankle_heights: np.ndarray) -> Tuple[int, int]:
    """Calculate optimal parameters based on the data."""
    # Calculate average cycle length from peaks
    peaks, _ = find_peaks(ankle_heights)
    if len(peaks) > 1:
        avg_cycle_length = np.mean(np.diff(peaks))
        min_length = int(avg_cycle_length * 0.7)  # 70% of average
        distance = int(avg_cycle_length * 0.3)    # 30% of average
    else:
        min_length = 30
        distance = 20
    
    return min_length, distance

def detect_gait_events(pose_data_list: List[Dict[str, Any]], 
                      distance: int = 20) -> List[GaitEvent]:
    """Detect all gait events (heel strikes and mid-swings) for both feet."""
    events = []
    right_ankle_heights = []
    left_ankle_heights = []
    
    # First pass: collect ankle heights
    for pose_data in pose_data_list:
        right_ankle = np.array([
            pose_data['right_ankle']['x'],
            pose_data['right_ankle']['y'],
            pose_data['right_ankle']['z']
        ])
        left_ankle = np.array([
            pose_data['left_ankle']['x'],
            pose_data['left_ankle']['y'],
            pose_data['left_ankle']['z']
        ])
        right_ankle_heights.append(right_ankle[1])
        left_ankle_heights.append(left_ankle[1])
    
    right_ankle_heights = np.array(right_ankle_heights)
    left_ankle_heights = np.array(left_ankle_heights)
    
    # Find events for both feet
    right_heel_strikes, _ = find_peaks(-right_ankle_heights, distance=distance)
    right_mid_swings, _ = find_peaks(right_ankle_heights, distance=distance)
    left_heel_strikes, _ = find_peaks(-left_ankle_heights, distance=distance)
    left_mid_swings, _ = find_peaks(left_ankle_heights, distance=distance)
    
    # Create event objects
    for idx in right_heel_strikes:
        events.append(GaitEvent('heel_strike', 'right', idx, right_ankle_heights[idx]))
    for idx in left_heel_strikes:
        events.append(GaitEvent('heel_strike', 'left', idx, left_ankle_heights[idx]))
    for idx in right_mid_swings:
        events.append(GaitEvent('mid_swing', 'right', idx, right_ankle_heights[idx]))
    for idx in left_mid_swings:
        events.append(GaitEvent('mid_swing', 'left', idx, left_ankle_heights[idx]))
    
    # Sort events by frame number
    events.sort(key=lambda x: x.frame)
    
    return events, right_ankle_heights, left_ankle_heights

def is_complete_cycle(cycle_events: List[GaitEvent]) -> bool:
    """Check if a cycle contains all necessary gait events."""
    event_types = set()
    for event in cycle_events:
        event_types.add((event.type, event.foot))
    
    required_events = {
        ('heel_strike', 'right'),
        ('heel_strike', 'left'),
        ('mid_swing', 'right'),
        ('mid_swing', 'left')
    }
    
    return required_events.issubset(event_types)

def detect_gait_cycles(pose_data_list: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], List[List[GaitEvent]], np.ndarray, np.ndarray]:
    """Detect complete gait cycles from pose data using event-based detection."""
    # Detect all gait events
    events, right_heights, left_heights = detect_gait_events(pose_data_list)
    
    if not events:
        return [], [], right_heights, left_heights
    
    # Find right heel strikes to use as cycle boundaries
    right_heel_strikes = [e for e in events if e.type == 'heel_strike' and e.foot == 'right']
    
    if len(right_heel_strikes) < 2:
        return [], [], right_heights, left_heights
    
    # Split into cycles
    cycles = []
    cycle_events = []
    start_idx = 0
    
    for i in range(1, len(right_heel_strikes)):
        end_idx = right_heel_strikes[i].frame
        cycle = pose_data_list[start_idx:end_idx]
        
        # Get events in this cycle
        current_cycle_events = [e for e in events if start_idx <= e.frame < end_idx]
        
        if len(cycle) >= 30 and is_complete_cycle(current_cycle_events):
            cycles.append(cycle)
            cycle_events.append(current_cycle_events)
        
        start_idx = end_idx
    
    return cycles, cycle_events, right_heights, left_heights

def analyze_cycle(cycle: List[Dict[str, Any]], events: List[GaitEvent]) -> Dict[str, float]:
    """Analyze a single gait cycle with event timing."""
    stats = {
        'cycle_length': len(cycle),
        'right_heel_strike_frame': None,
        'right_mid_swing_frame': None,
        'left_heel_strike_frame': None,
        'left_mid_swing_frame': None,
        'right_ankle_min': float('inf'),
        'right_ankle_max': float('-inf'),
        'left_ankle_min': float('inf'),
        'left_ankle_max': float('-inf'),
        'step_widths': [],
        'hip_widths': []
    }
    
    # Record event frames
    for event in events:
        if event.type == 'heel_strike' and event.foot == 'right':
            stats['right_heel_strike_frame'] = event.frame
        elif event.type == 'mid_swing' and event.foot == 'right':
            stats['right_mid_swing_frame'] = event.frame
        elif event.type == 'heel_strike' and event.foot == 'left':
            stats['left_heel_strike_frame'] = event.frame
        elif event.type == 'mid_swing' and event.foot == 'left':
            stats['left_mid_swing_frame'] = event.frame
    
    # Calculate other statistics
    for pose_data in cycle:
        # Get ankle heights
        right_ankle = np.array([
            pose_data['right_ankle']['x'],
            pose_data['right_ankle']['y'],
            pose_data['right_ankle']['z']
        ])
        left_ankle = np.array([
            pose_data['left_ankle']['x'],
            pose_data['left_ankle']['y'],
            pose_data['left_ankle']['z']
        ])
        
        # Update min/max heights
        stats['right_ankle_min'] = min(stats['right_ankle_min'], right_ankle[1])
        stats['right_ankle_max'] = max(stats['right_ankle_max'], right_ankle[1])
        stats['left_ankle_min'] = min(stats['left_ankle_min'], left_ankle[1])
        stats['left_ankle_max'] = max(stats['left_ankle_max'], left_ankle[1])
        
        # Get hip positions
        right_hip = np.array([
            pose_data['right_hip']['x'],
            pose_data['right_hip']['y'],
            pose_data['right_hip']['z']
        ])
        left_hip = np.array([
            pose_data['left_hip']['x'],
            pose_data['left_hip']['y'],
            pose_data['left_hip']['z']
        ])
        
        # Calculate widths
        stats['step_widths'].append(np.abs(right_ankle[0] - left_ankle[0]))
        stats['hip_widths'].append(np.abs(right_hip[0] - left_hip[0]))
    
    # Calculate means and stds
    stats['step_width_mean'] = np.mean(stats['step_widths'])
    stats['step_width_std'] = np.std(stats['step_widths'])
    stats['hip_width_mean'] = np.mean(stats['hip_widths'])
    stats['hip_width_std'] = np.std(stats['hip_widths'])
    
    # Calculate timing metrics
    if all(stats[f'{foot}_{event}_frame'] is not None 
           for foot in ['right', 'left'] 
           for event in ['heel_strike', 'mid_swing']):
        stats['right_stance_phase'] = (stats['right_mid_swing_frame'] - stats['right_heel_strike_frame']) / len(cycle)
        stats['left_stance_phase'] = (stats['left_mid_swing_frame'] - stats['left_heel_strike_frame']) / len(cycle)
        stats['double_support'] = abs(stats['right_heel_strike_frame'] - stats['left_heel_strike_frame']) / len(cycle)
    
    return stats

def plot_cycle_analysis(cycles: List[List[Dict[str, Any]]], 
                       cycle_events: List[List[GaitEvent]],
                       right_heights: np.ndarray,
                       left_heights: np.ndarray,
                       output_dir: str,
                       file_name: str):
    """Plot detailed analysis of gait cycles with events."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Ankle heights with events
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(right_heights, label='Right Ankle', color='blue')
    ax1.plot(left_heights, label='Left Ankle', color='red')
    
    # Plot events
    for events in cycle_events:
        for event in events:
            color = 'blue' if event.foot == 'right' else 'red'
            marker = 'v' if event.type == 'heel_strike' else '^'
            ax1.scatter(event.frame, event.height, color=color, marker=marker, s=100)
    
    ax1.set_title('Ankle Heights with Gait Events')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Height')
    ax1.legend()
    
    # Plot 2: Cycle statistics
    ax2 = plt.subplot(3, 1, 2)
    cycle_stats = [analyze_cycle(cycle, events) for cycle, events in zip(cycles, cycle_events)]
    stats_df = pd.DataFrame(cycle_stats)
    
    # Plot step and hip widths
    sns.boxplot(data=stats_df[['step_width_mean', 'hip_width_mean']], ax=ax2)
    ax2.set_title('Step and Hip Width Statistics')
    ax2.set_ylabel('Width')
    
    # Plot 3: Timing analysis
    ax3 = plt.subplot(3, 1, 3)
    timing_stats = stats_df[['right_stance_phase', 'left_stance_phase', 'double_support']]
    sns.boxplot(data=timing_stats, ax=ax3)
    ax3.set_title('Gait Phase Timing')
    ax3.set_ylabel('Proportion of Cycle')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}_analysis.png'))
    plt.close()
    
    # Save detailed statistics
    stats_df.to_csv(os.path.join(output_dir, f'{file_name}_stats.csv'))
    
    # Save cycle summary
    summary = {
        'total_cycles': len(cycles),
        'avg_cycle_length': np.mean([len(c) for c in cycles]),
        'avg_step_width': np.mean(stats_df['step_width_mean']),
        'avg_hip_width': np.mean(stats_df['hip_width_mean']),
        'avg_right_stance': np.mean(stats_df['right_stance_phase']),
        'avg_left_stance': np.mean(stats_df['left_stance_phase']),
        'avg_double_support': np.mean(stats_df['double_support'])
    }
    
    with open(os.path.join(output_dir, f'{file_name}_summary.txt'), 'w') as f:
        f.write("Gait Cycle Analysis Summary\n")
        f.write("=========================\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.4f}\n")

def extract_cycle_features(cycle: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract features from a gait cycle that match the vector database features."""
    # Initialize arrays to store features
    step_widths = []
    hip_widths = []
    right_hip_angles = []
    left_hip_angles = []
    step_lengths = []
    stride_lengths = []
    
    # Calculate features for each frame in the cycle
    for pose_data in cycle:
        # Physical features
        right_ankle = np.array([
            pose_data['right_ankle']['x'],
            pose_data['right_ankle']['y'],
            pose_data['right_ankle']['z']
        ])
        left_ankle = np.array([
            pose_data['left_ankle']['x'],
            pose_data['left_ankle']['y'],
            pose_data['left_ankle']['z']
        ])
        right_hip = np.array([
            pose_data['right_hip']['x'],
            pose_data['right_hip']['y'],
            pose_data['right_hip']['z']
        ])
        left_hip = np.array([
            pose_data['left_hip']['x'],
            pose_data['left_hip']['y'],
            pose_data['left_hip']['z']
        ])
        
        # Calculate step width and hip width
        step_widths.append(np.abs(right_ankle[0] - left_ankle[0]))
        hip_widths.append(np.abs(right_hip[0] - left_hip[0]))
        
        # Calculate hip angles
        right_knee = np.array([
            pose_data['right_knee']['x'],
            pose_data['right_knee']['y'],
            pose_data['right_knee']['z']
        ])
        left_knee = np.array([
            pose_data['left_knee']['x'],
            pose_data['left_knee']['y'],
            pose_data['left_knee']['z']
        ])
        
        right_thigh = right_knee - right_hip
        right_shin = right_ankle - right_knee
        left_thigh = left_knee - left_hip
        left_shin = left_ankle - left_knee
        
        right_hip_angle = np.arccos(np.dot(right_thigh, right_shin) / 
                                   (np.linalg.norm(right_thigh) * np.linalg.norm(right_shin)))
        left_hip_angle = np.arccos(np.dot(left_thigh, left_shin) / 
                                  (np.linalg.norm(left_thigh) * np.linalg.norm(left_shin)))
        
        right_hip_angles.append(right_hip_angle)
        left_hip_angles.append(left_hip_angle)
        
        # Calculate step and stride lengths
        step_lengths.append(np.abs(right_ankle[0] - left_ankle[0]))
        stride_lengths.append(np.abs(right_ankle[0]))
    
    # Calculate statistics
    features = {
        'step_width': np.mean(step_widths),
        'hip_width': np.mean(hip_widths),
        'right_hip_angle': np.mean(right_hip_angles),
        'left_hip_angle': np.mean(left_hip_angles),
        'mean_step_length': np.mean(step_lengths),
        'step_length_std': np.std(step_lengths),
        'mean_stride_length': np.mean(stride_lengths),
        'stride_length_std': np.std(stride_lengths)
    }
    
    return features

def extract_subject_id(file_name: str) -> str:
    """Extract subject ID from file name (e.g., 'S01' from 'converted_Sub1_Kinematics_T7.json')."""
    # Extract the subject number from the filename
    # Format: converted_Sub1_Kinematics_T7.json -> S01
    parts = file_name.split('_')
    if len(parts) >= 2 and parts[1].startswith('Sub'):
        # Extract the number from "Sub1", "Sub2", etc.
        sub_num = parts[1][3:]  # Remove "Sub" prefix
        if sub_num.isdigit():
            # Format as S01, S02, etc.
            return f"S{int(sub_num):02d}"
    raise ValueError(f"Invalid subject ID in filename: {file_name}")

def save_cycle_features(cycles: List[List[Dict[str, Any]]], 
                       cycle_events: List[List[GaitEvent]],
                       output_dir: str,
                       file_name: str) -> None:
    """Save features for each gait cycle as a separate CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract subject ID
    subject_id = extract_subject_id(file_name)
    
    # Process each cycle
    for i, (cycle, events) in enumerate(zip(cycles, cycle_events)):
        # Extract features
        features = extract_cycle_features(cycle)
        
        # Add cycle metadata
        features.update({
            'subject_id': subject_id,
            'trial_id': file_name,
            'cycle_index': i,
            'cycle_length': len(cycle),
            'right_heel_strike': events[0].frame if events else None,
            'left_heel_strike': events[1].frame if len(events) > 1 else None,
            'right_mid_swing': events[2].frame if len(events) > 2 else None,
            'left_mid_swing': events[3].frame if len(events) > 3 else None
        })
        
        # Save this cycle's features
        cycle_df = pd.DataFrame([features])
        cycle_df.to_csv(os.path.join(output_dir, f'{subject_id}_{file_name}_cycle_{i:03d}_features.csv'), index=False)

def main():
    # Set up paths
    pose_dir = "converted_poses"
    output_dir = "gait_cycles"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of pose files
    pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.json')]
    print(f"Found {len(pose_files)} pose files to process")
    
    # Group files by subject
    subject_files = {}
    for file_name in pose_files:
        try:
            subject_id = extract_subject_id(file_name)
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file_name)
        except ValueError as e:
            print(f"Warning: {str(e)}")
            continue
    
    print(f"Found {len(subject_files)} unique subjects: {sorted(subject_files.keys())}")
    if len(subject_files) != 10:
        print(f"Warning: Expected 10 unique subjects, found {len(subject_files)}")
        print("Missing subjects:", sorted(set(f"S{i:02d}" for i in range(1, 11)) - set(subject_files.keys())))
    
    # Process each file
    for file_name in tqdm(pose_files, desc="Processing files"):
        file_path = os.path.join(pose_dir, file_name)
        
        try:
            # Extract subject ID first
            subject_id = extract_subject_id(file_name)
            
            # Load pose data
            pose_data = load_pose_data(file_path)
            pose_landmarks = pose_data.get('pose_landmarks', [])
            
            if not pose_landmarks:
                print(f"Warning: No pose landmarks found in {file_name}")
                continue
            
            # Detect gait cycles
            cycles, cycle_events, right_heights, left_heights = detect_gait_cycles(pose_landmarks)
            
            if not cycles:
                print(f"Warning: No complete gait cycles found in {file_name}")
                continue
            
            # Save cycle features
            save_cycle_features(cycles, cycle_events, output_dir, file_name.replace('.json', ''))
            
            # Plot cycle analysis
            plot_cycle_analysis(cycles, cycle_events, right_heights, left_heights, 
                              output_dir, file_name.replace('.json', ''))
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    
    # Print summary of processed data
    print("\nProcessing Summary:")
    print(f"Total files processed: {len(pose_files)}")
    print(f"Unique subjects: {len(subject_files)}")
    print("Subjects:", sorted(subject_files.keys()))
    print("Files per subject:")
    for subject_id, files in sorted(subject_files.items()):
        print(f"  {subject_id}: {len(files)} files")

if __name__ == "__main__":
    main() 