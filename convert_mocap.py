import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm

def read_mocap_file(file_path):
    """Read motion capture file and return DataFrame."""
    try:
        # Read the header information
        with open(file_path, 'r') as file:
            header_lines = [next(file) for _ in range(4)]
        
        # Get marker names from line 3
        markers = header_lines[2].strip().split(',')[2:]
        markers = [m for m in markers if m]
        
        # Print available markers for verification
        print(f"\nAvailable markers in {os.path.basename(file_path)}:")
        for marker in markers:
            print(f"- {marker}")
        
        # Read the actual data with proper type conversion
        df = pd.read_csv(file_path, skiprows=3, header=None, dtype=str)
        
        # Create column names
        columns = ['Frame', 'Sub Frame']
        for marker in markers:
            columns.extend([f'{marker}_X', f'{marker}_Y', f'{marker}_Z'])
        
        df.columns = columns
        
        # Convert numeric columns to float, ignoring errors
        for col in df.columns[2:]:  # Skip Frame and Sub Frame
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows where all marker values are NaN
        df = df.dropna(subset=df.columns[2:], how='all')
        
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def calculate_joint_center(df, lateral_marker, medial_marker):
    """Calculate joint center from lateral and medial markers."""
    try:
        joint_center = pd.DataFrame()
        for coord in ['X', 'Y', 'Z']:
            lateral = df[f'{lateral_marker}_{coord}']
            medial = df[f'{medial_marker}_{coord}']
            
            # Print first few values for verification
            print(f"\nCalculating {lateral_marker} - {medial_marker} center:")
            print(f"Lateral {coord}: {lateral.head()}")
            print(f"Medial {coord}: {medial.head()}")
            
            joint_center[coord] = (lateral + medial) / 2
            print(f"Center {coord}: {joint_center[coord].head()}")
            
        return joint_center
    except Exception as e:
        print(f"Error calculating joint center: {str(e)}")
        return None

def normalize_coordinates(df):
    """Normalize coordinates to 0-1 range."""
    normalized = df.copy()
    
    # Normalize each coordinate independently
    for coord in ['X', 'Y', 'Z']:
        min_val = normalized[coord].min()
        max_val = normalized[coord].max()
        if max_val > min_val:
            normalized[coord] = (normalized[coord] - min_val) / (max_val - min_val)
    
    return normalized

def get_marker_mapping(markers):
    """Map marker names to standard format."""
    mapping = {}
    
    # Define possible marker name variations with their common prefixes and suffixes
    marker_patterns = {
        'RHIP': [
            ['R', 'RIGHT', 'RH'],
            ['HIP', 'HIP_EXT', 'HIPEXT', 'HIP_EXTENSION', 'HIPEXTENSION', 'HIP_EXTRA'],
            ['R_HIP', 'RHIP', 'R_HIP_EXT', 'RHIP_EXT', 'RHip', 'RHipExt']
        ],
        'RHIPExt': [
            ['R', 'RIGHT', 'RH'],
            ['HIP_EXT', 'HIPEXT', 'HIP_EXTENSION', 'HIPEXTENSION', 'HIP_EXTRA'],
            ['R_HIP_EXT', 'RHIP_EXT', 'RHipExt']
        ],
        'LHIP': [
            ['L', 'LEFT', 'LH'],
            ['HIP', 'HIP_EXT', 'HIPEXT', 'HIP_EXTENSION', 'HIPEXTENSION', 'HIP_EXTRA'],
            ['L_HIP', 'LHIP', 'L_HIP_EXT', 'LHIP_EXT', 'LHip', 'LHipExt']
        ],
        'LHIPExt': [
            ['L', 'LEFT', 'LH'],
            ['HIP_EXT', 'HIPEXT', 'HIP_EXTENSION', 'HIPEXTENSION', 'HIP_EXTRA'],
            ['L_HIP_EXT', 'LHIP_EXT', 'LHipExt']
        ],
        'RLKN': [
            ['R', 'RIGHT', 'RH'],
            ['KNEE', 'KNEE_MED', 'KNEE_MEDIAL', 'KNEE_MID', 'KNEE_CENTER', 'KNEE_EXT', 'KNEEEXT'],
            ['R_KNEE', 'RKNEE', 'R_KNEE_MED', 'RKNEE_MED', 'RLKN']
        ],
        'RMKN': [
            ['R', 'RIGHT', 'RH'],
            ['KNEE_MED', 'KNEE_MEDIAL', 'KNEE_MID', 'KNEE_CENTER'],
            ['R_KNEE_MED', 'RKNEE_MED', 'RMKN']
        ],
        'LLKN': [
            ['L', 'LEFT', 'LH'],
            ['KNEE', 'KNEE_MED', 'KNEE_MEDIAL', 'KNEE_MID', 'KNEE_CENTER', 'KNEE_EXT', 'KNEEEXT'],
            ['L_KNEE', 'LKNEE', 'L_KNEE_MED', 'LKNEE_MED', 'LLKN']
        ],
        'LMKN': [
            ['L', 'LEFT', 'LH'],
            ['KNEE_MED', 'KNEE_MEDIAL', 'KNEE_MID', 'KNEE_CENTER'],
            ['L_KNEE_MED', 'LKNEE_MED', 'LMKN']
        ],
        'RLM': [
            ['R', 'RIGHT', 'RH'],
            ['ANKLE', 'ANKLE_MED', 'ANKLE_MEDIAL', 'ANKLE_MID', 'ANKLE_CENTER', 'ANKLE_EXT', 'ANKLEEXT'],
            ['R_ANKLE', 'RANKLE', 'R_ANKLE_MED', 'RANKLE_MED', 'RLM']
        ],
        'RMM': [
            ['R', 'RIGHT', 'RH'],
            ['ANKLE_MED', 'ANKLE_MEDIAL', 'ANKLE_MID', 'ANKLE_CENTER'],
            ['R_ANKLE_MED', 'RANKLE_MED', 'RMM']
        ],
        'LLM': [
            ['L', 'LEFT', 'LH'],
            ['ANKLE', 'ANKLE_MED', 'ANKLE_MEDIAL', 'ANKLE_MID', 'ANKLE_CENTER', 'ANKLE_EXT', 'ANKLEEXT'],
            ['L_ANKLE', 'LANKLE', 'L_ANKLE_MED', 'LANKLE_MED', 'LLM']
        ],
        'LMM': [
            ['L', 'LEFT', 'LH'],
            ['ANKLE_MED', 'ANKLE_MEDIAL', 'ANKLE_MID', 'ANKLE_CENTER'],
            ['L_ANKLE_MED', 'LANKLE_MED', 'LMM']
        ],
        'RCAL': [
            ['R', 'RIGHT', 'RH'],
            ['CAL', 'CALC', 'CALCANEUS', 'HEEL', 'HEEL_MARKER', 'CAL_EXT', 'CALEXT'],
            ['R_CAL', 'RCAL', 'R_CALC', 'RCALC', 'R_HEEL']
        ],
        'LCAL': [
            ['L', 'LEFT', 'LH'],
            ['CAL', 'CALC', 'CALCANEUS', 'HEEL', 'HEEL_MARKER', 'CAL_EXT', 'CALEXT'],
            ['L_CAL', 'LCAL', 'L_CALC', 'LCALC', 'L_HEEL']
        ],
        'RTOE': [
            ['R', 'RIGHT', 'RH'],
            ['TOE', 'BIG_TOE', 'TOE_MARKER', 'BIG_TOE_MARKER', 'TOE_EXT', 'TOEEXT'],
            ['R_TOE', 'RTOE', 'R_BIG_TOE', 'RBIG_TOE']
        ],
        'LTOE': [
            ['L', 'LEFT', 'LH'],
            ['TOE', 'BIG_TOE', 'TOE_MARKER', 'BIG_TOE_MARKER', 'TOE_EXT', 'TOEEXT'],
            ['L_TOE', 'LTOE', 'L_BIG_TOE', 'LBIG_TOE']
        ]
    }
    
    # Find matching markers
    for standard_name, patterns in marker_patterns.items():
        for marker in markers:
            # Remove subject prefix and clean marker name
            marker_name = marker.split(':')[-1].replace('_', '').replace('-', '').upper()
            
            # Check each pattern combination
            for prefix in patterns[0]:
                for suffix in patterns[1]:
                    # Try direct combination
                    pattern = f"{prefix}{suffix}".replace('_', '').replace('-', '').upper()
                    if pattern in marker_name or marker_name in pattern:
                        mapping[standard_name] = marker
                        print(f"Found match: {marker} -> {standard_name} (pattern: {pattern})")
                        break
                    
                    # Try with underscore
                    pattern = f"{prefix}_{suffix}".replace('_', '').replace('-', '').upper()
                    if pattern in marker_name or marker_name in pattern:
                        mapping[standard_name] = marker
                        print(f"Found match: {marker} -> {standard_name} (pattern: {pattern})")
                        break
                
                if standard_name in mapping:
                    break
            
            # If no match found with patterns, try exact variations
            if standard_name not in mapping:
                for variation in patterns[2]:
                    var_clean = variation.replace('_', '').replace('-', '').upper()
                    if var_clean in marker_name or marker_name in var_clean:
                        mapping[standard_name] = marker
                        print(f"Found match: {marker} -> {standard_name} (variation: {variation})")
                        break
            
            if standard_name in mapping:
                break
    
    # Verify all required markers are found
    missing_markers = [name for name in marker_patterns.keys() if name not in mapping]
    if missing_markers:
        print(f"\nWarning: Missing markers for {', '.join(missing_markers)}")
        print("Available markers:", markers)
    
    return mapping

def convert_to_mediapipe_format(df, output_file):
    """Convert motion capture data to OpenPose format."""
    try:
        # Get marker mapping
        markers = [col.split('_')[0] for col in df.columns[2:] if col.endswith('_X')]
        marker_mapping = get_marker_mapping(markers)
        
        # Verify all required markers are present
        required_markers = ['RHIP', 'RHIPExt', 'LHIP', 'LHIPExt', 
                          'RLKN', 'RMKN', 'LLKN', 'LMKN',
                          'RLM', 'RMM', 'LLM', 'LMM',
                          'RCAL', 'LCAL', 'RTOE', 'LTOE']
        
        missing_markers = [marker for marker in required_markers if marker not in marker_mapping]
        if missing_markers:
            print(f"\nError: Missing required markers: {', '.join(missing_markers)}")
            return
        
        # Print marker mapping for verification
        print("\nMarker mapping:")
        for standard_name, actual_name in marker_mapping.items():
            print(f"{standard_name} -> {actual_name}")
        
        # Calculate joint centers with verification
        print("\nCalculating joint centers...")
        
        # Hip centers (OpenPose keypoints 8, 11)
        hip_center_right = calculate_joint_center(df, marker_mapping['RHIP'], marker_mapping['RHIPExt'])
        hip_center_left = calculate_joint_center(df, marker_mapping['LHIP'], marker_mapping['LHIPExt'])
        
        if hip_center_right is None or hip_center_left is None:
            print("\nError: Failed to calculate hip centers")
            return
        
        # Knee centers (OpenPose keypoints 9, 12)
        knee_center_right = calculate_joint_center(df, marker_mapping['RLKN'], marker_mapping['RMKN'])
        knee_center_left = calculate_joint_center(df, marker_mapping['LLKN'], marker_mapping['LMKN'])
        
        if knee_center_right is None or knee_center_left is None:
            print("\nError: Failed to calculate knee centers")
            return
        
        # Ankle centers (OpenPose keypoints 10, 13)
        ankle_center_right = calculate_joint_center(df, marker_mapping['RLM'], marker_mapping['RMM'])
        ankle_center_left = calculate_joint_center(df, marker_mapping['LLM'], marker_mapping['LMM'])
        
        if ankle_center_right is None or ankle_center_left is None:
            print("\nError: Failed to calculate ankle centers")
            return
        
        # Map the joints to our format (OpenPose keypoints)
        joint_mapping = {
            # Right leg (8-10)
            'right_hip': normalize_coordinates(hip_center_right),
            'right_knee': normalize_coordinates(knee_center_right),
            'right_ankle': normalize_coordinates(ankle_center_right),
            
            # Left leg (11-13)
            'left_hip': normalize_coordinates(hip_center_left),
            'left_knee': normalize_coordinates(knee_center_left),
            'left_ankle': normalize_coordinates(ankle_center_left),
            
            # Right foot (14-15)
            'right_heel': normalize_coordinates(df[[f"{marker_mapping['RCAL']}_X", f"{marker_mapping['RCAL']}_Y", f"{marker_mapping['RCAL']}_Z"]].rename(columns=lambda x: x.split('_')[-1])),
            'right_big_toe': normalize_coordinates(df[[f"{marker_mapping['RTOE']}_X", f"{marker_mapping['RTOE']}_Y", f"{marker_mapping['RTOE']}_Z"]].rename(columns=lambda x: x.split('_')[-1])),
            
            # Left foot (16-17)
            'left_heel': normalize_coordinates(df[[f"{marker_mapping['LCAL']}_X", f"{marker_mapping['LCAL']}_Y", f"{marker_mapping['LCAL']}_Z"]].rename(columns=lambda x: x.split('_')[-1])),
            'left_big_toe': normalize_coordinates(df[[f"{marker_mapping['LTOE']}_X", f"{marker_mapping['LTOE']}_Y", f"{marker_mapping['LTOE']}_Z"]].rename(columns=lambda x: x.split('_')[-1]))
        }
        
        # Convert to our JSON format
        frames = []
        num_frames = len(df)
        valid_frames = 0
        
        for frame_idx in tqdm(range(num_frames), desc="Converting frames", leave=False):
            frame_data = {}
            frame_valid = True
            
            for joint_name, joint_data in joint_mapping.items():
                # Check if any coordinate is NaN
                if pd.isna(joint_data['X'].iloc[frame_idx]) or \
                   pd.isna(joint_data['Y'].iloc[frame_idx]) or \
                   pd.isna(joint_data['Z'].iloc[frame_idx]):
                    frame_valid = False
                    continue
                    
                frame_data[joint_name] = {
                    'x': float(joint_data['X'].iloc[frame_idx]),
                    'y': float(joint_data['Y'].iloc[frame_idx]),
                    'z': float(joint_data['Z'].iloc[frame_idx]),
                    'visibility': 1.0  # Motion capture data is always visible
                }
            
            # Only append frame if it has all required joints
            if frame_valid and len(frame_data) == len(joint_mapping):
                frames.append(frame_data)
                valid_frames += 1
        
        # Create final data structure
        data = {
            'fps': 100,  # Motion capture data is typically 100Hz
            'total_frames': num_frames,
            'processed_frames': valid_frames,
            'pose_landmarks': frames,
            'detection_rate': (valid_frames / num_frames) * 100 if num_frames > 0 else 0,
            'valid_pose_rate': (valid_frames / num_frames) * 100 if num_frames > 0 else 0,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\nSuccessfully converted {valid_frames}/{num_frames} frames")
        
    except Exception as e:
        print(f"Error converting to OpenPose format: {str(e)}")

def process_mocap_directory(input_dir, output_dir):
    """Process all CSV files in the input directory and convert them to OpenPose format."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Filter files to only include T1 to T15
    csv_files = [f for f in csv_files if any(f.endswith(f'_T{i}.csv') for i in range(1, 16))]
    
    # Sort files to ensure consistent processing order
    csv_files.sort()
    
    print(f"\nProcessing {len(csv_files)} files in {input_dir}")
    
    # Process each file with progress bar
    for file in tqdm(csv_files, desc="Processing files", leave=False):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, f"converted_{os.path.splitext(file)[0]}.json")
        
        try:
            # Read the motion capture data
            mocap_df = read_mocap_file(input_file)
            if mocap_df is None:
                print(f"\nError: Could not read file {file}")
                continue
                
            # Convert to OpenPose format
            convert_to_mediapipe_format(mocap_df, output_file)
            
        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
            continue

def main():
    """Main function to process all subject directories."""
    # Process subjects from Sub1 to Sub10
    subject_dirs = []
    for subject_num in range(1, 11):
        input_dir = f"Sub{subject_num}/Kinematics"
        if os.path.exists(input_dir):
            subject_dirs.append((subject_num, input_dir))
    
    print(f"Found {len(subject_dirs)} subject directories to process")
    
    # Process each subject directory with progress bar
    for subject_num, input_dir in tqdm(subject_dirs, desc="Processing subjects"):
        print(f"\nProcessing subject {subject_num}...")
        output_dir = "converted_poses"
        process_mocap_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 