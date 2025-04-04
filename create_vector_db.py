import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
import gc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Initialize CUDA for PyTorch
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()  # Clear GPU memory
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU - No GPU available")


def load_gait_analysis(file_path: str) -> Dict[str, Any]:
    """Load gait analysis data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing gait analysis data.
        
    Returns:
        Dictionary containing the gait analysis data.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate the angle between three points.
    
    Args:
        p1: First point coordinates.
        p2: Second point coordinates (vertex).
        p3: Third point coordinates.
        
    Returns:
        Angle in degrees between the three points.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def calculate_physical_features(pose_data: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate physical features from pose data.
    
    Args:
        pose_data: Dictionary containing pose keypoints.
        
    Returns:
        Tuple containing step width and hip width.
    """
    left_ankle = np.array([
        pose_data['left_ankle']['x'],
        pose_data['left_ankle']['y'],
        pose_data['left_ankle']['z']
    ])
    right_ankle = np.array([
        pose_data['right_ankle']['x'],
        pose_data['right_ankle']['y'],
        pose_data['right_ankle']['z']
    ])
    left_hip = np.array([
        pose_data['left_hip']['x'],
        pose_data['left_hip']['y'],
        pose_data['left_hip']['z']
    ])
    right_hip = np.array([
        pose_data['right_hip']['x'],
        pose_data['right_hip']['y'],
        pose_data['right_hip']['z']
    ])
    
    step_width = np.abs(left_ankle[0] - right_ankle[0])
    hip_width = np.abs(left_hip[0] - right_hip[0])
    
    return step_width, hip_width


def calculate_angles(pose_data: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate hip angles from pose data.
    
    Args:
        pose_data: Dictionary containing pose keypoints.
        
    Returns:
        Tuple containing left and right hip angles in degrees.
    """
    left_hip = np.array([
        pose_data['left_hip']['x'],
        pose_data['left_hip']['y'],
        pose_data['left_hip']['z']
    ])
    left_knee = np.array([
        pose_data['left_knee']['x'],
        pose_data['left_knee']['y'],
        pose_data['left_knee']['z']
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
    right_knee = np.array([
        pose_data['right_knee']['x'],
        pose_data['right_knee']['y'],
        pose_data['right_knee']['z']
    ])
    right_ankle = np.array([
        pose_data['right_ankle']['x'],
        pose_data['right_ankle']['y'],
        pose_data['right_ankle']['z']
    ])
    
    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    return left_angle, right_angle


def calculate_step_length(pose_data: Dict[str, Any], 
                         prev_pose_data: Optional[Dict[str, Any]] = None) -> float:
    """Calculate step length from current and previous pose data.
    
    Args:
        pose_data: Current pose data.
        prev_pose_data: Previous pose data, if available.
        
    Returns:
        Step length, or 0 if previous pose data is not available.
    """
    if prev_pose_data is None:
        return 0.0
    
    current_ankle = np.array([
        pose_data['right_ankle']['x'],
        pose_data['right_ankle']['y'],
        pose_data['right_ankle']['z']
    ])
    prev_ankle = np.array([
        prev_pose_data['right_ankle']['x'],
        prev_pose_data['right_ankle']['y'],
        prev_pose_data['right_ankle']['z']
    ])
    
    return np.linalg.norm(current_ankle - prev_ankle)


def calculate_stride_length(pose_data: Dict[str, Any], 
                          prev_pose_data: Optional[Dict[str, Any]] = None) -> float:
    """Calculate stride length from current and previous pose data.
    
    Args:
        pose_data: Current pose data.
        prev_pose_data: Previous pose data, if available.
        
    Returns:
        Stride length, or 0 if previous pose data is not available.
    """
    if prev_pose_data is None:
        return 0.0
    
    current_ankle = np.array([
        pose_data['left_ankle']['x'],
        pose_data['left_ankle']['y'],
        pose_data['left_ankle']['z']
    ])
    prev_ankle = np.array([
        prev_pose_data['left_ankle']['x'],
        prev_pose_data['left_ankle']['y'],
        prev_pose_data['left_ankle']['z']
    ])
    
    return np.linalg.norm(current_ankle - prev_ankle)


def extract_gait_features_batch(pose_data_list: List[Dict[str, Any]]) -> np.ndarray:
    """Extract gait features from a batch of pose data.
    
    Args:
        pose_data_list: List of pose data dictionaries.
        
    Returns:
        Array of extracted gait features averaged over the window.
    """
    features = []
    prev_pose_data = None
    
    for pose_data in pose_data_list:
        # Calculate physical features
        step_width, hip_width = calculate_physical_features(pose_data)
        
        # Calculate angles
        left_angle, right_angle = calculate_angles(pose_data)
        
        # Calculate step and stride lengths
        step_length = calculate_step_length(pose_data, prev_pose_data)
        stride_length = calculate_stride_length(pose_data, prev_pose_data)
        
        # Store features
        features.append([
            step_width,
            hip_width,
            left_angle,
            right_angle,
            step_length,
            step_length,  # Using same value for std dev placeholder
            stride_length,
            stride_length  # Using same value for std dev placeholder
        ])
        
        prev_pose_data = pose_data
    
    # Convert to numpy array and average over the window
    features = np.array(features)
    return np.mean(features, axis=0)  # Average over the window dimension


def test_window_sizes(pose_dir: str, output_dir: str, window_sizes: List[int] = [50, 100, 150, 200, 250]) -> None:
    """Test different window sizes and analyze their impact on feature extraction.
    
    Args:
        pose_dir: Directory containing pose data files.
        output_dir: Directory to save the vector database.
        window_sizes: List of window sizes to test (in frames at 100 FPS):
            - 50 frames = 0.5 seconds (half step cycle)
            - 100 frames = 1.0 seconds (one full step cycle)
            - 150 frames = 1.5 seconds (one and a half step cycles)
            - 200 frames = 2.0 seconds (two full step cycles)
            - 250 frames = 2.5 seconds (two and a half step cycles)
    """
    # Create output directory for window size analysis
    analysis_dir = os.path.join(output_dir, 'window_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Initialize results storage
    results = {
        'window_sizes': [],
        'num_sequences': [],
        'feature_importance': [],
        'feature_statistics': []
    }
    
    # Test each window size
    for window_size in window_sizes:
        print(f"\nTesting window size: {window_size}")
        
        # Create vector database with current window size
        window_output_dir = os.path.join(output_dir, f'window_{window_size}')
        features_tensor, metadata = create_vector_database(
            pose_dir, 
            window_output_dir,
            window_size
        )
        
        if features_tensor is None:
            continue
        
        # Run feature analysis
        print("Running feature analysis...")
        from analyze_features import load_vector_database, calculate_feature_importance, calculate_feature_statistics
        
        # Calculate feature importance
        importance_scores = calculate_feature_importance(features_tensor, metadata)
        
        # Calculate feature statistics
        stats_df = calculate_feature_statistics(features_tensor)
        
        # Store results
        results['window_sizes'].append(window_size)
        results['num_sequences'].append(len(features_tensor))
        results['feature_importance'].append(importance_scores)
        results['feature_statistics'].append(stats_df)
        
        # Save individual window size results
        window_analysis_dir = os.path.join(analysis_dir, f'window_{window_size}')
        os.makedirs(window_analysis_dir, exist_ok=True)
        
        # Save importance scores
        importance_df = pd.DataFrame({
            'feature_index': range(len(importance_scores)),
            'importance': importance_scores
        })
        importance_df.to_csv(os.path.join(window_analysis_dir, 'feature_importance.csv'), index=False)
        
        # Save statistics
        stats_df.to_csv(os.path.join(window_analysis_dir, 'feature_statistics.csv'))
        
        # Generate plots
        from analyze_features import plot_feature_distributions, plot_feature_correlations
        plot_feature_distributions(features_tensor, window_analysis_dir)
        plot_feature_correlations(features_tensor, window_analysis_dir)
    
    # Plot comparative results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Number of sequences
    plt.subplot(2, 2, 1)
    plt.plot(results['window_sizes'], results['num_sequences'], 'bo-')
    plt.xlabel('Window Size')
    plt.ylabel('Number of Sequences')
    plt.title('Sequences vs Window Size')
    
    # Plot 2: Feature importance comparison
    plt.subplot(2, 2, 2)
    for i in range(len(results['feature_importance'][0])):
        importance = [imp[i] for imp in results['feature_importance']]
        plt.plot(results['window_sizes'], importance, label=f'Feature {i}', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance vs Window Size')
    plt.legend()
    
    # Plot 3: Feature means comparison
    plt.subplot(2, 2, 3)
    for i in range(len(results['feature_statistics'][0])):
        means = [stats['mean'].iloc[i] for stats in results['feature_statistics']]
        plt.plot(results['window_sizes'], means, label=f'Feature {i}', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('Feature Mean')
    plt.title('Feature Means vs Window Size')
    plt.legend()
    
    # Plot 4: Feature stds comparison
    plt.subplot(2, 2, 4)
    for i in range(len(results['feature_statistics'][0])):
        stds = [stats['std'].iloc[i] for stats in results['feature_statistics']]
        plt.plot(results['window_sizes'], stds, label=f'Feature {i}', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('Feature Standard Deviation')
    plt.title('Feature Standard Deviations vs Window Size')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'window_size_comparison.png'))
    plt.close()
    
    # Save comprehensive results summary
    with open(os.path.join(analysis_dir, 'results_summary.txt'), 'w') as f:
        f.write("Window Size Analysis Results\n")
        f.write("==========================\n\n")
        
        for i, window_size in enumerate(results['window_sizes']):
            f.write(f"Window Size: {window_size}\n")
            f.write(f"Number of Sequences: {results['num_sequences'][i]}\n")
            
            f.write("\nFeature Importance:\n")
            for j, importance in enumerate(results['feature_importance'][i]):
                f.write(f"  Feature {j}: {importance:.4f}\n")
            
            f.write("\nFeature Statistics:\n")
            stats = results['feature_statistics'][i]
            for j in range(len(stats)):
                f.write(f"  Feature {j}:\n")
                f.write(f"    Mean: {stats['mean'].iloc[j]:.4f}\n")
                f.write(f"    Std: {stats['std'].iloc[j]:.4f}\n")
                f.write(f"    Min: {stats['min'].iloc[j]:.4f}\n")
                f.write(f"    Max: {stats['max'].iloc[j]:.4f}\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"\nWindow size analysis complete. Results saved to {analysis_dir}/")


def calculate_feature_importance(features: torch.Tensor, metadata: List[Dict]) -> np.ndarray:
    """Calculate feature importance scores using mutual information.
    
    Args:
        features: Tensor of shape (n_samples, n_features)
        metadata: List of metadata dictionaries
        
    Returns:
        Array of feature importance scores
    """
    # Extract subject IDs from metadata
    subject_ids = [m['subject_id'] for m in metadata]
    
    # Convert features to numpy array
    X = features.cpu().numpy()
    y = np.array(subject_ids)
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    
    # Normalize scores to sum to 1
    mi_scores = mi_scores / mi_scores.sum()
    
    return mi_scores


def calculate_feature_statistics(features: torch.Tensor) -> List[Dict]:
    """Calculate statistics for each feature.
    
    Args:
        features: Tensor of shape (n_samples, n_features)
        
    Returns:
        List of dictionaries containing statistics for each feature
    """
    # Convert features to numpy array
    X = features.cpu().numpy()
    
    # Calculate statistics for each feature
    stats = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        stats.append({
            'mean': np.mean(feature),
            'std': np.std(feature),
            'min': np.min(feature),
            'max': np.max(feature)
        })
    
    return stats


def create_vector_database(pose_dir: str, output_dir: str, window_size: int = 60) -> Tuple[torch.Tensor, List[Dict], StandardScaler]:
    """Create a vector database from pose data files.
    
    Args:
        pose_dir: Directory containing pose data files.
        output_dir: Directory to save the vector database.
        window_size: Size of the sliding window for feature extraction.
        
    Returns:
        Tuple containing the features tensor, metadata, and scaler.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of pose files
    pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.json')]
    print(f"Found {len(pose_files)} pose files to process")
    
    # Initialize lists for features and metadata
    all_features = []
    metadata = []
    
    # Process each file
    for file_name in tqdm(pose_files, desc="Processing files"):
        file_path = os.path.join(pose_dir, file_name)
        
        try:
            # Load gait analysis data
            gait_data = load_gait_analysis(file_path)
            
            # Get pose landmarks
            pose_landmarks = gait_data.get('pose_landmarks', [])
            if not pose_landmarks:
                print(f"Warning: No pose landmarks found in {file_name}")
                continue
            
            # Process each window
            for i in range(0, len(pose_landmarks), window_size):
                window = pose_landmarks[i:i + window_size]
                if len(window) < window_size:
                    continue
                
                # Extract features for this window
                window_features = extract_gait_features_batch(window)
                
                # Store features and metadata
                all_features.append(window_features)
                metadata.append({
                    'file': file_name,
                    'start_frame': i,
                    'end_frame': i + window_size
                })
                
                # Clear memory periodically
                if len(all_features) % 1000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    
    if not all_features:
        print("No features were extracted. Please check the input data and try again.")
        return None, [], None
    
    # Convert features to numpy array
    all_features = np.array(all_features)
    
    # Normalize features
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    
    # Convert to tensor
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    
    # Save vector database
    torch.save(features_tensor, os.path.join(output_dir, 'gait_features.pt'))
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Vector database created with {len(features_tensor)} sequences")
    return features_tensor, metadata, scaler


if __name__ == "__main__":
    pose_dir = "converted_poses"
    output_dir = "vector_db"
    
    # Test different window sizes
    test_window_sizes(pose_dir, output_dir) 