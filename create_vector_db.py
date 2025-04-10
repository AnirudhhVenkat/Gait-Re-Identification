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
from extract_gait_cycles import detect_gait_cycles, analyze_cycle
import glob

# Initialize CUDA for PyTorch
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()  # Clear GPU memory
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU - No GPU available")


def load_pose_data(file_path: str) -> dict:
    """Load pose data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_physical_features(pose_data: dict) -> np.ndarray:
    """Calculate physical features from pose data."""
    # Get keypoints
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
    
    # Calculate step width (horizontal distance between ankles)
    step_width = np.abs(right_ankle[0] - left_ankle[0])
    
    # Calculate hip width (horizontal distance between hips)
    hip_width = np.abs(right_hip[0] - left_hip[0])
    
    return np.array([step_width, hip_width])


def calculate_angles(pose_data: dict) -> np.ndarray:
    """Calculate joint angles from pose data."""
    # Get keypoints
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
    
    # Calculate vectors
    right_thigh = right_knee - right_hip
    right_shin = right_ankle - right_knee
    left_thigh = left_knee - left_hip
    left_shin = left_ankle - left_knee
    
    # Calculate angles using dot product
    right_hip_angle = np.arccos(np.dot(right_thigh, right_shin) / 
                               (np.linalg.norm(right_thigh) * np.linalg.norm(right_shin)))
    left_hip_angle = np.arccos(np.dot(left_thigh, left_shin) / 
                              (np.linalg.norm(left_thigh) * np.linalg.norm(left_shin)))
    
    return np.array([right_hip_angle, left_hip_angle])


def calculate_step_length(pose_data: dict) -> float:
    """Calculate step length from pose data."""
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
    
    # Calculate step length (horizontal distance between ankles)
    return np.abs(right_ankle[0] - left_ankle[0])


def calculate_stride_length(pose_data: dict) -> float:
    """Calculate stride length from pose data."""
    right_ankle = np.array([
        pose_data['right_ankle']['x'],
        pose_data['right_ankle']['y'],
        pose_data['right_ankle']['z']
    ])
    
    # Calculate stride length (horizontal distance between consecutive right ankle positions)
    return np.abs(right_ankle[0])


def extract_gait_features(cycle: list) -> np.ndarray:
    """Extract gait features from a complete gait cycle."""
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
        physical = calculate_physical_features(pose_data)
        step_widths.append(physical[0])
        hip_widths.append(physical[1])
        
        # Angular features
        angles = calculate_angles(pose_data)
        right_hip_angles.append(angles[0])
        left_hip_angles.append(angles[1])
        
        # Dynamic features
        step_lengths.append(calculate_step_length(pose_data))
        stride_lengths.append(calculate_stride_length(pose_data))
    
    # Calculate statistics
    features = np.array([
        np.mean(step_widths),      # 0: Step width
        np.mean(hip_widths),       # 1: Hip width
        np.mean(right_hip_angles), # 2: Right hip angle
        np.mean(left_hip_angles),  # 3: Left hip angle
        np.mean(step_lengths),     # 4: Mean step length
        np.std(step_lengths),      # 5: Step length std dev
        np.mean(stride_lengths),   # 6: Mean stride length
        np.std(stride_lengths)     # 7: Stride length std dev
    ])
    
    return features


def load_cycle_features(cycle_dir: str, test_ratio: float = 0.2) -> tuple:
    """Load cycle feature CSV files and split into training and testing sets."""
    # Get all feature CSV files
    csv_files = glob.glob(os.path.join(cycle_dir, '*_features.csv'))
    
    # Group files by subject
    subject_files = {}
    for csv_file in csv_files:
        # Extract subject ID from filename (e.g., "S01" from "S01_01_01_cycle_000_features.csv")
        subject_id = os.path.basename(csv_file).split('_')[0]
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(csv_file)
    
    # Split into train and test sets by subject
    train_files = []
    test_files = []
    for subject_id, files in subject_files.items():
        # Randomly select test files for this subject
        n_test = max(1, int(len(files) * test_ratio))
        test_indices = np.random.choice(len(files), n_test, replace=False)
        for i, file in enumerate(files):
            if i in test_indices:
                test_files.append(file)
            else:
                train_files.append(file)
    
    # Load training data
    train_dfs = []
    for csv_file in train_files:
        df = pd.read_csv(csv_file)
        # Verify subject ID matches filename
        subject_id = os.path.basename(csv_file).split('_')[0]
        if not all(df['subject_id'] == subject_id):
            print(f"Warning: Subject ID mismatch in {csv_file}")
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # Load test data
    test_dfs = []
    for csv_file in test_files:
        df = pd.read_csv(csv_file)
        # Verify subject ID matches filename
        subject_id = os.path.basename(csv_file).split('_')[0]
        if not all(df['subject_id'] == subject_id):
            print(f"Warning: Subject ID mismatch in {csv_file}")
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Loaded {len(train_df)} training samples from {len(train_files)} files")
    print(f"Loaded {len(test_df)} test samples from {len(test_files)} files")
    print(f"Unique subjects in training: {train_df['subject_id'].nunique()}")
    print(f"Unique subjects in test: {test_df['subject_id'].nunique()}")
    
    return train_df, test_df


def create_vector_database(cycle_dir: str, output_dir: str) -> None:
    """Create vector database from gait cycle features."""
    print("Loading and splitting cycle features...")
    train_df, test_df = load_cycle_features(cycle_dir)
    
    # Extract feature columns (excluding metadata)
    feature_columns = [
        'step_width', 'hip_width', 'right_hip_angle', 'left_hip_angle',
        'mean_step_length', 'step_length_std', 'mean_stride_length', 'stride_length_std'
    ]
    
    # Get feature values
    train_features = train_df[feature_columns].values
    test_features = test_df[feature_columns].values
    
    # Normalize features using training data
    print("Normalizing features...")
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)
    
    # Convert to tensors
    train_features_tensor = torch.tensor(train_features_norm, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features_norm, dtype=torch.float32)
    
    # Create metadata
    train_metadata = {
        'subject_ids': train_df['subject_id'].values,
        'trial_ids': train_df['trial_id'].values,
        'cycle_indices': train_df['cycle_index'].values,
        'cycle_lengths': train_df['cycle_length'].values,
        'feature_names': feature_columns
    }
    
    test_metadata = {
        'subject_ids': test_df['subject_id'].values,
        'trial_ids': test_df['trial_id'].values,
        'cycle_indices': test_df['cycle_index'].values,
        'cycle_lengths': test_df['cycle_length'].values,
        'feature_names': feature_columns
    }
    
    # Save vector database
    print("Saving vector database...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    torch.save(train_features_tensor, os.path.join(output_dir, 'train_features.pt'))
    with open(os.path.join(output_dir, 'train_metadata.pkl'), 'wb') as f:
        pickle.dump(train_metadata, f)
    
    # Save test data
    torch.save(test_features_tensor, os.path.join(output_dir, 'test_features.pt'))
    with open(os.path.join(output_dir, 'test_metadata.pkl'), 'wb') as f:
        pickle.dump(test_metadata, f)
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Vector database created with {len(train_features_tensor)} training samples and {len(test_features_tensor)} test samples")
    print(f"Features: {feature_columns}")


def calculate_feature_importance(features: torch.Tensor, metadata: List[Dict]) -> np.ndarray:
    """Calculate feature importance scores using mutual information."""
    # Extract subject IDs from file paths
    subject_ids = []
    for m in metadata:
        # Extract subject ID from file path (e.g., "subject_001_walk_01.json" -> "001")
        file_name = m['file']  # Using 'file' key consistently
        subject_id = file_name.split('_')[1]  # Get the number part
        subject_ids.append(subject_id)
    
    # Convert features to numpy array
    X = features.cpu().numpy()
    y = np.array(subject_ids)
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    
    # Normalize scores to sum to 1
    mi_scores = mi_scores / mi_scores.sum()
    
    return mi_scores


def calculate_feature_statistics(features: torch.Tensor) -> pd.DataFrame:
    """Calculate statistics for each feature."""
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
    
    # Convert to DataFrame
    return pd.DataFrame(stats)


def test_window_sizes(pose_dir: str, output_dir: str, window_sizes: List[int] = [50, 100, 150, 200, 250]) -> None:
    """Test different window sizes and analyze their impact on feature extraction."""
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
        create_vector_database(pose_dir, window_output_dir)
        
        # Load the vector database
        vector_db_path = os.path.join(window_output_dir, 'vector_db.pkl')
        if not os.path.exists(vector_db_path):
            print(f"Warning: Vector database not found at {vector_db_path}")
            continue
            
        # Load using pickle
        with open(vector_db_path, 'rb') as f:
            vector_db = pickle.load(f)
            
        features_tensor = vector_db['features']
        metadata = vector_db['metadata']
        
        # Run feature analysis
        print("Running feature analysis...")
        
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


def main():
    # Set up paths
    cycle_dir = "gait_cycles"
    output_dir = "vector_db"
    
    # Create vector database
    create_vector_database(cycle_dir, output_dir)


if __name__ == "__main__":
    main() 