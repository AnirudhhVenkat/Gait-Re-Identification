import numpy as np
import pandas as pd
import faiss
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import joblib
from sklearn.manifold import TSNE
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

def load_gait_analysis(file_path: str) -> Dict:
    """Load gait analysis data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_features(analysis: Dict, subject_id: str) -> np.ndarray:
    """Extract features from gait analysis data with subject-specific normalization."""
    # Get real-time features
    real_time_features = analysis['real_time_features']
    
    # Calculate statistics from real-time features
    joint_angles = []
    structural_measurements = []
    dynamic_measurements = []
    symmetry_scores = []
    
    for frame in real_time_features:
        # Extract joint angles
        joint_angles.append([
            frame['joint_angles']['left_hip'],
            frame['joint_angles']['right_hip'],
            frame['joint_angles']['left_knee'],
            frame['joint_angles']['right_knee'],
            frame['joint_angles']['left_ankle'],
            frame['joint_angles']['right_ankle']
        ])
        
        # Extract structural measurements
        structural_measurements.append([
            frame['step_width'],
            frame['hip_width']
        ])
        
        # Extract dynamic measurements (without cadence)
        dynamic_measurements.append([
            frame['step_length'],
            frame['walking_speed']
        ])
        
        # Extract symmetry score
        symmetry_scores.append(frame['symmetry'])
    
    # Convert to numpy arrays
    joint_angles = np.array(joint_angles)
    structural_measurements = np.array(structural_measurements)
    dynamic_measurements = np.array(dynamic_measurements)
    symmetry_scores = np.array(symmetry_scores)
    
    # Subject-specific normalization
    # Use the first trial's measurements as reference for this subject
    reference_measurements = {
        'step_width': np.mean(structural_measurements[:, 0]),
        'hip_width': np.mean(structural_measurements[:, 1]),
        'step_length': np.mean(dynamic_measurements[:, 0])
    }
    
    # Normalize structural measurements relative to subject's reference
    structural_measurements[:, 0] /= reference_measurements['step_width']
    structural_measurements[:, 1] /= reference_measurements['hip_width']
    dynamic_measurements[:, 0] /= reference_measurements['step_length']
    
    # Calculate statistics
    features = []
    
    # Joint angle statistics
    for i in range(6):  # 6 joint angles
        features.extend([
            float(np.mean(joint_angles[:, i])),
            float(np.std(joint_angles[:, i]))
        ])
    
    # Structural measurement statistics (now normalized)
    for i in range(2):  # step_width and hip_width
        features.extend([
            float(np.mean(structural_measurements[:, i])),
            float(np.std(structural_measurements[:, i]))
        ])
    
    # Dynamic measurement statistics (without cadence)
    for i in range(2):  # step_length and walking_speed
        features.extend([
            float(np.mean(dynamic_measurements[:, i])),
            float(np.std(dynamic_measurements[:, i]))
        ])
    
    # Symmetry statistics
    features.extend([
        float(np.mean(symmetry_scores)),
        float(np.std(symmetry_scores))
    ])
    
    # Add temporal features
    temporal_features = analysis['temporal_features']
    features.extend([
        float(temporal_features['cycle_durations']['left']['mean']),
        float(temporal_features['cycle_durations']['left']['std']),
        float(temporal_features['cycle_durations']['right']['mean']),
        float(temporal_features['cycle_durations']['right']['std'])
    ])
    
    # Add dynamic features
    dynamic_features = analysis['dynamic_features']
    features.extend([
        float(dynamic_features['step_lengths']['mean']),
        float(dynamic_features['step_lengths']['std']),
        float(dynamic_features['stride_lengths']['mean']),
        float(dynamic_features['stride_lengths']['std'])
    ])
    
    return np.array(features, dtype=np.float32)

def create_embeddings(features: np.ndarray, subject_id: int, model) -> np.ndarray:
    """Create embeddings from features using the neural network model with subject information."""
    # Reshape features for the model (add batch dimension)
    features = features.reshape(1, -1)
    subject_id = np.array([subject_id])
    
    # Get embeddings from the model
    embeddings = model.predict([features, subject_id])
    return embeddings

def analyze_similarities(embeddings: np.ndarray, k: int = 5) -> Dict:
    """Analyze similarities between embeddings with enhanced subject-specific metrics."""
    # Calculate cosine similarities (embeddings are already L2 normalized)
    similarities = np.dot(embeddings, embeddings.T)
    
    # Get top-k nearest neighbors for each embedding
    k = min(k, len(embeddings))
    nearest_neighbors = []
    
    for i in range(len(embeddings)):
        # Get indices of k nearest neighbors (excluding self)
        indices = np.argsort(similarities[i])[-k-1:-1][::-1]
        # Get similarity scores
        scores = similarities[i][indices]
        nearest_neighbors.append({
            'indices': indices.tolist(),
            'scores': scores.tolist()
        })
    
    return {
        'similarities': similarities.tolist(),
        'nearest_neighbors': nearest_neighbors,
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities))
    }

def visualize_embeddings(embeddings: np.ndarray, file_info: List[Dict], output_dir: Path) -> None:
    """Visualize embeddings using t-SNE and save as PNG."""
    # Extract subject IDs for coloring
    subject_ids = [info['subject_id'] for info in file_info]
    
    # Apply t-SNE to reduce dimensionality to 2D
    print("Applying t-SNE to reduce dimensionality...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by subject
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=subject_ids, cmap='tab10', alpha=0.6)
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Subject")
    plt.gca().add_artist(legend1)
    
    # Add title and labels
    plt.title("t-SNE Visualization of Gait Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_dir / "embeddings_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Embedding visualization saved to {output_dir / 'embeddings_visualization.png'}")

def save_feature_info(output_dir: Path) -> None:
    """Save detailed information about the features being used."""
    feature_info = {
        "total_features": 30,
        "feature_groups": [
            {
                "name": "Joint Angles",
                "count": 12,
                "features": [
                    "Left Hip (mean, std)",
                    "Right Hip (mean, std)",
                    "Left Knee (mean, std)",
                    "Right Knee (mean, std)",
                    "Left Ankle (mean, std)",
                    "Right Ankle (mean, std)"
                ]
            },
            {
                "name": "Structural Measurements",
                "count": 4,
                "features": [
                    "Step Width (mean, std)",
                    "Hip Width (mean, std)"
                ]
            },
            {
                "name": "Dynamic Measurements",
                "count": 4,
                "features": [
                    "Step Length (mean, std)",
                    "Walking Speed (mean, std)"
                ]
            },
            {
                "name": "Symmetry",
                "count": 2,
                "features": [
                    "Symmetry Score (mean, std)"
                ]
            },
            {
                "name": "Temporal Features",
                "count": 4,
                "features": [
                    "Left Cycle Duration (mean, std)",
                    "Right Cycle Duration (mean, std)"
                ]
            },
            {
                "name": "Dynamic Features",
                "count": 4,
                "features": [
                    "Step Lengths (mean, std)",
                    "Stride Lengths (mean, std)"
                ]
            }
        ],
        "normalization": {
            "subject_specific": [
                "Step Width",
                "Hip Width",
                "Step Length"
            ],
            "global": [
                "Joint Angles",
                "Walking Speed",
                "Symmetry",
                "Cycle Durations"
            ]
        },
        "excluded_features": [
            "Cadence"
        ]
    }
    
    # Save feature information
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"Feature information saved to {output_dir / 'feature_info.json'}")

def create_vector_db(input_dir: str, output_dir: str, model) -> None:
    """Create vector database from gait analysis data with subject-specific processing."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save feature information
    save_feature_info(output_dir)
    
    # Load all gait analysis files
    input_dir = Path(input_dir)
    analysis_files = list(input_dir.glob("analysis_*.json"))
    
    if not analysis_files:
        print(f"No gait analysis files found in {input_dir}")
        return
    
    # Extract features and create embeddings
    features_list = []
    embeddings_list = []
    file_info = []
    
    # Group files by subject
    subject_files = {}
    for file_path in analysis_files:
        # Extract subject from filename (e.g., "analysis_converted_Sub9_Kinematics_T1.json")
        parts = file_path.stem.split('_')
        subject = parts[2]  # This will be "SubX"
        if subject not in subject_files:
            subject_files[subject] = []
        subject_files[subject].append(file_path)
    
    # Process each subject's files
    for subject, files in tqdm(subject_files.items(), desc="Processing subjects"):
        # Extract subject ID (e.g., "Sub9" -> 9)
        subject_id = int(subject.replace('Sub', ''))
        
        for file_path in tqdm(files, desc=f"Processing {subject}", leave=False):
            try:
                # Load analysis data
                analysis = load_gait_analysis(str(file_path))
                
                # Extract features with subject-specific normalization
                features = extract_features(analysis, subject)
                
                # Verify feature dimension
                if features.shape[0] != 30:
                    print(f"Warning: Skipping {file_path} - incorrect feature dimension: {features.shape[0]}")
                    continue
                    
                features_list.append(features)
                
                # Create embeddings with subject information
                embeddings = create_embeddings(features, subject_id, model)
                embeddings_list.append(embeddings)
                
                # Store file info with more details
                file_info.append({
                    'file_path': str(file_path),
                    'subject': subject,
                    'subject_id': subject_id,
                    'trial': file_path.stem.split('_')[-1],
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    if not features_list:
        print("No valid features extracted")
        return
    
    # Convert lists to numpy arrays
    features_array = np.vstack(features_list)
    embeddings_array = np.vstack(embeddings_list)
    
    # Create FAISS index with cosine similarity
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Using inner product for cosine similarity
    index.add(embeddings_array.astype('float32'))
    
    # Save index and metadata
    faiss.write_index(index, str(output_dir / "gait_index.faiss"))
    
    # Save file info
    with open(output_dir / "file_info.json", 'w') as f:
        json.dump(file_info, f, indent=2)
    
    # Analyze similarities with subject-specific metrics
    similarities = analyze_similarities(embeddings_array)
    
    # Save similarity analysis
    with open(output_dir / "similarity_analysis.json", 'w') as f:
        json.dump(similarities, f, indent=2)
    
    # Calculate and save accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(similarities, file_info)
    with open(output_dir / "accuracy_metrics.json", 'w') as f:
        json.dump(accuracy_metrics, f, indent=2)
    
    # Save subject-specific statistics
    subject_stats = calculate_subject_stats(embeddings_array, file_info)
    with open(output_dir / "subject_stats.json", 'w') as f:
        json.dump(subject_stats, f, indent=2)
    
    # Visualize and save embeddings
    visualize_embeddings(embeddings_array, file_info, output_dir)
    
    print(f"Vector database created successfully in {output_dir}")
    print(f"Total embeddings: {len(embeddings_array)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Feature dimension: {features_array.shape[1]}")
    print(f"Number of subjects: {len(subject_files)}")

def calculate_accuracy_metrics(similarities: Dict, file_info: List[Dict]) -> Dict:
    """Calculate accuracy metrics for the similarity analysis."""
    # Extract subject IDs
    subjects = [info['subject'] for info in file_info]
    
    # Calculate metrics for each embedding
    metrics = []
    for i, neighbors in enumerate(similarities['nearest_neighbors']):
        # Get subject of current embedding
        current_subject = subjects[i]
        
        # Get subjects of neighbors
        neighbor_subjects = [subjects[idx] for idx in neighbors['indices']]
        
        # Calculate match confidence (percentage of neighbors with same subject)
        match_confidence = sum(1 for s in neighbor_subjects if s == current_subject) / len(neighbor_subjects)
        
        # Calculate similarity confidence (average similarity score)
        similarity_confidence = np.mean(neighbors['scores'])
        
        # Combined confidence (weighted average)
        combined_confidence = 0.7 * match_confidence + 0.3 * similarity_confidence
        
        metrics.append({
            'index': i,
            'subject': current_subject,
            'match_confidence': float(match_confidence),
            'similarity_confidence': float(similarity_confidence),
            'combined_confidence': float(combined_confidence)
        })
    
    # Calculate overall metrics
    match_accuracies = [m['match_confidence'] for m in metrics]
    similarity_confidences = [m['similarity_confidence'] for m in metrics]
    combined_confidences = [m['combined_confidence'] for m in metrics]
    
    return {
        'match_accuracy': {
            'mean': float(np.mean(match_accuracies)),
            'std': float(np.std(match_accuracies))
        },
        'similarity_confidence': {
            'mean': float(np.mean(similarity_confidences)),
            'std': float(np.std(similarity_confidences))
        },
        'combined_confidence': {
            'mean': float(np.mean(combined_confidences)),
            'std': float(np.std(combined_confidences))
        },
        'detailed_metrics': metrics
    }

def calculate_subject_stats(embeddings: np.ndarray, file_info: List[Dict]) -> Dict:
    """Calculate subject-specific statistics for the embeddings."""
    # Group embeddings by subject
    subject_embeddings = {}
    for i, info in enumerate(file_info):
        subject = info['subject']
        if subject not in subject_embeddings:
            subject_embeddings[subject] = []
        subject_embeddings[subject].append(embeddings[i])
    
    # Calculate statistics for each subject
    stats = {}
    for subject, emb_list in subject_embeddings.items():
        emb_array = np.array(emb_list)
        stats[subject] = {
            'num_samples': len(emb_array),
            'mean_embedding': emb_array.mean(axis=0).tolist(),
            'std_embedding': emb_array.std(axis=0).tolist(),
            'intra_subject_similarity': float(np.mean(np.dot(emb_array, emb_array.T)))
        }
    
    return stats

def create_model(input_dim: int = 30, embedding_dim: int = 128, num_subjects: int = 10) -> Model:
    """Create and compile the neural network model for subject identification."""
    # Input layers
    feature_input = Input(shape=(input_dim,))
    subject_input = Input(shape=(1,), dtype='int32')
    
    # Subject embedding layer (increased dimension for better discrimination)
    subject_embedding = Embedding(num_subjects, 32)(subject_input)
    subject_embedding = Flatten()(subject_embedding)
    
    # Feature processing with increased capacity
    x = Dense(512, activation='relu')(feature_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Combine with subject embedding
    x = Concatenate()([x, subject_embedding])
    
    # Additional layers for better feature extraction
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(192, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Final embedding layer with L2 normalization
    outputs = Dense(embedding_dim, activation='linear')(x)
    outputs = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(outputs)
    
    # Create model
    model = Model(inputs=[feature_input, subject_input], outputs=outputs)
    
    # Compile model with triplet loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    # Define directories
    input_dir = "gait_analysis"
    output_dir = "vector_db"
    
    # Create and train the model
    model = create_model()
    
    # Create vector database
    create_vector_db(input_dir, output_dir, model)

if __name__ == "__main__":
    main() 