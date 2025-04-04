import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import heapq
from collections import defaultdict
from create_vector_db import create_vector_database, calculate_feature_importance, calculate_feature_statistics

# Set device and maximize GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configure PyTorch for maximum GPU usage
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = torch.cuda.memory_allocated(0)
    print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
    print(f"Free GPU memory: {(total_memory - free_memory) / 1024**3:.2f} GB")

def load_vector_database(db_dir: str) -> Tuple[torch.Tensor, List[Dict], StandardScaler]:
    """Load the vector database components."""
    features_path = os.path.join(db_dir, 'gait_features.pt')
    metadata_path = os.path.join(db_dir, 'metadata.pkl')
    scaler_path = os.path.join(db_dir, 'scaler.pkl')
    
    # Load features directly to GPU
    features = torch.load(features_path, map_location=device)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return features, metadata, scaler

def extract_subject_id(filename: str) -> str:
    """Extract subject ID from filename."""
    parts = filename.split('_')
    return parts[1] if len(parts) > 1 else filename

def select_features(features: torch.Tensor, metadata: List[Dict], n_features: int = 50) -> Tuple[torch.Tensor, List[int]]:
    """Select the most discriminative features using mutual information."""
    # Get subject labels
    labels = [extract_subject_id(m['file']) for m in metadata]
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    
    # Convert features to numpy for feature selection
    X = features.cpu().numpy()
    
    # Select top features using mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    
    # Return selected features and their indices
    return features[:, selected_indices], selected_indices

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize features using robust scaling."""
    # Convert to numpy for scaling
    X = features.cpu().numpy()
    
    # Apply robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to tensor and move to device
    return torch.tensor(X_scaled, dtype=torch.float32, device=device)

def find_top_k_matches_gpu_optimized(query_features: torch.Tensor, 
                                   database_features: torch.Tensor,
                                   metadata: List[Dict], 
                                   k: int = 5, 
                                   batch_size: int = 1024) -> List[List[Dict]]:
    """Find top-k matches using GPU-optimized approach with weighted cosine similarity.
    Features used (with importance weights):
    0. Step width (0.0254)
    1. Hip width (0.0086)
    2. Left hip angle (0.0622)
    3. Right hip angle (0.0516)
    4. Mean step length (0.0083)
    5. Step length std dev (0.0076)
    6. Mean stride length (0.0218)
    7. Stride length std dev (0.0176)
    """
    n_queries = len(query_features)
    results = []
    
    # Feature importance weights based on analysis
    weights = torch.tensor([
        0.0254,  # Step width
        0.0086,  # Hip width
        0.0622,  # Left hip angle (most important)
        0.0516,  # Right hip angle
        0.0083,  # Mean step length
        0.0076,  # Step length std dev
        0.0218,  # Mean stride length
        0.0176   # Stride length std dev
    ], device=device)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Calculate optimal batch size
    if device.type == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        memory_per_query = database_features.element_size() * database_features.shape[1] * 2
        optimal_batch_size = min(batch_size, available_memory // memory_per_query)
        print(f"Using batch size: {optimal_batch_size}")
    else:
        optimal_batch_size = batch_size
    
    # Process queries in batches
    for i in tqdm(range(0, n_queries, optimal_batch_size), desc="Processing queries"):
        batch_end = min(i + optimal_batch_size, n_queries)
        query_batch = query_features[i:batch_end]
        
        # Apply feature weights
        query_weighted = query_batch * weights
        db_weighted = database_features * weights
        
        # Normalized weighted cosine similarity
        query_norm = torch.nn.functional.normalize(query_weighted, p=2, dim=1)
        db_norm = torch.nn.functional.normalize(db_weighted, p=2, dim=1)
        batch_similarity = torch.mm(query_norm, db_norm.t())
        
        # Get top-k matches
        batch_top_k_values, batch_top_k_indices = torch.topk(batch_similarity, k=min(k, batch_similarity.shape[1]), dim=1)
        
        # Process results
        for j in range(len(query_batch)):
            matches = []
            for l in range(k):
                idx = batch_top_k_indices[j][l].item()
                matches.append({
                    'index': idx,
                    'similarity': batch_top_k_values[j][l].item(),
                    'subject_id': extract_subject_id(metadata[idx]['file']),
                    'file': metadata[idx]['file'],
                    'start_frame': metadata[idx]['start_frame'],
                    'end_frame': metadata[idx]['end_frame']
                })
            results.append(matches)
        
        # Clear memory
        del batch_similarity, batch_top_k_values, batch_top_k_indices
        torch.cuda.empty_cache()
    
    return results

def evaluate_matching_accuracy(results: List[List[Dict]], query_metadata: List[Dict]) -> Dict:
    """Evaluate matching accuracy metrics."""
    true_subjects = [extract_subject_id(m['file']) for m in query_metadata]
    predicted_subjects = [r[0]['subject_id'] for r in results]  # Top-1 prediction
    
    accuracy = accuracy_score(true_subjects, predicted_subjects)
    precision = precision_score(true_subjects, predicted_subjects, average='weighted')
    recall = recall_score(true_subjects, predicted_subjects, average='weighted')
    f1 = f1_score(true_subjects, predicted_subjects, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_subjects, predicted_subjects)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_similarity_distribution(results: List[List[Dict]], output_dir: str):
    """Plot distribution of similarity scores."""
    similarities = [r[0]['similarity'] for r in results]  # Top-1 similarity scores
    
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, kde=True)
    plt.title('Distribution of Top-1 Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, output_dir: str):
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Subject')
    plt.ylabel('True Subject')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    # Set up paths
    pose_dir = "converted_poses"
    vector_db_path = "vector_db/window_60/vector_database.pt"
    metrics_path = "vector_db/window_60/metrics.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    
    # Load or create vector database
    if os.path.exists(vector_db_path):
        print("Loading existing vector database...")
        vector_db = torch.load(vector_db_path)
        features = vector_db['features']
        metadata = vector_db['metadata']
        scaler = vector_db['scaler']
    else:
        print("Creating new vector database...")
        features, metadata, scaler = create_vector_database(pose_dir, os.path.dirname(vector_db_path), window_size=60)
        vector_db = {
            'features': features,
            'metadata': metadata,
            'scaler': scaler
        }
        torch.save(vector_db, vector_db_path)
    
    # Print database statistics
    print(f"\nDatabase Statistics:")
    print(f"Number of sequences: {len(metadata)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Window size: 60 frames (0.6 seconds at 100 FPS)")
    
    # Calculate feature importance
    print("\nCalculating feature importance...")
    importance_scores = calculate_feature_importance(features, metadata)
    
    # Print feature importance
    print("\nFeature Importance Scores:")
    for i, score in enumerate(importance_scores):
        print(f"Feature {i}: {score:.4f}")
    
    # Calculate feature statistics
    print("\nCalculating feature statistics...")
    feature_stats = calculate_feature_statistics(features)
    
    # Print feature statistics
    print("\nFeature Statistics:")
    for i, stats in enumerate(feature_stats):
        print(f"Feature {i}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        f.write("Gait Recognition System Metrics\n")
        f.write("=============================\n\n")
        f.write(f"Database Statistics:\n")
        f.write(f"Number of sequences: {len(metadata)}\n")
        f.write(f"Feature dimension: {features.shape[1]}\n")
        f.write(f"Window size: 60 frames (0.6 seconds at 100 FPS)\n\n")
        f.write("Feature Importance Scores:\n")
        for i, score in enumerate(importance_scores):
            f.write(f"Feature {i}: {score:.4f}\n")
        f.write("\nFeature Statistics:\n")
        for i, stats in enumerate(feature_stats):
            f.write(f"Feature {i}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n")
            f.write(f"  Min: {stats['min']:.4f}\n")
            f.write(f"  Max: {stats['max']:.4f}\n")
    
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 