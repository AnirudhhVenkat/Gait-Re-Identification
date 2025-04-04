import os
import torch
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Tuple
import pickle

def load_vector_database(db_dir: str) -> Tuple[torch.Tensor, List[Dict], StandardScaler]:
    """Load the vector database components."""
    features_path = os.path.join(db_dir, 'gait_features.pt')
    metadata_path = os.path.join(db_dir, 'metadata.pkl')
    scaler_path = os.path.join(db_dir, 'scaler.pkl')
    
    # Load features directly to GPU
    features = torch.load(features_path, map_location='cpu')  # Load to CPU for analysis
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return features, metadata, scaler

def extract_subject_id(filename: str) -> str:
    """Extract subject ID from filename."""
    parts = filename.split('_')
    return parts[1] if len(parts) > 1 else filename

def calculate_feature_importance(features: torch.Tensor, metadata: List[Dict]) -> np.ndarray:
    """Calculate feature importance using mutual information."""
    # Get subject labels
    labels = [extract_subject_id(m['file']) for m in metadata]
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    
    # Convert features to numpy for feature selection
    X = features.cpu().numpy()
    
    # Calculate mutual information scores
    importance_scores = mutual_info_classif(X, y)
    
    return importance_scores

def calculate_feature_statistics(features: torch.Tensor) -> pd.DataFrame:
    """Calculate basic statistics for each feature."""
    X = features.cpu().numpy()
    
    stats = {
        'mean': np.mean(X, axis=0),
        'std': np.std(X, axis=0),
        'min': np.min(X, axis=0),
        'max': np.max(X, axis=0),
        'skewness': pd.DataFrame(X).skew().values,
        'kurtosis': pd.DataFrame(X).kurtosis().values
    }
    
    return pd.DataFrame(stats)

def plot_feature_distributions(features: torch.Tensor, output_dir: str):
    """Plot distribution of each feature."""
    X = features.cpu().numpy()
    n_features = X.shape[1]
    
    # Create subplots
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    # Feature names
    feature_names = [
        'Step width',
        'Hip width',
        'Left hip angle',
        'Right hip angle',
        'Mean step length',
        'Step length std dev',
        'Mean stride length',
        'Stride length std dev'
    ]
    
    for i in range(n_features):
        sns.histplot(X[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f'Feature {i}: {feature_names[i]}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    
    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()

def plot_feature_correlations(features: torch.Tensor, output_dir: str):
    """Plot correlation matrix between features."""
    X = features.cpu().numpy()
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=range(8), yticklabels=range(8))
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = 'feature_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vector database
    print("Loading vector database...")
    features, metadata, scaler = load_vector_database('vector_db')
    
    # Calculate feature importance
    print("Calculating feature importance...")
    importance_scores = calculate_feature_importance(features, metadata)
    
    # Save importance scores
    importance_df = pd.DataFrame({
        'feature_index': range(len(importance_scores)),
        'importance': importance_scores
    })
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Calculate feature statistics
    print("Calculating feature statistics...")
    stats_df = calculate_feature_statistics(features)
    stats_df.to_csv(os.path.join(output_dir, 'feature_statistics.csv'))
    
    # Generate plots
    print("Generating visualizations...")
    plot_feature_distributions(features, output_dir)
    plot_feature_correlations(features, output_dir)
    
    # Print summary
    print("\nFeature Analysis Summary:")
    print("\nFeature Importance Scores:")
    for i, score in enumerate(importance_scores):
        print(f"Feature {i}: {score:.4f}")
    
    print("\nFeature Statistics:")
    print(stats_df)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 