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
from sklearn.metrics import precision_recall_fscore_support

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

def load_vector_database(vector_db_dir: str) -> tuple:
    """Load vector database components."""
    try:
        # Load training data
        train_features = torch.load(os.path.join(vector_db_dir, 'train_features.pt'))
        with open(os.path.join(vector_db_dir, 'train_metadata.pkl'), 'rb') as f:
            train_metadata = pickle.load(f)
        
        # Load test data
        test_features = torch.load(os.path.join(vector_db_dir, 'test_features.pt'))
        with open(os.path.join(vector_db_dir, 'test_metadata.pkl'), 'rb') as f:
            test_metadata = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(vector_db_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Validate data
        if len(train_features) == 0 or len(test_features) == 0:
            raise ValueError("Empty feature tensors found")
        
        if len(train_metadata['subject_ids']) != len(train_features):
            raise ValueError("Mismatch between training features and metadata")
        
        if len(test_metadata['subject_ids']) != len(test_features):
            raise ValueError("Mismatch between test features and metadata")
        
        # Verify subject IDs
        train_subjects = set(train_metadata['subject_ids'])
        test_subjects = set(test_metadata['subject_ids'])
        
        if not train_subjects or not test_subjects:
            raise ValueError("No subject IDs found in metadata")
        
        print(f"Training subjects: {sorted(train_subjects)}")
        print(f"Test subjects: {sorted(test_subjects)}")
        print(f"Training samples per subject: {pd.Series(train_metadata['subject_ids']).value_counts().to_dict()}")
        print(f"Test samples per subject: {pd.Series(test_metadata['subject_ids']).value_counts().to_dict()}")
        
        return train_features, train_metadata, test_features, test_metadata, scaler
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found in {vector_db_dir}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading vector database: {str(e)}")

def calculate_similarity_matrix(train_features: torch.Tensor, 
                             test_features: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity matrix between test and training feature vectors."""
    # Validate input tensors
    if train_features.dim() != 2 or test_features.dim() != 2:
        raise ValueError("Input tensors must be 2D")
    
    if train_features.size(1) != test_features.size(1):
        raise ValueError("Feature dimensions must match")
    
    # Normalize features
    train_features_norm = train_features / train_features.norm(dim=1, keepdim=True)
    test_features_norm = test_features / test_features.norm(dim=1, keepdim=True)
    
    # Calculate similarity matrix
    similarity_matrix = torch.mm(test_features_norm, train_features_norm.t())
    
    # Validate output
    if similarity_matrix.size(0) != test_features.size(0) or similarity_matrix.size(1) != train_features.size(0):
        raise ValueError("Invalid similarity matrix dimensions")
    
    return similarity_matrix

def evaluate_matches(similarity_matrix: torch.Tensor, 
                    test_subject_ids: np.ndarray,
                    train_subject_ids: np.ndarray,
                    top_k_values: list = [1, 5, 10]) -> dict:
    """Evaluate match accuracy for different top-k values."""
    results = {}
    
    # Get indices of top-k matches for each test sample
    _, top_k_indices = torch.topk(similarity_matrix, max(top_k_values), dim=1)
    
    for k in top_k_values:
        # Get top-k matches
        top_k_matches = top_k_indices[:, :k]
        
        # Check if correct subject is in top-k matches
        correct_matches = np.array([
            test_subject_ids[i] in train_subject_ids[top_k_matches[i]]
            for i in range(len(test_subject_ids))
        ])
        
        # Calculate accuracy (same as recall for this task)
        accuracy = np.mean(correct_matches)
        
        # For precision, we need to count how many of the top-k matches are correct
        # For each test sample, count how many of its k matches are correct
        true_positives = np.sum([
            np.sum(test_subject_ids[i] == train_subject_ids[top_k_matches[i]])
            for i in range(len(test_subject_ids))
        ])
        
        # Total predictions made = number of test samples * k
        total_predictions = len(test_subject_ids) * k
        
        # Calculate metrics
        precision = true_positives / total_predictions if total_predictions > 0 else 0
        recall = accuracy  # In this case, recall is the same as accuracy
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-subject accuracy
        subject_accuracies = {}
        for subject_id in set(test_subject_ids):
            subject_mask = test_subject_ids == subject_id
            if np.any(subject_mask):
                subject_acc = np.mean(correct_matches[subject_mask])
                subject_accuracies[subject_id] = subject_acc
        
        results[k] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'subject_accuracies': subject_accuracies
        }
    
    return results

def plot_results(results: dict, output_dir: str) -> None:
    """Plot evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy vs top-k
    plt.figure(figsize=(10, 6))
    k_values = list(results.keys())
    accuracies = [results[k]['accuracy'] for k in k_values]
    
    plt.plot(k_values, accuracies, 'bo-')
    plt.xlabel('Top-k')
    plt.ylabel('Accuracy')
    plt.title('Top-k Match Accuracy (Test vs Training)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_topk.png'))
    plt.close()
    
    # Plot precision, recall, and F1 scores
    plt.figure(figsize=(10, 6))
    metrics = ['precision', 'recall', 'f1']
    for metric in metrics:
        values = [results[k][metric] for k in k_values]
        plt.plot(k_values, values, label=metric.capitalize())
    
    plt.xlabel('Top-k')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Scores (Test vs Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_vs_topk.png'))
    plt.close()
    
    # Plot per-subject accuracy
    plt.figure(figsize=(12, 6))
    subjects = sorted(set().union(*[set(results[k]['subject_accuracies'].keys()) for k in k_values]))
    x = np.arange(len(subjects))
    width = 0.25
    
    for i, k in enumerate(k_values):
        accuracies = [results[k]['subject_accuracies'].get(subj, 0) for subj in subjects]
        plt.bar(x + i*width, accuracies, width, label=f'Top-{k}')
    
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('Per-Subject Accuracy for Different Top-k Values')
    plt.xticks(x + width, subjects)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_accuracy.png'))
    plt.close()

def save_results(results: dict, output_dir: str) -> None:
    """Save evaluation results to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Gait Re-Identification Evaluation Results (Test vs Training)\n")
        f.write("========================================================\n\n")
        
        for k, metrics in results.items():
            f.write(f"Top-{k} Results:\n")
            f.write(f"  Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            
            f.write("\n  Per-Subject Accuracy:\n")
            for subject_id, acc in sorted(metrics['subject_accuracies'].items()):
                f.write(f"    Subject {subject_id}: {acc:.4f}\n")
            f.write("\n")
        
        # Add summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total Subjects: {len(set().union(*[set(results[k]['subject_accuracies'].keys()) for k in results.keys()]))}\n")
        f.write(f"Best Top-1 Subject: {max(results[1]['subject_accuracies'].items(), key=lambda x: x[1])[0]} ({max(results[1]['subject_accuracies'].values()):.4f})\n")
        f.write(f"Worst Top-1 Subject: {min(results[1]['subject_accuracies'].items(), key=lambda x: x[1])[0]} ({min(results[1]['subject_accuracies'].values()):.4f})\n")
        f.write(f"Average Top-1 Accuracy: {np.mean(list(results[1]['subject_accuracies'].values())):.4f}\n")
        f.write(f"Standard Deviation: {np.std(list(results[1]['subject_accuracies'].values())):.4f}\n")

def main():
    # Set up paths
    vector_db_dir = "vector_db"
    output_dir = "evaluation_results"
    
    try:
        # Load vector database
        print("Loading vector database...")
        train_features, train_metadata, test_features, test_metadata, scaler = load_vector_database(vector_db_dir)
        
        print(f"Training samples: {len(train_features)}")
        print(f"Test samples: {len(test_features)}")
        
        # Calculate similarity matrix
        print("Calculating similarity matrix...")
        similarity_matrix = calculate_similarity_matrix(train_features, test_features)
        
        # Evaluate matches
        print("Evaluating matches...")
        results = evaluate_matches(similarity_matrix, 
                                 test_metadata['subject_ids'],
                                 train_metadata['subject_ids'])
        
        # Plot and save results
        print("Saving results...")
        plot_results(results, output_dir)
        save_results(results, output_dir)
        
        # Print results
        print("\nEvaluation Results:")
        for k, metrics in results.items():
            print(f"\nTop-{k}:")
            print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print("\n  Per-Subject Accuracy:")
            for subject_id, acc in sorted(metrics['subject_accuracies'].items()):
                print(f"    Subject {subject_id}: {acc:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 