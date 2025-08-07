"""
Analysis module for the manifolds analysis project.

This module contains core analysis functions including PCA and metrics computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import defaultdict

from .config import Config
from .embeddings import extract_all_layer_representations
from .datasets import TemporalDataset


def analyze_all_layers(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    context_name: str,
    model=None,
    tokenizer=None,
    config: Optional[Config] = None
) -> Dict[str, Dict]:
    """
    Perform PCA analysis across all layers.
    
    Args:
        train_df: Training dataframe with temporal data
        test_df: Test dataframe with temporal data
        context_name: Name of the context being analyzed
        model: Language model (required)
        tokenizer: Tokenizer (required)
        config: Configuration object (optional)
        
    Returns:
        dict: Dictionary with PCA results for each layer
    """
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer are required")
    
    if config is None:
        config = Config()
    
    # Combine train and test data
    all_texts = train_df['text'].tolist() + test_df['text'].tolist()
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Tokenize all texts
    encodings = tokenizer(all_texts, truncation=True, padding=True, return_tensors='pt', max_length=128)
    
    # Create dataset
    dataset = TemporalDataset(encodings, all_df)
    
    # Extract representations from all layers
    layer_representations, labels = extract_all_layer_representations(model, dataset, config.device)
    
    # Perform PCA on each layer
    layer_pca_results = {}
    num_layers = len(layer_representations)
    
    for layer_idx in range(num_layers):
        reprs = layer_representations[layer_idx]
        
        # Apply PCA
        # Ensure n_components doesn't exceed the minimum of samples or features
        n_components = min(10, reprs.shape[0] - 1, reprs.shape[1])
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(reprs)
        
        layer_pca_results[layer_idx] = {
            'pca_transformed': pca_transformed,
            'pca': pca,
            'explained_variance': pca.explained_variance_ratio_,
            'n_components': pca.n_components_
        }
    
    # Split back into train/test
    train_size = len(train_df)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(all_df)))
    
    # Return results
    results = {
        **layer_pca_results,
        'num_samples': len(all_df),
        'train_indices': train_indices,
        'test_indices': test_indices,
        'layer_representations': layer_representations,
        'train_df': train_df,
        'test_df': test_df,
        'all_df': all_df,
        'labels': labels
    }
    
    return results


def compute_manifold_metrics(
    pca_results: np.ndarray, 
    labels: List[int]
) -> Dict[str, float]:
    """
    Compute metrics for manifold quality.
    
    Args:
        pca_results: PCA-transformed data
        labels: Labels for each data point
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Compute centroid for each class
    unique_labels = np.unique(labels)
    centroids = {}
    
    for label in unique_labels:
        mask = np.array(labels) == label
        centroids[label] = np.mean(pca_results[mask], axis=0)
    
    # Compute within-class and between-class distances
    within_class_dist = []
    between_class_dist = []
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        class_points = pca_results[mask]
        
        # Within-class distances
        if len(class_points) > 1:
            dists = cdist(class_points, [centroids[label]])
            within_class_dist.extend(dists.flatten())
        
        # Between-class distances
        for j, other_label in enumerate(unique_labels):
            if i < j:
                dist = np.linalg.norm(centroids[label] - centroids[other_label])
                between_class_dist.append(dist)
    
    metrics['mean_within_class_dist'] = np.mean(within_class_dist) if within_class_dist else 0
    metrics['mean_between_class_dist'] = np.mean(between_class_dist) if between_class_dist else 0
    metrics['separation_ratio'] = (
        metrics['mean_between_class_dist'] / metrics['mean_within_class_dist'] 
        if metrics['mean_within_class_dist'] > 0 else 0
    )
    
    return metrics


def compute_original_space_metrics(
    layer_representations: Dict[int, np.ndarray],
    labels: List[int],
    temporal_words: List[str],
    control_words: Dict[str, List[str]]
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics in the original embedding space (before PCA).
    
    Args:
        layer_representations: Embeddings for each layer
        labels: Labels for temporal data
        temporal_words: List of temporal words
        control_words: Dictionary of control words
        
    Returns:
        dict: Metrics for each layer
    """
    metrics_by_layer = {}
    
    for layer_idx, embeddings in layer_representations.items():
        metrics = compute_manifold_metrics(embeddings, labels)
        metrics_by_layer[layer_idx] = metrics
    
    return metrics_by_layer


def identify_outliers(
    transformed_data: np.ndarray,
    labels: List[int],
    df: pd.DataFrame,
    threshold_percentile: int = 85
) -> pd.DataFrame:
    """
    Identify outliers in the transformed data.
    
    Args:
        transformed_data: PCA-transformed data
        labels: Labels for each point
        df: Original dataframe with metadata
        threshold_percentile: Percentile threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataframe with outlier information
    """
    # Compute distances from class centroids
    unique_labels = np.unique(labels)
    distances = np.zeros(len(labels))
    
    for label in unique_labels:
        mask = np.array(labels) == label
        class_points = transformed_data[mask]
        centroid = np.mean(class_points, axis=0)
        
        # Compute distances for this class
        class_distances = np.linalg.norm(class_points - centroid, axis=1)
        distances[mask] = class_distances
    
    # Identify outliers
    threshold = np.percentile(distances, threshold_percentile)
    outlier_mask = distances > threshold
    
    # Create outlier dataframe
    outlier_df = df[outlier_mask].copy()
    outlier_df['distance_from_centroid'] = distances[outlier_mask]
    outlier_df['pca_x'] = transformed_data[outlier_mask, 0]
    outlier_df['pca_y'] = transformed_data[outlier_mask, 1]
    
    return outlier_df.sort_values('distance_from_centroid', ascending=False)


def find_best_layer(layer_metrics: Dict[int, Dict[str, float]]) -> int:
    """
    Find the best layer based on separation metrics.
    
    Args:
        layer_metrics: Dictionary of metrics for each layer
        
    Returns:
        int: Index of the best layer
    """
    best_layer = max(
        layer_metrics.keys(),
        key=lambda k: layer_metrics[k].get('separation_ratio', 0)
    )
    return best_layer


def analyze_control_temporal_across_layers(
    temporal_results: Dict,
    control_representations: Dict[int, np.ndarray],
    control_labels: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    Analyze relationship between temporal and control words across layers.
    
    Args:
        temporal_results: Results from temporal analysis
        control_representations: Control word embeddings by layer
        control_labels: Labels for control words
        
    Returns:
        dict: Metrics for each layer
    """
    layer_metrics = {}
    
    # Get the number of layers (excluding metadata keys)
    layer_indices = [k for k in temporal_results.keys() if isinstance(k, int)]
    
    for layer_idx in layer_indices:
        # Get temporal PCA results
        temporal_pca = temporal_results[layer_idx]['pca_transformed']
        temporal_labels = temporal_results['labels']
        
        # Get control embeddings for this layer
        control_embeddings = control_representations[layer_idx]
        
        # Apply same PCA transformation to control words
        pca = temporal_results[layer_idx]['pca']
        control_pca = pca.transform(control_embeddings)
        
        # Compute temporal centroid (average across all temporal points)
        temporal_centroid = np.mean(temporal_pca[:, :2], axis=0)
        
        # Compute control centroid
        control_centroid = np.mean(control_pca[:, :2], axis=0)
        
        # Compute spreads (using first 2 PCs for visualization)
        temporal_distances = np.linalg.norm(temporal_pca[:, :2] - temporal_centroid, axis=1)
        control_distances = np.linalg.norm(control_pca[:, :2] - control_centroid, axis=1)
        
        temporal_radius_mean = np.mean(temporal_distances)
        temporal_radius_std = np.std(temporal_distances)
        control_radius_mean = np.mean(control_distances)
        control_radius_std = np.std(control_distances)
        
        # Compute separation
        centroid_distance = np.linalg.norm(temporal_centroid - control_centroid)
        
        # Separation ratio (higher is better)
        if temporal_radius_mean + control_radius_mean > 0:
            separation_ratio = centroid_distance / (temporal_radius_mean + control_radius_mean)
        else:
            separation_ratio = 0
        
        layer_metrics[layer_idx] = {
            'temporal_centroid': temporal_centroid,
            'control_centroid': control_centroid,
            'temporal_radius_mean': temporal_radius_mean,
            'temporal_radius_std': temporal_radius_std,
            'control_radius_mean': control_radius_mean,
            'control_radius_std': control_radius_std,
            'centroid_distance': centroid_distance,
            'separation_ratio': separation_ratio,
            'n_temporal_points': len(temporal_pca),
            'n_control_points': len(control_pca)
        }
    
    return layer_metrics


def compute_pca_variance_explained(
    embeddings: np.ndarray,
    n_components: int = 10
) -> np.ndarray:
    """
    Compute variance explained by PCA components.
    
    Args:
        embeddings: Input embeddings
        n_components: Number of components to compute
        
    Returns:
        np.ndarray: Variance explained by each component
    """
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    return pca.explained_variance_ratio_


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        np.ndarray: Distance matrix
    """
    return cdist(embeddings, embeddings, metric='euclidean')