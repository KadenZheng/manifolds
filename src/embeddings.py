"""
Embeddings module for the manifolds analysis project.

This module contains functions for extracting embeddings from language models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .datasets import TemporalDataset, ControlDataset
from collections import defaultdict


def extract_all_layer_representations(
    model: PreTrainedModel, 
    dataset: Dataset, 
    device: str, 
    batch_size: int = 16
) -> Tuple[Dict[int, np.ndarray], List]:
    """
    Extract representations from all layers of the model.
    
    Args:
        model: The language model to extract from
        dataset: Dataset to process
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for processing
        
    Returns:
        tuple: (layer_representations, labels)
            - layer_representations: Dict mapping layer index to embeddings array
            - labels: List of labels/metadata for each sample
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_layer_representations = defaultdict(list)
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representations"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model outputs with hidden states
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Extract hidden states from all layers
            hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer
            
            # Process each layer
            for layer_idx, layer_hidden_state in enumerate(hidden_states):
                batch_representations = layer_hidden_state[:, -1, :].cpu().numpy()  # Last token
                all_layer_representations[layer_idx].append(batch_representations)
            
            # Collect labels if they exist in the batch
            if 'day' in batch:
                all_labels.extend(batch['day'].cpu().numpy())
            elif 'label' in batch:
                all_labels.extend(batch['label'])
    
    # Stack all representations for each layer
    layer_representations = {}
    for layer_idx, reprs_list in all_layer_representations.items():
        layer_representations[layer_idx] = np.vstack(reprs_list)
    
    return layer_representations, all_labels


def extract_control_embeddings(
    control_words: List[str], 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    device: str
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Extract embeddings for control words.
    
    Args:
        control_words: List of control words
        model: The language model
        tokenizer: Tokenizer for the model
        device: Device to run on
        
    Returns:
        tuple: (layer_representations, labels)
            - layer_representations: Dict mapping layer index to embeddings
            - labels: List of labels for each embedding
    """
    # Tokenize control words
    encodings = tokenizer(
        control_words,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=128
    )
    
    # Create dataset
    control_dataset = ControlDataset(encodings, control_words)
    
    # Extract representations from all layers
    layer_representations, labels = extract_all_layer_representations(
        model, control_dataset, device, batch_size=32
    )
    
    return layer_representations, labels


def extract_single_layer_embeddings(
    model: PreTrainedModel,
    dataset: Dataset,
    layer_idx: int,
    device: str,
    batch_size: int = 16
) -> np.ndarray:
    """
    Extract embeddings from a single layer.
    
    Args:
        model: The language model
        dataset: Dataset to process
        layer_idx: Which layer to extract from
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        np.ndarray: Embeddings from the specified layer
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting layer {layer_idx}"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get representations from specified layer
            layer_hidden = hidden_states[layer_idx]
            last_token_reps = layer_hidden[:, -1, :].cpu().numpy()
            embeddings.append(last_token_reps)
    
    return np.vstack(embeddings)


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a set of embeddings.
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        dict: Statistics including mean norm, std, etc.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'min_norm': np.min(norms),
        'max_norm': np.max(norms),
        'mean_std_per_dim': np.mean(np.std(embeddings, axis=0)),
        'n_samples': len(embeddings),
        'n_dims': embeddings.shape[1]
    }