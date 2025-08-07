"""
Source package for the manifolds analysis project.

This package contains modular components for analyzing language model embeddings:
- config: Configuration classes and model setup
- datasets: Dataset classes for temporal and control data
- embeddings: Functions for extracting embeddings from models
- analysis: Core analysis functions (PCA, metrics)
- visualization: Plotting and visualization functions
- utils: Utility functions and helpers
"""

__version__ = "0.1.0"

# Make key components available at package level
from .config import Config, load_model_and_tokenizer
from .datasets import (
    TemporalDataset, 
    TemporalDatasetCreator, 
    ControlDataset,
    get_control_words,
    create_incremental_datasets,
    prepare_control_data
)
from .embeddings import extract_all_layer_representations, extract_control_embeddings
from .analysis import (
    analyze_all_layers, 
    compute_manifold_metrics,
    analyze_control_temporal_across_layers,
    find_best_layer,
    identify_outliers
)

__all__ = [
    "Config",
    "load_model_and_tokenizer",
    "TemporalDataset",
    "TemporalDatasetCreator", 
    "ControlDataset",
    "get_control_words",
    "create_incremental_datasets",
    "prepare_control_data",
    "extract_all_layer_representations",
    "extract_control_embeddings",
    "analyze_all_layers",
    "compute_manifold_metrics",
    "analyze_control_temporal_across_layers",
    "find_best_layer",
    "identify_outliers"
]