"""
Utility module for the manifolds analysis project.

This module contains utility functions and helpers.
"""

import os
import json
import warnings
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import ipywidgets as widgets


def display_original_space_metrics(
    metrics_dict: Dict[int, Dict[str, float]], 
    layer_to_show: Optional[int] = None
):
    """
    Display metrics in a formatted table.
    
    Args:
        metrics_dict: Dictionary of metrics by layer
        layer_to_show: Optional specific layer to display
    """
    if layer_to_show is not None:
        # Display single layer
        if layer_to_show in metrics_dict:
            df = pd.DataFrame([metrics_dict[layer_to_show]], index=[f"Layer {layer_to_show}"])
            display(df.round(4))
        else:
            print(f"Layer {layer_to_show} not found in metrics")
    else:
        # Display all layers
        df = pd.DataFrame(metrics_dict).T
        df.index.name = 'Layer'
        display(df.round(4))


def save_layer_metrics_html(
    fig, 
    filename: str = "layer_metrics_interactive.html",
    include_plotlyjs: str = 'cdn'
):
    """
    Save a plotly figure to HTML file.
    
    Args:
        fig: Plotly figure object
        filename: Output filename
        include_plotlyjs: How to include plotly.js ('cdn', 'inline', etc.)
    """
    fig.write_html(
        filename,
        include_plotlyjs=include_plotlyjs,
        config={'displayModeBar': True, 'displaylogo': False}
    )
    print(f"Saved interactive plot to {filename}")


def create_layer_selector_widget(max_layers: int) -> widgets.IntSlider:
    """
    Create a layer selector widget.
    
    Args:
        max_layers: Maximum number of layers
        
    Returns:
        ipywidgets.IntSlider: Layer selector widget
    """
    return widgets.IntSlider(
        value=8,
        min=0,
        max=max_layers,
        step=1,
        description='Layer:',
        continuous_update=False
    )


def format_results_summary(results: Dict[str, Any]) -> str:
    """
    Format analysis results into a readable summary.
    
    Args:
        results: Dictionary of results
        
    Returns:
        str: Formatted summary
    """
    summary_lines = []
    
    for key, value in results.items():
        if isinstance(value, dict):
            summary_lines.append(f"\n{key}:")
            for sub_key, sub_value in value.items():
                summary_lines.append(f"  {sub_key}: {sub_value}")
        else:
            summary_lines.append(f"{key}: {value}")
    
    return "\n".join(summary_lines)


def save_results_to_json(
    results: Dict[str, Any], 
    filename: str = "analysis_results.json"
):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Results to save
        filename: Output filename
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    converted_results = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_results_from_json(filename: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        dict: Loaded results
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def check_gpu_availability():
    """Check and display GPU availability information."""
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seeds set to {seed}")


def suppress_warnings():
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Suppress specific transformers warnings
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    
    print("Warnings suppressed")


def create_results_directory(base_dir: str = "results") -> str:
    """
    Create a timestamped results directory.
    
    Args:
        base_dir: Base directory name
        
    Returns:
        str: Path to created directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_dir}_{timestamp}"
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    
    return results_dir