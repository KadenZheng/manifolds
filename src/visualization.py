"""
Visualization module for the manifolds analysis project.

This module contains all plotting and visualization functions,
both static (matplotlib) and interactive (plotly).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .analysis import compute_manifold_metrics


# Color utilities
def clear_color_map(n: int) -> List[Tuple[float, float, float]]:
    """
    Generate clear, distinct colors for n categories.
    
    Args:
        n: Number of colors needed
        
    Returns:
        list: List of RGB tuples
    """
    if n <= 10:
        # Use tab10 colormap for small n
        colors = plt.cm.tab10(np.linspace(0, 1, n))
    else:
        # Use rainbow for larger n
        colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    return [tuple(c[:3]) for c in colors]


# Static visualizations
def create_layer_pca_visualization(
    results_dict: Dict, 
    context_name: str, 
    save_path: Optional[str] = None,
    color_by_result: bool = True
) -> plt.Figure:
    """
    Create visualization of PCA projections across all layers.
    
    Args:
        results_dict: Dictionary with PCA results for each layer
        context_name: Name of the context
        save_path: Optional path to save the figure
        color_by_result: If True, color by result day; if False, color by starting day
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Get layer indices (filter out metadata keys)
    layer_indices = sorted([k for k in results_dict.keys() if isinstance(k, int)])
    n_layers = len(layer_indices)
    
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else axes
    if n_layers == 1:
        axes = [axes]
    
    # Use consistent colors for days of week
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Get labels to use for coloring
    if color_by_result and 'all_df' in results_dict and 'result_day' in results_dict['all_df'].columns:
        # Use result day for coloring
        color_labels = results_dict['all_df']['result_day'].values
        color_label_name = "Result Day"
    else:
        # Use starting day for coloring (default behavior)
        color_labels = results_dict['labels']
        color_label_name = "Starting Day"
    
    for idx, layer_idx in enumerate(layer_indices):
        ax = axes[idx]
        
        # Get PCA transformed data for this layer
        transformed = results_dict[layer_idx]['pca_transformed']
        
        # Get variance explained for this layer
        explained_variance = results_dict[layer_idx]['explained_variance']
        
        # Plot each day with its color
        for day in range(7):
            mask = color_labels == day
            if np.any(mask):
                ax.scatter(transformed[mask, 0], transformed[mask, 1], 
                          c=[colors[day]], label=days_of_week[day], 
                          alpha=0.6, s=50)
        
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
        ax.grid(True, alpha=0.3)
        
        # Add legend to first subplot only
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_label_name)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{context_name}: PCA Projections Across All Layers (Colored by {color_label_name})', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_temporal_control_comparison(
    temporal_results: Dict,
    control_representations: Dict[int, np.ndarray],
    control_labels: List[str],
    layer_idx: int = 8
) -> plt.Figure:
    """
    Plot comparison between temporal and control words.
    
    Args:
        temporal_results: Results from temporal analysis
        control_representations: Control word embeddings
        control_labels: Labels for control words
        layer_idx: Which layer to visualize
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get temporal data
    temporal_pca = temporal_results[layer_idx]['pca_transformed']
    temporal_labels = temporal_results['labels']
    
    # Transform control embeddings using same PCA
    pca = temporal_results[layer_idx]['pca']
    control_pca = pca.transform(control_representations[layer_idx])
    
    # Plot control words
    ax.scatter(control_pca[:, 0], control_pca[:, 1], 
              c='gray', alpha=0.3, s=30, label='Control words')
    
    # Plot temporal words by day
    day_colors = plt.cm.rainbow(np.linspace(0, 1, 7))
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Check if we have result_day information
    if 'all_df' in temporal_results and 'result_day' in temporal_results['all_df'].columns:
        # Use result day for coloring
        result_days = temporal_results['all_df']['result_day'].values
        for day in range(7):
            mask = result_days == day
            day_points = temporal_pca[mask, :2]
            ax.scatter(day_points[:, 0], day_points[:, 1],
                      c=[day_colors[day]], label=days_of_week[day],
                      s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
    else:
        # Fallback to starting day
        for day in range(7):
            mask = [label == day for label in temporal_labels]
            day_points = temporal_pca[mask, :2]
            ax.scatter(day_points[:, 0], day_points[:, 1],
                      c=[day_colors[day]], label=days_of_week[day],
                      s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Compute and plot centroids
    temporal_centroid = np.mean(temporal_pca[:, :2], axis=0)
    control_centroid = np.mean(control_pca[:, :2], axis=0)
    
    ax.scatter(*temporal_centroid, marker='*', s=1000, c='blue', edgecolor='black', label='Temporal centroid')
    ax.scatter(*control_centroid, marker='*', s=1000, c='red', edgecolor='black', label='Control centroid')
    
    # Add circles to show spread
    temporal_radii = np.linalg.norm(temporal_pca[:, :2] - temporal_centroid, axis=1)
    control_radii = np.linalg.norm(control_pca[:, :2] - control_centroid, axis=1)
    
    circle1 = plt.Circle(temporal_centroid, temporal_radii.mean(), 
                        fill=False, color='blue', linewidth=2, linestyle='--', alpha=0.5)
    circle2 = plt.Circle(control_centroid, control_radii.mean(), 
                        fill=False, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Temporal vs Control Words - Layer {layer_idx}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_multilayer_comparison(layer_metrics: Dict[int, Dict[str, float]]) -> plt.Figure:
    """
    Create multi-panel comparison of metrics across layers.
    
    Args:
        layer_metrics: Dictionary of metrics for each layer
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    layers = sorted(layer_metrics.keys())
    
    # Extract metrics
    temporal_radii = [layer_metrics[l]['temporal_radius_mean'] for l in layers]
    control_radii = [layer_metrics[l]['control_radius_mean'] for l in layers]
    centroid_distances = [layer_metrics[l]['centroid_distance'] for l in layers]
    separation_ratios = [layer_metrics[l]['separation_ratio'] for l in layers]
    
    # Plot 1: Cluster spreads
    ax = axes[0, 0]
    ax.plot(layers, temporal_radii, 'b-o', label='Temporal spread', linewidth=2)
    ax.plot(layers, control_radii, 'r-o', label='Control spread', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean radius from centroid')
    ax.set_title('Cluster Spreads Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Centroid separation
    ax = axes[0, 1]
    ax.plot(layers, centroid_distances, 'g-o', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Distance between centroids')
    ax.set_title('Temporal-Control Separation')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Separation ratio
    ax = axes[1, 0]
    ax.plot(layers, separation_ratios, 'm-o', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Separation ratio')
    ax.set_title('Normalized Separation (higher = better)')
    ax.grid(True, alpha=0.3)
    
    # Find best layer
    best_layer = max(layer_metrics.keys(), key=lambda k: layer_metrics[k]['separation_ratio'])
    ax.axvline(x=best_layer, color='red', linestyle='--', alpha=0.5)
    ax.text(best_layer, ax.get_ylim()[1]*0.9, f'Best: Layer {best_layer}', 
            ha='center', va='bottom', color='red')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.text(0.1, 0.8, f'Best layer: {best_layer}', transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.6, f'Max separation ratio: {layer_metrics[best_layer]["separation_ratio"]:.3f}', 
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.4, f'At layer {best_layer}:', transform=ax.transAxes, fontsize=12)
    ax.text(0.15, 0.3, f'- Temporal spread: {layer_metrics[best_layer]["temporal_radius_mean"]:.3f}', 
            transform=ax.transAxes, fontsize=10)
    ax.text(0.15, 0.2, f'- Control spread: {layer_metrics[best_layer]["control_radius_mean"]:.3f}', 
            transform=ax.transAxes, fontsize=10)
    ax.text(0.15, 0.1, f'- Centroid distance: {layer_metrics[best_layer]["centroid_distance"]:.3f}', 
            transform=ax.transAxes, fontsize=10)
    ax.axis('off')
    
    plt.suptitle('Multi-layer Analysis: Temporal vs Control Words', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_distance_relationships(
    temporal_results: Dict,
    control_representations: Dict[int, np.ndarray],
    control_labels: List[str],
    layer_idx: int = 8
) -> plt.Figure:
    """
    Visualize distance relationships in the manifold.
    
    Args:
        temporal_results: Results from temporal analysis
        control_representations: Control word embeddings
        control_labels: Labels for control words
        layer_idx: Which layer to visualize
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Implementation will be extracted from notebook
    
    plt.suptitle(f'Distance Relationship Analysis - Layer {layer_idx}', fontsize=14)
    plt.tight_layout()
    return fig


def plot_original_space_metrics_evolution(
    metrics_dict: Dict[int, Dict[str, float]]
) -> plt.Figure:
    """
    Plot evolution of metrics across layers in original space.
    
    Args:
        metrics_dict: Metrics for each layer
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Implementation will be extracted from notebook
    
    plt.suptitle('Evolution of Manifold Metrics Across Layers (Original Space)', fontsize=16)
    plt.tight_layout()
    return fig


# Interactive visualizations
def create_interactive_pca_with_hover(
    results_dict: Dict,
    context_name: str,
    text_data: Optional[pd.DataFrame] = None,
    color_by_result: bool = True
) -> go.Figure:
    """
    Create interactive PCA visualization with layer dropdown and hover text.
    
    Args:
        results_dict: Results from analyze_all_layers containing PCA data
        context_name: Name of the context (e.g., 'add_plur')
        text_data: DataFrame containing original text data (optional)
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure with dropdown
    """
    # Get available layers
    layer_indices = sorted([k for k in results_dict.keys() if isinstance(k, int)])
    
    # Days of the week for coloring
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create figure
    fig = go.Figure()
    
    # Get texts if available (should match the order of all_texts used in analysis)
    all_texts = None
    if text_data is not None and 'text' in text_data.columns:
        all_texts = text_data['text'].tolist()
    
    # Get labels to use for coloring
    if color_by_result and 'all_df' in results_dict and 'result_day' in results_dict['all_df'].columns:
        # Use result day for coloring
        color_labels = results_dict['all_df']['result_day'].values
        starting_labels = results_dict['labels']  # Keep starting labels for hover text
    else:
        # Use starting day for coloring (default behavior)
        color_labels = results_dict['labels']
        starting_labels = results_dict['labels']
    
    # Add traces for each layer (initially all hidden except first)
    for i, layer_idx in enumerate(layer_indices):
        pca_data = results_dict[layer_idx]['pca_transformed']
        
        # Create hover text
        hover_texts = []
        if all_texts and len(all_texts) == len(starting_labels):
            # Texts match labels length
            for j, (start_label, text) in enumerate(zip(starting_labels, all_texts)):
                # Convert numeric labels to day names
                if isinstance(start_label, (int, np.integer)):
                    start_day_name = DAYS[start_label] if 0 <= start_label < 7 else f"Day {start_label}"
                else:
                    start_day_name = start_label
                    
                # Get result day if available
                if color_by_result and 'all_df' in results_dict and 'result_day' in results_dict['all_df'].columns:
                    result_day_idx = results_dict['all_df']['result_day'].iloc[j]
                    result_day_name = DAYS[result_day_idx] if 0 <= result_day_idx < 7 else f"Day {result_day_idx}"
                    hover_texts.append(f"Text: {text}<br>Start: {start_day_name}<br>Result: {result_day_name}")
                else:
                    hover_texts.append(f"Day: {start_day_name}<br>Text: {text}")
        else:
            # Just show the day labels
            hover_texts = []
            for j, start_label in enumerate(starting_labels):
                if isinstance(start_label, (int, np.integer)):
                    start_day_name = DAYS[start_label] if 0 <= start_label < 7 else f"Day {start_label}"
                else:
                    start_day_name = start_label
                    
                # Get result day if available
                if color_by_result and 'all_df' in results_dict and 'result_day' in results_dict['all_df'].columns:
                    result_day_idx = results_dict['all_df']['result_day'].iloc[j]
                    result_day_name = DAYS[result_day_idx] if 0 <= result_day_idx < 7 else f"Day {result_day_idx}"
                    hover_texts.append(f"Start: {start_day_name}<br>Result: {result_day_name}")
                else:
                    hover_texts.append(f"Day: {start_day_name}")
        
        # Add a scatter trace for this layer
        for day_idx, day in enumerate(DAYS):
            # Use color_labels for masking
            mask = np.array(color_labels == day_idx)
            
            day_pca = pca_data[mask]
            day_hover = [h for h, m in zip(hover_texts, mask) if m]
            
            # Debug: check if we have data for this day (uncomment if needed)
            # if i == 0 and day_pca.shape[0] > 0:
            #     print(f"Debug - {day}: {day_pca.shape[0]} points")
            
            fig.add_trace(go.Scatter(
                x=day_pca[:, 0] if day_pca.shape[0] > 0 else [],
                y=day_pca[:, 1] if day_pca.shape[0] > 0 else [],
                mode='markers',
                name=day,
                text=day_hover,
                hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                marker=dict(
                    size=10,
                    color=px.colors.qualitative.Set3[day_idx % len(px.colors.qualitative.Set3)],
                    line=dict(width=1, color='white')
                ),
                visible=(i == 0),  # Only first layer visible initially
                legendgroup=day,
                showlegend=(i == 0)  # Only show legend for first layer
            ))
    
    # Create dropdown menu
    dropdown_buttons = []
    for i, layer_idx in enumerate(layer_indices):
        # Get variance explained for this layer
        explained_variance = results_dict[layer_idx]['explained_variance']
        
        # Determine which traces to show for this layer
        visibility = []
        for j in range(len(fig.data)):
            # Each layer has 7 traces (one per day)
            layer_group = j // 7
            visibility.append(layer_group == i)
        
        dropdown_buttons.append(
            dict(
                label=f"Layer {layer_idx}",
                method="update",
                args=[{"visible": visibility},
                      {"title": f"{context_name} - Layer {layer_idx} PCA Projection",
                       "xaxis.title": f"PC1 ({explained_variance[0]:.1%} variance)",
                       "yaxis.title": f"PC2 ({explained_variance[1]:.1%} variance)"}]
            )
        )
    
    # Update layout
    # Get variance explained for the first layer (shown by default)
    first_layer_variance = results_dict[layer_indices[0]]['explained_variance']
    
    fig.update_layout(
        title=f"{context_name} - Layer {layer_indices[0]} PCA Projection",
        xaxis_title=f"PC1 ({first_layer_variance[0]:.1%} variance)",
        yaxis_title=f"PC2 ({first_layer_variance[1]:.1%} variance)",
        hovermode='closest',
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.15,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        height=700,
        width=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Equal aspect ratio
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig


def create_interactive_temporal_control_visualization(
    temporal_results: Dict,
    control_representations: Dict[int, np.ndarray],
    control_labels: List[str]
) -> go.Figure:
    """
    Create interactive visualization with layer slider.
    
    Args:
        temporal_results: Results from temporal analysis
        control_representations: Control word embeddings
        control_labels: Labels for control words
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure
    """
    fig = go.Figure()
    
    # Implementation will be extracted from notebook
    
    return fig


def create_interactive_layer_metrics(
    layer_metrics: Dict[int, Dict[str, float]],
    show_individual_layers: bool = False
) -> go.Figure:
    """
    Create interactive visualization of layer metrics.
    
    Args:
        layer_metrics: Metrics for each layer
        show_individual_layers: If True, create a view for individual layer details
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure
    """
    layers = sorted(layer_metrics.keys())
    
    if show_individual_layers:
        # Create a detailed view for each layer with dropdown
        fig = go.Figure()
        
        # Create bar charts for each layer
        for i, layer_idx in enumerate(layers):
            metrics = layer_metrics[layer_idx]
            
            # Prepare data for bar chart
            metric_names = ['Temporal Radius', 'Control Radius', 'Centroid Distance', 'Separation Ratio']
            metric_values = [
                metrics['temporal_radius_mean'],
                metrics['control_radius_mean'],
                metrics['centroid_distance'],
                metrics['separation_ratio']
            ]
            
            # Add bar trace
            fig.add_trace(go.Bar(
                x=metric_names,
                y=metric_values,
                name=f'Layer {layer_idx}',
                text=[f'{v:.3f}' for v in metric_values],
                textposition='auto',
                visible=(i == 0),  # Only first layer visible initially
                marker_color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
            ))
        
        # Create dropdown
        dropdown_buttons = []
        for i, layer_idx in enumerate(layers):
            visibility = [j == i for j in range(len(layers))]
            dropdown_buttons.append(
                dict(
                    label=f"Layer {layer_idx}",
                    method="update",
                    args=[{"visible": visibility},
                          {"title": f"Layer {layer_idx} Metrics"}]
                )
            )
        
        fig.update_layout(
            title=f"Layer {layers[0]} Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.15,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            height=600,
            width=800,
            showlegend=False
        )
        
    else:
        # Create the overview line plot (cleaner version)
        fig = go.Figure()
        
        # Define metrics and their properties
        metrics_config = [
            {
                'key': 'temporal_radius_mean',
                'name': 'Temporal Radius',
                'color': '#3498db',
                'yaxis': 'y',
                'description': 'Average distance of temporal points from centroid'
            },
            {
                'key': 'control_radius_mean', 
                'name': 'Control Radius',
                'color': '#e74c3c',
                'yaxis': 'y',
                'description': 'Average distance of control points from centroid'
            },
            {
                'key': 'centroid_distance',
                'name': 'Centroid Distance',
                'color': '#2ecc71',
                'yaxis': 'y2',
                'description': 'Distance between temporal and control centroids'
            },
            {
                'key': 'separation_ratio',
                'name': 'Separation Ratio',
                'color': '#9b59b6',
                'yaxis': 'y3',
                'description': 'Ratio of centroid distance to combined radii'
            }
        ]
        
        # Add traces
        for metric in metrics_config:
            fig.add_trace(go.Scatter(
                x=layers,
                y=[layer_metrics[l][metric['key']] for l in layers],
                mode='lines+markers',
                name=metric['name'],
                line=dict(color=metric['color'], width=3),
                marker=dict(size=8),
                yaxis=metric['yaxis'],
                hovertemplate=(
                    f"<b>{metric['name']}</b><br>" +
                    f"Layer: %{{x}}<br>" +
                    f"Value: %{{y:.3f}}<br>" +
                    f"<i>{metric['description']}</i>" +
                    "<extra></extra>"
                )
            ))
        
        # Find best layer
        best_layer = max(layer_metrics.keys(), key=lambda k: layer_metrics[k]['separation_ratio'])
        
        # Add vertical line for best layer with annotation
        fig.add_vline(
            x=best_layer, 
            line_dash="dash", 
            line_color="gray",
            line_width=2
        )
        
        fig.add_annotation(
            x=best_layer,
            y=1,
            yref="paper",
            text=f"Best Layer: {best_layer}<br>Sep. Ratio: {layer_metrics[best_layer]['separation_ratio']:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            ax=40,
            ay=-40,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Update layout with cleaner styling
        fig.update_layout(
            title=dict(
                text="Layer Metrics Analysis",
                font=dict(size=20),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Layer Index",
                dtick=1,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                title="Radius",
                side="left",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis2=dict(
                title="Centroid Distance",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False
            ),
            yaxis3=dict(
                title="Separation Ratio",
                overlaying="y",
                side="right",
                position=0.92,
                showgrid=False,
                zeroline=False,
                tickformat='.2f'
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(128,128,128,0.3)',
                borderwidth=1
            ),
            height=700,
            width=1000,
            plot_bgcolor='rgba(250,250,250,0.8)',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        # Add subtle background shading for better readability
        fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=False)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=False)
    
    return fig


def save_interactive_visualizations(
    temporal_results: Dict,
    control_representations: Dict[int, np.ndarray],
    control_labels: List[str],
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Save all interactive visualizations to HTML files.
    
    Args:
        temporal_results: Results from temporal analysis
        control_representations: Control word embeddings
        control_labels: Labels for control words
        output_dir: Directory to save files
        
    Returns:
        dict: Mapping of visualization names to file paths
    """
    saved_files = {}
    
    # Implementation will be extracted from notebook
    
    return saved_files


# Specialized plot functions
def plot_metrics_for_layer(
    layer_idx: int,
    metrics_data: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create detailed metrics plot for a specific layer.
    
    Args:
        layer_idx: Which layer to plot
        metrics_data: Metrics data for the layer
        save_path: Optional path to save figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Implementation will be extracted from notebook
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_outlier_visualization(
    outlier_df: pd.DataFrame,
    layer_idx: int,
    interactive: bool = True
) -> Any:
    """
    Create visualization of outliers.
    
    Args:
        outlier_df: DataFrame with outlier information
        layer_idx: Which layer the outliers are from
        interactive: Whether to create interactive (plotly) or static (matplotlib) plot
        
    Returns:
        Figure object (plotly or matplotlib)
    """
    if interactive:
        fig = go.Figure()
        # Interactive implementation
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Static implementation
    
    return fig