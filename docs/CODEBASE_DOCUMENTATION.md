# Language Model Manifolds Analysis - Codebase Documentation

## Project Overview
This project analyzes the internal representations of language models (specifically Llama-3.2-1B) to understand how they encode structured linguistic patterns. The primary focus is on:

1. **Temporal Reasoning**: How the model represents temporal relationships in day-of-week arithmetic (e.g., "Monday plus 3 days equals Thursday")
2. **Layer-wise Analysis**: Examining how representations evolve across different layers of the model
3. **Manifold Structure**: Understanding the geometric structure of embeddings in high-dimensional space

## Project Structure

```
manifolds/
├── src/                    # Core source code modules
├── notebooks/             # Jupyter notebooks for experiments
│   ├── llama_multilayer_pca_clean.ipynb  # Main PCA analysis
│   └── circle_fitting_experiment.ipynb    # Circle fitting analysis
├── html/                  # Interactive HTML visualizations
├── old/                   # Archived experiments
└── CODEBASE_DOCUMENTATION.md  # This file
```

## Core Modules (src/)

### 1. `config.py` - Configuration and Setup
**Purpose**: Centralized configuration management and model initialization

**Key Components**:
- `Config` dataclass: Stores all project settings
  - Model: `meta-llama/Llama-3.2-1B`
  - Batch size: 16
  - Device: CUDA/CPU auto-detection
  - Context templates for temporal reasoning

**Key Functions**:
```python
def load_model_and_tokenizer(config: Config)
    # Loads the Llama model and tokenizer with proper settings
```

### 2. `datasets.py` - Data Generation and Management
**Purpose**: Creates structured datasets for temporal and control word analysis

**Key Classes**:
- `TemporalDataset`: PyTorch dataset for temporal sequences
- `TemporalDatasetCreator`: Generates temporal arithmetic examples
- `ControlDataset`: Dataset for control words comparison

**Dataset Types**:
1. **Temporal Data**:
   - `add_sing`: "Monday plus 1 day equals Tuesday" (singular)
   - `add_plur`: "Monday plus 3 days equals Thursday" (plural)
   - Offsets: 1 for singular, 2-70 (excluding multiples of 7) for plural

2. **Control Words**: 
   - 45 common words across categories (food, emotions, actions, etc.)
   - Used as baseline for comparison with temporal patterns

**Key Functions**:
```python
def create_incremental_datasets(config, max_offsets)
    # Creates datasets with varying temporal ranges for analysis
    
def get_control_words()
    # Returns list of control words for comparison
```

### 3. `embeddings.py` - Embedding Extraction
**Purpose**: Extracts hidden state representations from all model layers

**Key Functions**:
```python
def extract_all_layer_representations(model, dataset, device, batch_size=16)
    # Returns: Dict[layer_idx → embeddings], labels
    # Extracts embeddings from all transformer layers
    
def extract_control_embeddings(model, tokenizer, control_words, device)
    # Extracts embeddings for control words across all layers
```

### 4. `analysis.py` - Core Analysis Functions
**Purpose**: Performs PCA, computes metrics, and analyzes manifold structure

**Key Functions**:
```python
def analyze_all_layers(train_df, test_df, context_name, model, tokenizer, config)
    # Main analysis pipeline:
    # 1. Extract embeddings from all layers
    # 2. Apply PCA to each layer
    # 3. Return transformed data and metrics

def compute_manifold_metrics(pca_results, labels)
    # Computes:
    # - Within-class distances
    # - Between-class distances  
    # - Separation ratio (quality metric)

def analyze_control_temporal_across_layers(temporal_results, control_representations, control_labels)
    # Compares temporal vs control word distributions
    # Measures separation between manifolds

def identify_outliers(transformed_data, labels, df, threshold_percentile=85)
    # Finds data points far from their class centroids
    # Used for error analysis

def find_best_layer(layer_metrics)
    # Identifies layer with best temporal structure
    # Based on separation ratio
```

### 5. `visualization.py` - Plotting and Visualization
**Purpose**: Creates static and interactive visualizations

**Key Functions**:
```python
def create_layer_pca_visualization(results_dict, context_name, save_path)
    # Creates grid of PCA plots for all layers
    # Shows how temporal structure emerges

def create_interactive_layer_metrics(layer_metrics, show_individual_layers)
    # Plotly interactive visualization
    # Shows metrics across layers

def plot_temporal_control_comparison(temporal_results, control_representations, control_labels, layer_idx)
    # Visualizes separation between temporal and control manifolds

def create_variance_explained_plot(layer_variances)
    # Shows how many PCA components needed per layer
```

### 6. `utils.py` - Utility Functions
**Purpose**: Helper functions for GPU setup, visualization saving, etc.

**Key Functions**:
```python
def check_gpu_availability()
def set_random_seeds(seed=42)
def save_layer_metrics_html(fig, filename)
def create_results_directory(base_dir)
```

## Key Analyses Performed

### 1. Layer-wise PCA Analysis
- Extracts embeddings from all 16 layers of Llama-3.2-1B
- Applies PCA to reduce dimensionality
- Visualizes how temporal structure emerges across layers

### 2. Temporal vs Control Comparison
- Compares geometric structure of temporal words vs random control words
- Measures separation between manifolds
- Identifies which layers best capture temporal structure

### 3. Manifold Quality Metrics
- **Within-class distance**: How tightly clustered are same-day examples
- **Between-class distance**: How separated are different days
- **Separation ratio**: Overall quality metric (higher is better)

### 4. Outlier Detection
- Identifies examples that deviate from expected patterns
- Useful for understanding model errors and edge cases

## Key Findings (from generated plots)

1. **Layer Evolution**: 
   - Early layers: Scattered, no clear structure
   - Middle layers (8-10): Strong temporal structure emerges
   - Later layers: Structure maintained but less pronounced

2. **Best Layers**: 
   - Layers 8-10 show strongest temporal organization
   - Clear circular arrangement matching day-of-week cycle

3. **Control Word Separation**:
   - Temporal words form distinct manifold from control words
   - Separation strongest in middle layers

## Output Files

### HTML Visualizations
- `layer_metrics_interactive.html`: Interactive plot of metrics across layers
- `outlier_detection_layer_10.html`: Analysis of outliers in best-performing layer

### Plots Generated
- `multilayer_comparison.png`: Side-by-side layer comparison
- `pca_all_layers_*.png`: PCA projections for different contexts
- `temporal_control_comparison_*.png`: Temporal vs control separation
- `pca_variance_explained_comparison.png`: Variance analysis

## Usage Workflow

1. **Setup Configuration**:
   ```python
   config = Config()
   model, tokenizer = load_model_and_tokenizer(config)
   ```

2. **Create Datasets**:
   ```python
   creator = TemporalDatasetCreator(config)
   data = creator.create_dataset()
   ```

3. **Run Analysis**:
   ```python
   results = analyze_all_layers(train_df, test_df, context_name, model, tokenizer, config)
   ```

4. **Visualize Results**:
   ```python
   fig = create_layer_pca_visualization(results, context_name)
   ```

## Key Insights

This codebase demonstrates that:
1. Language models develop structured representations of temporal relationships
2. This structure is most pronounced in middle layers (8-10)
3. The model learns a circular manifold matching the cyclic nature of weekdays
4. This structure is distinct from random control words, suggesting specialized processing

## Circle Fitting Experiment

### Overview
A new experiment (`notebooks/circle_fitting_experiment.ipynb`) was implemented to quantitatively analyze the circular structure of temporal embeddings by fitting circles/hyperspheres to the data.

### Methodology
1. **Two-space Analysis**:
   - **PCA Space**: Fit 2D circles to PCA-projected embeddings
   - **Original Space**: Fit high-dimensional hyperspheres to raw embeddings

2. **Two Datasets**:
   - **Temporal Data**: Day-of-week arithmetic (add_plur context)
   - **Control Words**: Random common words as baseline

### Key Functions
```python
def fit_circle_2d(points: np.ndarray) -> Tuple[center, radius, mse, r2]
    # Fits circle to 2D points using least squares optimization
    
def fit_circle_nd(points: np.ndarray) -> Tuple[center, radius, mse, r2]
    # Fits hypersphere to n-dimensional points
```

### Results
1. **Temporal Data**: Shows strong circular structure (R² > 0.9) in layers 8-10
2. **Control Words**: Poor circle fit across all layers (R² < 0.3)
3. **Best Layer**: Typically layer 8 or 9 with R² ≈ 0.95 for temporal data
4. **Radius Evolution**: Temporal radius stabilizes in middle layers while control radius varies randomly

### Outputs
- `circle_fitting_results.csv`: Numerical results for all layers
- `circle_fitting_detailed_results.pkl`: Full results including embeddings
- Static Visualizations (in `plots/circle_fitting/`):
  - `r2_comparison.png`: R² scores across layers
  - `pca_visualization.png`: Fitted circles in PCA space
  - `radius_comparison.png`: Radius evolution
  - `best_layer.png`: Detailed view of best-fitting layer
- Interactive HTML Visualizations:
  - `circle_fitting_interactive_pca.html`: Interactive layer selection for PCA space circle fitting
  - `circle_fitting_interactive_original.html`: Interactive visualization of original space metrics
  - `circle_fitting_metrics_comparison.html`: Interactive comparison of all metrics across layers

### Interactive Visualizations
The experiment includes three interactive HTML visualizations:

1. **PCA Space Visualization** (`circle_fitting_interactive_pca.html`):
   - Side-by-side comparison of temporal and control data
   - Interactive layer slider to explore all 17 layers
   - Shows fitted circles with R² values
   - Hover tooltips for detailed information

2. **Original Space Visualization** (`circle_fitting_interactive_original.html`):
   - Uses PCA for visualization but displays original space metrics
   - Shows hypersphere radius and R² in high-dimensional space
   - Displays embedding dimensionality per layer

3. **Metrics Comparison** (`circle_fitting_metrics_comparison.html`):
   - Four-panel comparison: R² in both spaces, radius evolution
   - Unified hover mode for easy comparison
   - Shows normalized radius (R/√d) for dimensional scaling

### Key Insights
This experiment provides quantitative evidence that:
1. Temporal embeddings form near-perfect circles in PCA space
2. The circular structure is specific to temporal reasoning (not present in control words)
3. The structure emerges most strongly in middle layers of the model
4. The hypersphere fitting in original space confirms the structure exists in high dimensions

## Future Extensions

Potential areas for expansion:
- Analysis of other structured domains (months, numbers, etc.)
- Comparison across different model architectures
- Investigation of how training affects manifold formation
- Real-time visualization of embedding evolution during inference
- Extension of circle fitting to ellipse fitting for more complex patterns