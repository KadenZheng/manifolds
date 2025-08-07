# Language Model Manifolds Analysis

An in-depth analysis of how large language models (specifically Llama-3.2-1B) encode structured linguistic patterns in their internal representations, with a focus on temporal reasoning and geometric manifold structures.

## 🎯 Project Overview

This project investigates how language models internally represent structured knowledge, particularly focusing on:

- **Temporal Reasoning**: How models encode day-of-week arithmetic (e.g., "Monday plus 3 days equals Thursday")
- **Geometric Structure**: Discovery that temporal embeddings form circular/cyclic manifolds matching the cyclic nature of weekdays
- **Layer-wise Evolution**: How representations evolve and specialize across transformer layers
- **Comparative Analysis**: Contrasting structured temporal patterns with unstructured control words

## 🔬 Key Findings

### 1. Circular Manifold Discovery
- Temporal embeddings naturally organize into near-perfect circles in PCA space (R² > 0.95)
- The circular structure emerges most strongly in middle layers (layers 8-10)
- Control words show no such geometric organization (R² < 0.3)

### 2. Dimensionality Reduction
- Temporal data requires 11.3% fewer PCA components than control words for 95% variance
- Suggests more efficient, structured encoding of temporal relationships
- Best compression occurs in early layers

### 3. Layer Specialization
- Early layers: Basic feature extraction
- Middle layers: Strongest circular structure emerges
- Later layers: Task-specific refinement

## 📁 Project Structure

```
manifolds/
├── src/                          # Core analysis modules
│   ├── config.py                 # Configuration and model setup
│   ├── datasets.py               # Data generation (temporal & control)
│   ├── embeddings.py             # Embedding extraction
│   ├── analysis.py               # PCA and metrics computation
│   ├── visualization.py          # Plotting utilities
│   └── utils.py                  # Helper functions
│
├── notebooks/                    # Experimental notebooks
│   └── analysis/
│       ├── circle_fitting/       # Circle/hypersphere fitting experiments
│       ├── pca/                  # PCA visualizations
│       ├── pairwise_distance/    # Distance metrics analysis
│       └── variance_explained/   # Variance analysis
│
├── results/                      # Analysis outputs
│   ├── data/                     # Processed results (CSV, PKL)
│   ├── figures/                  # Static visualizations
│   └── html/                     # Interactive dashboards
│       ├── circle_fitting/       # Circle fitting visualizations
│       ├── pca/                  # PCA interactive plots
│       ├── pairwise_distances/   # Distance heatmaps
│       ├── layer_metrics/        # Layer-wise metrics
│       ├── variance/             # Variance explained plots
│       ├── analysis/             # Miscellaneous analyses
│       └── index.html            # Main dashboard
│
├── docs/                         # Documentation
└── archive/                      # Previous experiments
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
pip install torch transformers
pip install numpy pandas scikit-learn scipy
pip install matplotlib plotly seaborn
pip install jupyter notebook
```

### Model Requirements
- Uses `meta-llama/Llama-3.2-1B` model
- Requires ~4GB GPU memory (or runs on CPU)
- Hugging Face account with model access

### Quick Start

1. **Clone the repository**
```bash
git clone [repository-url]
cd manifolds
```

2. **Set up imports and configuration**
```python
from src import Config, load_model_and_tokenizer
from src import analyze_all_layers, create_incremental_datasets

config = Config()
model, tokenizer = load_model_and_tokenizer(config)
```

3. **Run analysis**
```python
# Create temporal datasets
train_df, test_df = create_incremental_datasets(config, max_offsets=70)

# Analyze all layers
results = analyze_all_layers(
    train_df, test_df, 
    context_name='add_plur',
    model=model, 
    tokenizer=tokenizer
)
```

## 📊 Experiments

### 1. Circle Fitting Analysis
Quantifies the circular structure of temporal embeddings:
- Fits circles in 2D PCA space
- Fits hyperspheres in original high-dimensional space
- Compares temporal vs control word patterns
- **Notebook**: `notebooks/analysis/circle_fitting/circle_fitting.ipynb`

### 2. PCA Visualization
Visualizes embedding projections across layers:
- Interactive layer-wise exploration
- Analysis of multiples-of-7 patterns
- Comparison of singular vs plural contexts
- **Notebooks**: `notebooks/analysis/pca/`

### 3. Pairwise Distance Analysis
Examines distance relationships between embeddings:
- Within-class vs between-class distances
- Circular distance metrics
- Separation quality metrics
- **Notebook**: `notebooks/analysis/pairwise_distance/`

### 4. Variance Explained Analysis
Studies dimensionality requirements:
- Components needed for variance thresholds
- Layer-wise compression patterns
- Temporal vs control comparison
- **Notebook**: `notebooks/analysis/variance_explained/`

## 🎨 Interactive Visualizations

Access interactive dashboards in `results/html/`:

1. **Main Dashboard** (`index.html`): Overview of all analyses
2. **Circle Fitting** (`circle_fitting/`): Interactive circle/hypersphere fitting
3. **PCA Explorer** (`pca/`): Layer-wise PCA projections with controls
4. **Distance Heatmaps** (`pairwise_distances/`): Interactive distance matrices
5. **Variance Analysis** (`variance/`): Cumulative variance curves

## 📈 Key Metrics

- **R² Score**: Goodness of circle fit (temporal: >0.95, control: <0.3)
- **Separation Ratio**: Between-class / within-class distance ratio
- **Variance Explained**: Components needed for 95% variance
- **Radius Evolution**: Change in manifold radius across layers

## 🔧 Configuration

Modify `src/config.py` for custom settings:

```python
@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = 16
    device: str = "cuda"  # or "cpu"
    n_components: int = 2  # PCA dimensions
    random_state: int = 42
```

## 📚 Dataset Types

### Temporal Data
- **add_sing**: "Monday plus 1 day equals Tuesday"
- **add_plur**: "Monday plus X days equals [Result]" (X ∈ 2-70, excluding multiples of 7)

### Control Data
- 45 common words across categories (food, emotions, actions, etc.)
- Used as baseline for comparison

## 🤝 Contributing

Contributions are welcome! Areas for extension:
- Analysis of other structured domains (months, numbers)
- Comparison across different model architectures
- Investigation of training dynamics
- Real-time visualization during inference

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@software{manifolds_analysis,
  title = {Language Model Manifolds Analysis},
  author = {[Your Name]},
  year = {2024},
  url = {[repository-url]}
}
```

## 📝 License

[Specify your license here]

## 🙏 Acknowledgments

- Llama model by Meta AI
- Built with PyTorch and Hugging Face Transformers
- Visualization powered by Plotly and Matplotlib

---

For detailed documentation, see `docs/CODEBASE_DOCUMENTATION.md`
