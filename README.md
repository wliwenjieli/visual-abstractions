# Abstraction in Vision Models - Code and Data

This package contains all code and data for the manuscript "Abstraction in Vision Models: How do neural networks learn to abstract?"

## Contents

### Code
- `code/extraction/` - Model embedding extraction and evaluation pipeline
- `code/analysis/` - Statistical analyses and figure generation

### Data
- `data/stimuli/` - All experimental stimuli (identity, geometry, relation tasks)
- `data/model_performance/` - Per-trial model performance (parquet files)
- `data/aggregated_metrics/` - Aggregated metrics CSV file

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10.9 or compatible version.

## Usage

### Demo Notebook

The demo notebook `demo_relational_bias_calculation.ipynb` provides a step-by-step walkthrough.

**Important:** Run the notebook from the `submission_package/` directory:

```bash
cd submission_package
jupyter notebook demo_relational_bias_calculation.ipynb
```

The notebook uses relative paths (`./data/`) and expects to be run from this directory.

### 1. Extract Model Embeddings and Compute Metrics

```bash
cd code/extraction
python scripts/run_pipeline.py
```

This will:
1. Load models using the deepjuice library
2. Extract embeddings for all stimuli
3. Compute distances for each trial
4. Calculate three key metrics: semantic distance effect, regularity effect, relational bias

### 2. Run Statistical Analyses

```bash
cd code/analysis/multivariate

# GAM analysis with feature importance
python gam_deviance_reduction_figure.py

# Complete GAM analysis
python multivariate_gam.py

# OLS regression
python multivariate_regression.py

# Random Forest analysis
python multivariate_random_forest.py
```

### 3. Generate Figures

```bash
cd code/analysis/figures

# Main combined figure
python figure_combined_with_thresholds.py

# GAM effects visualization
python gam_effects_visualization.py

# Task-specific effects
python figure_semantic_distance_effect.py
python figure_regularity_effect.py
```

## Data Files

### Parquet Files (data/model_performance/)
Each parquet file contains per-trial distances for all models:
- `identity.parquet` - Identity task (63 trials × N models)
- `geometry.parquet` - Geometry task (63 trials × N models)
- `relation.parquet` - Relation task (126 trials × N models)
- `relation_patternonly.parquet` - Pattern-only variant

Columns include:
- Model identifiers (source, family, name, weights)
- Human distance (ground truth)
- Model distance (computed from embeddings)
- Trial metadata

### CSV File (data/aggregated_metrics/)
- `all_metrics_deduplicated.csv` - Aggregated metrics for all models

Key columns:
- Model metadata (architecture, dataset, training objective, etc.)
- `sem_dist_effect` - Semantic distance effect (Spearman correlation)
- `diff_symbolic_r` - Regularity effect (difference in correlations)
- `relational_bias` - Relational bias (proportion choosing relational match)
- Model size (parameters), dataset size, etc.

## Statistical Tests

All statistical tests are two-sided unless otherwise noted:
- **Spearman correlations**: Two-sided (scipy.stats.spearmanr)
- **Regression coefficients**: Two-sided (statsmodels OLS)
- **Binomial tests**: One-sided (α = 0.05, testing for above/below chance)

## Dependencies

See `requirements.txt` for complete list. Key packages:
- pandas, numpy, scipy for data processing
- scikit-learn for machine learning
- pygam for Generalized Additive Models
- statsmodels for regression
- matplotlib, seaborn for visualization
- torch, torchvision for model loading (extraction only)

## Citation

[Citation information to be added]

## Contact

[Contact information to be added]
