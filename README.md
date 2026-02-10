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

### Demo Notebook (Recommended Starting Point)

The demo notebook `demo_relational_bias.ipynb` provides an interactive, step-by-step walkthrough of computing relational bias for a single model. This is the easiest way to understand the methodology and reproduce results.

#### What the Demo Shows

The notebook demonstrates:
1. **Automatic DeepNSD installation** - No manual setup required
2. **Loading a pre-trained model** - Using the publicly available DeepNSD library
3. **Feature extraction** - Extracting embeddings from the penultimate layer
4. **Distance computation** - Computing cosine distances between image embeddings
5. **Relational bias calculation** - Computing the proportion of trials where the model prefers relational over perceptual matches

#### Running the Demo

**Important:** Run the notebook from the `submission_package/` directory:

```bash
cd submission_package
jupyter notebook demo_relational_bias.ipynb
```

#### Requirements

The demo requires:
- Python 3.10+ (tested with 3.10.9)
- Jupyter Notebook or JupyterLab
- PyTorch 2.0+ with torchvision
- Basic scientific Python packages (numpy, pandas, matplotlib, scikit-learn)

**Note:** The notebook will automatically clone and install [DeepNSD](https://github.com/ColinConwell/DeepNSD) if not already available. DeepNSD is the publicly available library that provides the core functionality for model loading and feature extraction.

#### About DeepJuice vs DeepNSD

The extraction pipeline uses **DeepJuice**, which is currently in private beta. For public reproducibility, the demo notebook uses **DeepNSD**, which is the open-source library containing DeepJuice's core functionality.

If you're interested in using DeepJuice directly, please contact the authors.


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

## Dependencies

See `requirements.txt` for complete list. Key packages:
- pandas, numpy, scipy for data processing
- scikit-learn for machine learning
- pygam for Generalized Additive Models
- statsmodels for regression
- matplotlib, seaborn for visualization
- torch, torchvision for model loading (extraction only)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

[Citation information to be added]

## Contact

[Contact information to be added]
