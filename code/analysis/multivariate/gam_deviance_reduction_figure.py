"""
GAM Deviance Reduction Figure
Using smoothing parameters from partial_dependence_actual_2_smooth.png:
- n_splines = 6
- lambda = 1.5

This figure shows the unique deviance explained by each predictor/category.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, te
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Smoothing levels (from partial_dependence_figure.py)
SMOOTHNESS_LEVELS = [
    {'n_splines': 5, 'lam': 3.0, 'label': '1_most_smooth'},
    {'n_splines': 6, 'lam': 1.5, 'label': '2_smooth'},
    {'n_splines': 7, 'lam': 0.8, 'label': '3_moderate'},
    {'n_splines': 8, 'lam': 0.4, 'label': '4_flexible'},
    {'n_splines': 10, 'lam': 0.1, 'label': '5_least_smooth'},
]

print(f"Will generate {len(SMOOTHNESS_LEVELS)} versions with different smoothing factors")

# Color scheme consistent with project
TASK_COLORS = {
    'sem_dist_effect': '#808080',      # grey
    'diff_symbolic_r': '#fc2190',      # pink
    'relational_bias': '#238b21',      # green
}

TASK_LABELS = {
    'sem_dist_effect': 'Semantic Distance Effect',
    'diff_symbolic_r': 'Regularity Effect',
    'relational_bias': 'Relational Bias',
}

PREDICTOR_LABELS = {
    'dataset_size_log10': 'Dataset Size',
    'model_params_log10': 'Model Size',
    'is_classification': 'Classification',
    'is_contrastive': 'Contrastive',
    'is_generative': 'Generative',
    'is_vlm': 'VLM',
    'is_finetuned': 'Finetuned',
    'is_cnn': 'CNN',
    'is_transformer': 'Transformer',
}

AGGREGATED_LABELS = {
    'dataset_size_log10': 'Dataset Size',
    'model_params_log10': 'Model Size',
    'Architecture': 'Architecture',
    'Training Objectives': 'Training Objectives',
}

# Set plot style - Nature journal quality (large fonts, no grid)
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.titlesize': 28,
    'font.family': 'sans-serif',
    'axes.linewidth': 2.0,
    'legend.fontsize': 18,
    'axes.grid': False,
})

# Load data
df = pd.read_csv('/user_data/wenjiel2/abstraction/for_steve/all_metrics_deduplicated.csv')

# Define predictors and targets
continuous_predictors = ['dataset_size_log10', 'model_params_log10']
binary_predictors = ['is_classification', 'is_contrastive', 'is_generative', 'is_vlm', 'is_finetuned', 'is_cnn', 'is_transformer']
all_predictors = continuous_predictors + binary_predictors
targets = ['sem_dist_effect', 'diff_symbolic_r', 'relational_bias']

# Convert boolean columns to numeric
for col in binary_predictors:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean data
cols_needed = all_predictors + targets
df_clean = df.dropna(subset=cols_needed).copy()
print(f"Samples: {len(df_clean)}")

# Prepare feature matrix
X = df_clean[all_predictors].astype(float).values
feature_names = all_predictors

# Output path
output_path = '/user_data/wenjiel2/abstraction/for_steve/figures_deduplicated'
os.makedirs(output_path, exist_ok=True)

# Aggregation setup
ARCHITECTURE_FEATURES = ['is_cnn', 'is_transformer']
TRAINING_OBJECTIVES_FEATURES = ['is_finetuned', 'is_contrastive', 'is_vlm', 'is_generative', 'is_classification']

# ============================================================================
# GAM ANALYSIS WITH SPECIFIED SMOOTHING
# ============================================================================

def build_gam_terms(n_splines, lam):
    """Build GAM terms with specified smoothing."""
    terms = (
        s(0, n_splines=n_splines, lam=lam) + s(1, n_splines=n_splines, lam=lam) +
        f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) +
        te(0, 7) + te(0, 8) + te(1, 7) + te(1, 8)
    )
    return terms

def compute_deviance_importance(X, y, n_splines, lam):
    """Compute feature importance as drop in deviance explained."""
    # Fit full model
    full_terms = (
        s(0, n_splines=n_splines, lam=lam) + s(1, n_splines=n_splines, lam=lam) +
        f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) +
        te(0, 7) + te(0, 8) + te(1, 7) + te(1, 8)
    )
    full_gam = LinearGAM(full_terms)
    full_gam.fit(X, y)
    full_deviance = full_gam.statistics_['pseudo_r2']['explained_deviance']

    importance = {}
    term_info = [
        (0, 'dataset_size_log10', 's'),
        (1, 'model_params_log10', 's'),
        (2, 'is_classification', 'f'),
        (3, 'is_contrastive', 'f'),
        (4, 'is_generative', 'f'),
        (5, 'is_vlm', 'f'),
        (6, 'is_finetuned', 'f'),
        (7, 'is_cnn', 'f'),
        (8, 'is_transformer', 'f'),
    ]

    for idx, name, term_type in term_info:
        # Build reduced model without this term
        terms_list = []
        for i, n, t in term_info:
            if i != idx:
                if t == 's':
                    terms_list.append(s(i, n_splines=n_splines, lam=lam))
                else:
                    terms_list.append(f(i))

        # Add interaction terms (only if main effects are present)
        if idx not in [0, 7]:
            terms_list.append(te(0, 7))
        if idx not in [0, 8]:
            terms_list.append(te(0, 8))
        if idx not in [1, 7]:
            terms_list.append(te(1, 7))
        if idx not in [1, 8]:
            terms_list.append(te(1, 8))

        # Combine terms
        reduced_terms = terms_list[0]
        for t in terms_list[1:]:
            reduced_terms = reduced_terms + t

        reduced_gam = LinearGAM(reduced_terms)
        try:
            reduced_gam.fit(X, y)
            reduced_deviance = reduced_gam.statistics_['pseudo_r2']['explained_deviance']
            importance[name] = (full_deviance - reduced_deviance) * 100
        except:
            importance[name] = 0

    return importance, full_deviance

# ============================================================================
# LOOP OVER ALL SMOOTHING LEVELS
# ============================================================================

# Colors for partial dependence
COLOR_DATASET = '#d62728'  # Red for dataset size
COLOR_MODEL = '#1f77b4'    # Blue for model size

for smooth_config in SMOOTHNESS_LEVELS:
    n_splines = smooth_config['n_splines']
    lam = smooth_config['lam']
    label = smooth_config['label']

    print(f"\n{'='*80}")
    print(f"Processing: {label} (n_splines={n_splines}, lambda={lam})")
    print("="*80)

    # Run GAM for each target
    results = {}
    deviance_importance = {}
    aggregated_importance = {}

    for target in targets:
        print(f"  Processing {target}...")
        y = df_clean[target].values

        # Fit GAM
        gam = LinearGAM(build_gam_terms(n_splines, lam))
        gam.fit(X, y)

        # CV R²
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, test_idx in kf.split(X):
            gam_cv = LinearGAM(build_gam_terms(n_splines, lam))
            gam_cv.fit(X[train_idx], y[train_idx])
            y_pred = gam_cv.predict(X[test_idx])
            cv_scores.append(r2_score(y[test_idx], y_pred))

        results[target] = {
            'model': gam,
            'r2_train': gam.statistics_['pseudo_r2']['explained_deviance'],
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
        }

        # Compute deviance importance
        importance, full_dev = compute_deviance_importance(X, y, n_splines, lam)
        deviance_importance[target] = {
            'importance': importance,
            'full_deviance': full_dev
        }

        # Aggregate
        aggregated_importance[target] = {
            'Dataset Size (log10)': importance['dataset_size_log10'],
            'Model Params (log10)': importance['model_params_log10'],
            'Architecture': importance['is_cnn'] + importance['is_transformer'],
            'Training Objectives': sum(importance[f] for f in TRAINING_OBJECTIVES_FEATURES),
        }

    # Print results
    print(f"\nGAM DEVIANCE REDUCTION (n_splines={n_splines}, lambda={lam})")
    for target in targets:
        print(f"\n{TASK_LABELS[target]}")
        print(f"  Pseudo R² = {results[target]['r2_train']:.4f}")
        print(f"  CV R² = {results[target]['cv_r2_mean']:.4f} ± {results[target]['cv_r2_std']:.4f}")
        print("  Unique Deviance Explained:")
        for name, val in sorted(aggregated_importance[target].items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:25s}: {val:6.2f}%")

    # ============================================================================
    # CREATE COMBINED FIGURE (2 rows: Deviance Reduction + Partial Dependence)
    # Row 1: Deviance reduction bar plots (individual features)
    # Row 2: Partial dependence with dual x-axes (dataset size on top, model size on bottom)
    # ============================================================================

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # Row 1: Deviance Reduction (Individual Features) - proportional to total R²
    # First pass: compute all values to find global max for shared x-axis
    all_scaled_values = []
    row1_data = []
    for col, target in enumerate(targets):
        importance = deviance_importance[target]['importance']
        cv_r2 = results[target]['cv_r2_mean']
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [PREDICTOR_LABELS.get(k, k) for k, v in sorted_items]
        raw_values = [v for k, v in sorted_items]
        # Scale so that bars sum to R² (proportional contribution)
        total_unique = sum(raw_values)
        if total_unique > 0:
            scaled_values = [v / total_unique * cv_r2 for v in raw_values]
        else:
            scaled_values = raw_values
        all_scaled_values.extend(scaled_values)
        row1_data.append((names, scaled_values, cv_r2))

    # Second pass: plot with fixed x-axis ticks
    for col, target in enumerate(targets):
        ax = axes[0, col]
        color = TASK_COLORS[target]
        names, values, cv_r2 = row1_data[col]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=color, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=22)
        ax.set_xlabel('Contribution to R²', fontsize=24)
        # Title with bold task name and italic R²
        ax.set_title(f'{TASK_LABELS[target]}\n($R^2$ = {cv_r2:.2f})',
                     fontweight='bold', fontsize=26)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_xlim(0, 0.16)  # Fixed x-axis range
        ax.set_xticks([0.00, 0.05, 0.10, 0.15])
        ax.invert_yaxis()

    # Row 2: Partial Dependence (Dataset Size on bottom, Model Size on top)
    for col, target in enumerate(targets):
        ax = axes[1, col]
        gam = results[target]['model']

        # Generate partial dependence for dataset_size_log10 (index 0)
        XX_data = gam.generate_X_grid(term=0, n=100)
        pdep_data, confi_data = gam.partial_dependence(term=0, X=XX_data, width=0.95)
        x_data = XX_data[:, 0]

        # Generate partial dependence for model_params_log10 (index 1)
        XX_model = gam.generate_X_grid(term=1, n=100)
        pdep_model, confi_model = gam.partial_dependence(term=1, X=XX_model, width=0.95)
        x_model = XX_model[:, 1]

        # Plot dataset size on primary (bottom) x-axis - RED
        line1, = ax.plot(x_data, pdep_data, color=COLOR_DATASET, linewidth=3, label='Dataset Size')
        ax.fill_between(x_data, confi_data[:, 0], confi_data[:, 1], color=COLOR_DATASET, alpha=0.2)

        ax.set_xlabel('Dataset Size (log10)', fontsize=24, color=COLOR_DATASET)
        ax.tick_params(axis='x', labelsize=20, colors=COLOR_DATASET)
        ax.set_ylabel('Partial Effect', fontsize=24)
        ax.tick_params(axis='y', labelsize=20)

        # Create secondary (top) x-axis for model size - BLUE
        ax2 = ax.twiny()
        line2, = ax2.plot(x_model, pdep_model, color=COLOR_MODEL, linewidth=3, label='Model Size')
        ax2.fill_between(x_model, confi_model[:, 0], confi_model[:, 1], color=COLOR_MODEL, alpha=0.2)

        ax2.set_xlabel('Model Size (log10)', fontsize=24, color=COLOR_MODEL)
        ax2.tick_params(axis='x', labelsize=20, colors=COLOR_MODEL)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Add legend in upper left corner
        ax.legend([line1, line2], ['Dataset Size', 'Model Size'], loc='upper left', fontsize=18)

    fig.suptitle(f'GAM Analysis (n_splines={n_splines}, λ={lam})',
                 fontsize=32, fontweight='bold', y=1.02)

    fig.text(0.5, -0.01, 'Shaded region = 95% CI',
             ha='center', fontsize=18, style='italic')

    plt.tight_layout()
    output_filename = f'{output_path}/gam_deviance_reduction_{label}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {output_filename}")
    plt.close()

print("\n" + "="*80)
print("All 5 versions generated!")
print("="*80)
