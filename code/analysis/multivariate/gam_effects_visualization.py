"""
GAM Effects Visualization

Creates comprehensive visualizations showing:
1. Partial dependence plots for continuous predictors (with direction)
2. Effect estimates for categorical predictors (with confidence intervals)
3. Combined coefficient summary showing sign and magnitude
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, te
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Color scheme
TASK_COLORS = {
    'sem_dist_effect': '#808080',      # grey
    'diff_symbolic_r': '#FF69B4',      # pink
    'relational_bias': '#228B22',      # dark green
}

TASK_LABELS = {
    'sem_dist_effect': 'Semantic Distance Effect',
    'diff_symbolic_r': 'Regularity Effect',
    'relational_bias': 'Relational Bias',
}

# Predictor labels
PREDICTOR_LABELS = {
    'dataset_size_log10': 'Dataset Size (log10)',
    'model_params_log10': 'Model Params (log10)',
    'is_classification': 'Classification',
    'is_contrastive': 'Contrastive',
    'is_generative': 'Generative',
    'is_vlm': 'VLM',
    'is_finetuned': 'Finetuned',
    'is_cnn': 'CNN',
    'is_transformer': 'Transformer',
}

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 20,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

# Load data
df = pd.read_csv('/user_data/wenjiel2/abstraction/for_steve/all_metrics_deduplicated.csv')

print("=" * 80)
print("GAM EFFECTS VISUALIZATION")
print("=" * 80)

# Define predictors
continuous_predictors = ['dataset_size_log10', 'model_params_log10']
binary_predictors = ['is_classification', 'is_contrastive', 'is_generative', 'is_vlm', 'is_finetuned', 'is_cnn', 'is_transformer']
all_predictors = continuous_predictors + binary_predictors
targets = ['sem_dist_effect', 'diff_symbolic_r', 'relational_bias']

# Convert boolean columns
for col in binary_predictors:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean data
cols_needed = all_predictors + targets
df_clean = df.dropna(subset=cols_needed).copy()
print(f"Samples: {len(df_clean)}")

# Prepare feature matrix
X = df_clean[all_predictors].astype(float).values

# GAM formula with interactions
gam_formula = (
    s(0, n_splines=10) + s(1, n_splines=10) +
    f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) +
    te(0, 7) + te(0, 8) + te(1, 7) + te(1, 8)
)

output_path = '/user_data/wenjiel2/abstraction/for_steve/figures_deduplicated'

# Fit GAMs and extract effects
results = {}
for target in targets:
    print(f"\nFitting GAM for {target}...")
    y = df_clean[target].values

    gam = LinearGAM(gam_formula)
    gam.gridsearch(X, y, progress=False)

    results[target] = {
        'gam': gam,
        'X': X,
        'y': y,
        'pseudo_r2': gam.statistics_['pseudo_r2']['explained_deviance']
    }

# ============================================================================
# FIGURE 1: Categorical Predictor Effects (Bar plot with error bars)
# ============================================================================
print("\nCreating categorical effects plot...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for idx, target in enumerate(targets):
    ax = axes[idx]
    gam = results[target]['gam']
    X = results[target]['X']
    y = results[target]['y']

    # For categorical predictors, compute the effect as the difference
    # in predicted value when the predictor is 1 vs 0, holding others at mean
    effects = {}

    # Create baseline: all continuous at mean, all categorical at 0
    X_baseline = np.zeros((1, X.shape[1]))
    X_baseline[0, 0] = X[:, 0].mean()  # dataset_size mean
    X_baseline[0, 1] = X[:, 1].mean()  # model_params mean

    baseline_pred = gam.predict(X_baseline)[0]

    # For each categorical predictor
    for i, pred_name in enumerate(binary_predictors):
        pred_idx = i + 2  # offset by 2 for continuous predictors

        # Set this predictor to 1
        X_effect = X_baseline.copy()
        X_effect[0, pred_idx] = 1

        effect_pred = gam.predict(X_effect)[0]
        effect = effect_pred - baseline_pred

        effects[pred_name] = {
            'effect': effect,
        }

    # Sort by absolute effect size
    sorted_preds = sorted(effects.keys(), key=lambda x: abs(effects[x]['effect']), reverse=True)

    # Plot
    y_pos = np.arange(len(sorted_preds))
    effect_vals = [effects[p]['effect'] for p in sorted_preds]

    # Color by sign: positive = task color, negative = red
    colors = [TASK_COLORS[target] if e >= 0 else '#D55E00' for e in effect_vals]

    bars = ax.barh(y_pos, effect_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([PREDICTOR_LABELS[p] for p in sorted_preds])
    ax.set_xlabel('Effect on Outcome')
    ax.set_title(f"{TASK_LABELS[target]}\n(Pseudo R² = {results[target]['pseudo_r2']:.3f})", fontweight='bold')
    ax.invert_yaxis()

plt.suptitle('GAM: Categorical Predictor Effects\n(Effect of setting predictor to 1 vs 0, holding others constant)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_path}/gam_categorical_effects.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}/gam_categorical_effects.png")
plt.close()


# ============================================================================
# FIGURE 2: Summary table of all effects
# ============================================================================
print("\nCreating summary table...")

# Print text summary
print("\n" + "=" * 80)
print("EFFECT DIRECTION SUMMARY")
print("=" * 80)

for target in targets:
    print(f"\n{TASK_LABELS[target]}:")
    print("-" * 50)

    gam = results[target]['gam']
    X = results[target]['X']

    # Continuous predictors - compute slope at median
    X_baseline = np.zeros((1, X.shape[1]))
    X_baseline[0, 0] = X[:, 0].mean()
    X_baseline[0, 1] = X[:, 1].mean()

    # Dataset size effect (change from 25th to 75th percentile)
    X_low = X_baseline.copy()
    X_high = X_baseline.copy()
    X_low[0, 0] = np.percentile(X[:, 0], 25)
    X_high[0, 0] = np.percentile(X[:, 0], 75)
    ds_effect = gam.predict(X_high)[0] - gam.predict(X_low)[0]
    sign = "+" if ds_effect > 0 else "-"
    print(f"  Dataset Size (25th→75th %ile): {sign}{abs(ds_effect):.4f}")

    # Model params effect
    X_low = X_baseline.copy()
    X_high = X_baseline.copy()
    X_low[0, 1] = np.percentile(X[:, 1], 25)
    X_high[0, 1] = np.percentile(X[:, 1], 75)
    mp_effect = gam.predict(X_high)[0] - gam.predict(X_low)[0]
    sign = "+" if mp_effect > 0 else "-"
    print(f"  Model Params (25th→75th %ile): {sign}{abs(mp_effect):.4f}")

    # Categorical predictors
    baseline_pred = gam.predict(X_baseline)[0]
    for i, pred_name in enumerate(binary_predictors):
        pred_idx = i + 2
        X_effect = X_baseline.copy()
        X_effect[0, pred_idx] = 1
        effect = gam.predict(X_effect)[0] - baseline_pred
        sign = "+" if effect > 0 else "-"
        print(f"  {PREDICTOR_LABELS[pred_name]:20s}: {sign}{abs(effect):.4f}")


# ============================================================================
# FIGURE 3: Combined heatmap of effects
# ============================================================================
print("\nCreating effects heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Compute all effects
all_effects = []
predictor_names = []

for target in targets:
    gam = results[target]['gam']
    X = results[target]['X']

    target_effects = []

    X_baseline = np.zeros((1, X.shape[1]))
    X_baseline[0, 0] = X[:, 0].mean()
    X_baseline[0, 1] = X[:, 1].mean()

    # Dataset size effect (standardized: per SD change)
    ds_std = X[:, 0].std()
    X_low = X_baseline.copy()
    X_high = X_baseline.copy()
    X_low[0, 0] = X[:, 0].mean() - ds_std/2
    X_high[0, 0] = X[:, 0].mean() + ds_std/2
    ds_effect = gam.predict(X_high)[0] - gam.predict(X_low)[0]
    target_effects.append(ds_effect)

    # Model params effect (per SD)
    mp_std = X[:, 1].std()
    X_low = X_baseline.copy()
    X_high = X_baseline.copy()
    X_low[0, 1] = X[:, 1].mean() - mp_std/2
    X_high[0, 1] = X[:, 1].mean() + mp_std/2
    mp_effect = gam.predict(X_high)[0] - gam.predict(X_low)[0]
    target_effects.append(mp_effect)

    # Categorical effects
    baseline_pred = gam.predict(X_baseline)[0]
    for i, pred_name in enumerate(binary_predictors):
        pred_idx = i + 2
        X_effect = X_baseline.copy()
        X_effect[0, pred_idx] = 1
        effect = gam.predict(X_effect)[0] - baseline_pred
        target_effects.append(effect)

    all_effects.append(target_effects)

    if target == targets[0]:
        predictor_names = ['Dataset Size\n(per SD)', 'Model Params\n(per SD)'] + \
                          [PREDICTOR_LABELS[p] for p in binary_predictors]

effects_matrix = np.array(all_effects).T

# Create heatmap
im = ax.imshow(effects_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

# Add text annotations
for i in range(len(predictor_names)):
    for j in range(len(targets)):
        val = effects_matrix[i, j]
        color = 'white' if abs(val) > 0.08 else 'black'
        ax.text(j, i, f'{val:+.3f}', ha='center', va='center', color=color, fontsize=11)

ax.set_xticks(range(len(targets)))
ax.set_xticklabels([TASK_LABELS[t] for t in targets], fontsize=14)
ax.set_yticks(range(len(predictor_names)))
ax.set_yticklabels(predictor_names, fontsize=12)

plt.colorbar(im, ax=ax, label='Effect Size', shrink=0.8)
ax.set_title('GAM Effect Sizes by Predictor and Outcome\n(Blue = negative, Red = positive)',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_path}/gam_effects_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}/gam_effects_heatmap.png")
plt.close()

print("\nDone!")
