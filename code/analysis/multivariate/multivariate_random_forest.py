"""
Random Forest Analysis

Random Forest regression to predict three target variables:
1. sem_dist_effect
2. diff_symbolic_r
3. relational_bias

Using predictors:
- Continuous: dataset_size_log10, model_params_log10
- Binary: is_classification, is_contrastive, is_generative, is_vlm, is_finetuned, is_cnn, is_transformer

Feature importance computed via permutation importance (drop in R² when feature is permuted).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Color scheme consistent with project
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

# Predictor labels for plotting
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

# Set plot style - publication quality fonts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 24,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

# Load data
df = pd.read_csv('/user_data/wenjiel2/abstraction/for_steve/all_metrics_deduplicated.csv')

print("=" * 80)
print("RANDOM FOREST ANALYSIS")
print("=" * 80)
print(f"\nTotal samples: {len(df)}")

# Define predictors and targets
continuous_predictors = ['dataset_size_log10', 'model_params_log10']
binary_predictors = ['is_classification', 'is_contrastive', 'is_generative', 'is_vlm', 'is_finetuned', 'is_cnn', 'is_transformer']
all_predictors = continuous_predictors + binary_predictors
targets = ['sem_dist_effect', 'diff_symbolic_r', 'relational_bias']

# Convert boolean columns to numeric
for col in binary_predictors:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for missing values and clean
cols_needed = all_predictors + targets
df_clean = df.dropna(subset=cols_needed).copy()
print(f"\nSamples after removing missing values: {len(df_clean)}")

# Prepare feature matrix
X = df_clean[all_predictors].astype(float).values
feature_names = all_predictors

print(f"\nFeatures ({len(feature_names)}):")
for i, name in enumerate(feature_names):
    print(f"  {i}: {name}")

# Output paths
output_path = '/user_data/wenjiel2/abstraction/for_steve/figures'
os.makedirs(output_path, exist_ok=True)
output_file = '/user_data/wenjiel2/abstraction/for_steve/random_forest_results.txt'

# Run Random Forest for each target
results = {}

for target in targets:
    print("\n" + "=" * 80)
    print(f"MODEL: Predicting {target}")
    print("=" * 80)

    y = df_clean[target].values

    # Fit Random Forest with reasonable hyperparameters
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X, y)

    # Training R²
    y_pred_train = rf.predict(X)
    r2_train = r2_score(y, y_pred_train)

    # Cross-validated R²
    print("\nComputing 5-fold cross-validated R²...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')

    # Permutation importance (more reliable than impurity-based importance)
    print("Computing permutation importance...")
    perm_importance = permutation_importance(rf, X, y, n_repeats=30, random_state=42, n_jobs=-1)

    # Store results
    results[target] = {
        'model': rf,
        'r2_train': r2_train,
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std,
        'feature_importance_impurity': rf.feature_importances_,
    }

    # Print statistics
    print("\n" + "-" * 40)
    print("KEY STATISTICS:")
    print("-" * 40)
    print(f"R² (training):         {r2_train:.4f}")
    print(f"R² (5-fold CV):        {results[target]['cv_r2_mean']:.4f} ± {results[target]['cv_r2_std']:.4f}")

    print("\n" + "-" * 40)
    print("PERMUTATION IMPORTANCE (Drop in R²):")
    print("-" * 40)

    # Sort by importance
    sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
    for idx in sorted_idx:
        name = feature_names[idx]
        imp = perm_importance.importances_mean[idx]
        std = perm_importance.importances_std[idx]
        print(f"  {PREDICTOR_LABELS.get(name, name):25s}: {imp*100:6.2f}% ± {std*100:.2f}%")

# Save results to file
with open(output_file, 'w') as outfile:
    outfile.write("=" * 80 + "\n")
    outfile.write("RANDOM FOREST ANALYSIS RESULTS\n")
    outfile.write("=" * 80 + "\n\n")

    for target in targets:
        outfile.write(f"\n{TASK_LABELS[target]}\n")
        outfile.write("-" * 50 + "\n")
        outfile.write(f"R² (training):  {results[target]['r2_train']:.4f}\n")
        outfile.write(f"R² (5-fold CV): {results[target]['cv_r2_mean']:.4f} ± {results[target]['cv_r2_std']:.4f}\n\n")

        outfile.write("Permutation Importance:\n")
        sorted_idx = np.argsort(results[target]['perm_importance_mean'])[::-1]
        for idx in sorted_idx:
            name = feature_names[idx]
            imp = results[target]['perm_importance_mean'][idx]
            std = results[target]['perm_importance_std'][idx]
            outfile.write(f"  {PREDICTOR_LABELS.get(name, name):25s}: {imp*100:6.2f}% ± {std*100:.2f}%\n")

print(f"\nResults saved to: {output_file}")

# ============================================================================
# AGGREGATE IMPORTANCE INTO CATEGORIES
# ============================================================================

# Architecture = Transformer + CNN
# Training Objectives = Finetuned + Contrastive + VLM + Generative + Classification

ARCHITECTURE_FEATURES = ['is_cnn', 'is_transformer']
TRAINING_OBJECTIVES_FEATURES = ['is_finetuned', 'is_contrastive', 'is_vlm', 'is_generative', 'is_classification']

# Get feature indices
arch_idx = [feature_names.index(f) for f in ARCHITECTURE_FEATURES]
train_obj_idx = [feature_names.index(f) for f in TRAINING_OBJECTIVES_FEATURES]
dataset_idx = feature_names.index('dataset_size_log10')
params_idx = feature_names.index('model_params_log10')

aggregated_importance = {}
for target in targets:
    perm_imp = results[target]['perm_importance_mean']
    aggregated = {
        'dataset_size_log10': perm_imp[dataset_idx] * 100,
        'model_params_log10': perm_imp[params_idx] * 100,
        'Architecture': sum(perm_imp[i] for i in arch_idx) * 100,
        'Training Objectives': sum(perm_imp[i] for i in train_obj_idx) * 100,
    }
    aggregated_importance[target] = aggregated

# Labels for aggregated predictors
AGGREGATED_LABELS = {
    'dataset_size_log10': 'Dataset Size (log10)',
    'model_params_log10': 'Model Params (log10)',
    'Architecture': 'Architecture',
    'Training Objectives': 'Training Objectives',
}

# Print aggregated results
print("\n" + "=" * 80)
print("AGGREGATED FEATURE IMPORTANCE (Permutation Importance)")
print("=" * 80)

for target in targets:
    print(f"\n{TASK_LABELS[target]} (CV R² = {results[target]['cv_r2_mean']:.4f})")
    print("-" * 50)
    sorted_imp = sorted(aggregated_importance[target].items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp:
        print(f"  {AGGREGATED_LABELS.get(name, name):25s}: {imp:6.2f}%")

# ============================================================================
# VISUALIZATION: Aggregated Feature Importance
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 8))

for ax, target in zip(axes, targets):
    color = TASK_COLORS[target]
    importance = aggregated_importance[target]

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [AGGREGATED_LABELS.get(k, k) for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=16)
    ax.set_xlabel('Permutation Importance (%)', fontsize=18)
    ax.set_title(f'{TASK_LABELS[target]}\n(CV R² = {results[target]["cv_r2_mean"]:.3f})',
                 fontweight='bold', fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.invert_yaxis()

fig.suptitle('Random Forest Feature Importance (Permutation-Based)',
             fontsize=24, fontweight='bold', y=1.02)

fig.text(0.5, -0.02, 'Architecture = CNN + Transformer | Training Objectives = Finetuned + Contrastive + VLM + Generative + Classification',
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig(f'{output_path}/rf_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\nSaved figure: {output_path}/rf_feature_importance.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Individual Feature Importance (Non-Aggregated)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 10))

for ax, target in zip(axes, targets):
    color = TASK_COLORS[target]
    perm_imp = results[target]['perm_importance_mean']
    perm_std = results[target]['perm_importance_std']

    # Sort by importance
    sorted_idx = np.argsort(perm_imp)[::-1]
    names = [PREDICTOR_LABELS.get(feature_names[i], feature_names[i]) for i in sorted_idx]
    values = [perm_imp[i] * 100 for i in sorted_idx]
    errors = [perm_std[i] * 100 for i in sorted_idx]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, xerr=errors, color=color, alpha=0.7, edgecolor='black', linewidth=1.0,
            error_kw={'capsize': 3, 'capthick': 1, 'elinewidth': 1})

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=14)
    ax.set_xlabel('Permutation Importance (%)', fontsize=18)
    ax.set_title(f'{TASK_LABELS[target]}\n(CV R² = {results[target]["cv_r2_mean"]:.3f})',
                 fontweight='bold', fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.invert_yaxis()

fig.suptitle('Random Forest Feature Importance (Individual Features)',
             fontsize=24, fontweight='bold', y=1.02)

fig.text(0.5, -0.02, 'Error bars show standard deviation across 30 permutation repeats',
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig(f'{output_path}/rf_feature_importance_individual.png', dpi=300, bbox_inches='tight')
print(f"Saved figure: {output_path}/rf_feature_importance_individual.png")
plt.close()

# ============================================================================
# COMPARISON: Random Forest vs GAM vs OLS
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

summary_data = []
for target in targets:
    summary_data.append({
        'Target': TASK_LABELS[target],
        'RF CV R²': f"{results[target]['cv_r2_mean']:.4f} ± {results[target]['cv_r2_std']:.4f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary
with open(output_file, 'a') as outfile:
    outfile.write("\n\n" + "=" * 80 + "\n")
    outfile.write("SUMMARY\n")
    outfile.write("=" * 80 + "\n")
    outfile.write(summary_df.to_string(index=False))

print("\nDone!")
