"""
Generalized Additive Model (GAM) Analysis

GAMs allow non-linear relationships between continuous predictors and outcomes,
while maintaining interpretability through partial dependence plots.

Predicts three target variables:
1. sem_dist_effect
2. diff_symbolic_r
3. relational_bias

Using predictors:
- Smooth terms for: dataset_size_log10, model_params_log10
- Factor terms for: is_classification, is_contrastive, is_generative, is_vlm, is_finetuned, is_cnn, is_transformer
- Interaction: smooth terms can vary by factor (e.g., model_size effect differs for CNN vs Transformer)

Outputs:
1. GAM results (text)
2. Partial dependence plots for continuous predictors
3. Coefficient bar plot for categorical predictors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, te
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
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
print("GENERALIZED ADDITIVE MODEL (GAM) ANALYSIS")
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
# Order: dataset_size_log10, model_params_log10, then binary predictors
X = df_clean[all_predictors].astype(float).values
feature_names = all_predictors

print(f"\nFeatures ({len(feature_names)}):")
for i, name in enumerate(feature_names):
    print(f"  {i}: {name}")

# Build GAM formula
# s(0) = smooth for dataset_size_log10
# s(1) = smooth for model_params_log10
# f(2) through f(8) = factor terms for binary predictors
# We can also add tensor product terms te(0, 2) for interactions

def build_gam_terms(n_splines=10, include_interactions=True):
    """Build GAM terms for the model."""
    terms = (
        s(0, n_splines=n_splines) +  # dataset_size_log10
        s(1, n_splines=n_splines) +  # model_params_log10
        f(2) +  # is_classification
        f(3) +  # is_contrastive
        f(4) +  # is_generative
        f(5) +  # is_vlm
        f(6) +  # is_finetuned
        f(7) +  # is_cnn
        f(8)    # is_transformer
    )

    if include_interactions:
        # Add interactions between continuous and key binary predictors
        # te() creates tensor product smooths
        terms = terms + te(0, 7) + te(0, 8)  # dataset_size x is_cnn, dataset_size x is_transformer
        terms = terms + te(1, 7) + te(1, 8)  # model_params x is_cnn, model_params x is_transformer

    return terms

# Run GAM for each target
results = {}

for target in targets:
    print("\n" + "=" * 80)
    print(f"MODEL: Predicting {target}")
    print("=" * 80)

    y = df_clean[target].values

    # Build and fit GAM with automatic lambda selection
    gam = LinearGAM(build_gam_terms(n_splines=10, include_interactions=True))

    # Grid search for optimal smoothing
    print("\nSearching for optimal smoothing parameters...")
    gam.gridsearch(X, y, progress=False)

    # Store results
    results[target] = {
        'model': gam,
        'r2_train': gam.statistics_['pseudo_r2']['explained_deviance'],
    }

    # Cross-validated R²
    print("Computing 5-fold cross-validated R²...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gam_cv = LinearGAM(build_gam_terms(n_splines=10, include_interactions=True))
        gam_cv.gridsearch(X_train, y_train, progress=False)

        y_pred = gam_cv.predict(X_test)
        cv_r2 = r2_score(y_test, y_pred)
        cv_r2_scores.append(cv_r2)

    results[target]['cv_r2_mean'] = np.mean(cv_r2_scores)
    results[target]['cv_r2_std'] = np.std(cv_r2_scores)

    # Print statistics
    print("\n" + "-" * 40)
    print("KEY STATISTICS:")
    print("-" * 40)
    print(f"Pseudo R² (training):  {results[target]['r2_train']:.4f}")
    print(f"R² (5-fold CV):        {results[target]['cv_r2_mean']:.4f} ± {results[target]['cv_r2_std']:.4f}")

    # Print GAM summary
    print("\n" + "-" * 40)
    print("GAM SUMMARY:")
    print("-" * 40)
    print(gam.summary())

# Summary comparison across models
print("\n" + "=" * 80)
print("SUMMARY COMPARISON ACROSS MODELS")
print("=" * 80)

summary_data = []
for target in targets:
    res = results[target]
    summary_data.append({
        'Target': target,
        'Pseudo R²': f"{res['r2_train']:.4f}",
        'CV R²': f"{res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save results to file
output_file = '/user_data/wenjiel2/abstraction/for_steve/gam_results.txt'
with open(output_file, 'w') as outfile:
    outfile.write("GENERALIZED ADDITIVE MODEL (GAM) RESULTS\n")
    outfile.write("=" * 80 + "\n\n")
    outfile.write("Model specification:\n")
    outfile.write("- Smooth terms (splines) for: dataset_size_log10, model_params_log10\n")
    outfile.write("- Factor terms for: is_classification, is_contrastive, is_generative, is_vlm, is_finetuned, is_cnn, is_transformer\n")
    outfile.write("- Tensor product interactions: dataset_size × is_cnn, dataset_size × is_transformer, model_params × is_cnn, model_params × is_transformer\n\n")

    for target in targets:
        res = results[target]
        outfile.write(f"\n{'=' * 80}\n")
        outfile.write(f"MODEL: Predicting {target}\n")
        outfile.write(f"{'=' * 80}\n")
        outfile.write(f"Pseudo R² (training): {res['r2_train']:.4f}\n")
        outfile.write(f"R² (5-fold CV): {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}\n\n")

print(f"\nFull results saved to: {output_file}")


# ============================================================================
# VISUALIZATION 1: Partial Dependence Plots for Continuous Predictors
# ============================================================================

output_path = '/user_data/wenjiel2/abstraction/for_steve/figures'
os.makedirs(output_path, exist_ok=True)

# Create partial dependence plots for each continuous predictor
fig, axes = plt.subplots(2, 3, figsize=(20, 14))

for col, target in enumerate(targets):
    gam = results[target]['model']
    color = TASK_COLORS[target]

    for row, (pred_idx, pred_name) in enumerate([(0, 'dataset_size_log10'), (1, 'model_params_log10')]):
        ax = axes[row, col]

        # Generate partial dependence
        XX = gam.generate_X_grid(term=pred_idx, n=100)
        pdep, confi = gam.partial_dependence(term=pred_idx, X=XX, width=0.95)

        # Get x values for this predictor
        x_vals = XX[:, pred_idx]

        # Plot
        ax.plot(x_vals, pdep, color=color, linewidth=2.5)
        ax.fill_between(x_vals, confi[:, 0], confi[:, 1], color=color, alpha=0.2)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel(PREDICTOR_LABELS[pred_name], fontsize=16)
        ax.set_ylabel('Partial Effect', fontsize=16)

        if row == 0:
            ax.set_title(f'{TASK_LABELS[target]}', fontweight='bold', fontsize=18)

# Add overall title
fig.suptitle('GAM Partial Dependence: Continuous Predictors',
             fontsize=24, fontweight='bold', y=1.02)

fig.text(0.5, -0.02, 'Shaded region = 95% confidence interval',
         ha='center', fontsize=14, style='italic')

plt.tight_layout()
plt.savefig(f'{output_path}/gam_partial_dependence.png', dpi=300, bbox_inches='tight')
print(f"\nSaved figure: {output_path}/gam_partial_dependence.png")
plt.close()


# ============================================================================
# VISUALIZATION 2: Feature Importance (Deviance Explained / Unique Variance)
# ============================================================================

def compute_deviance_importance(X, y, feature_names, n_splines=10):
    """
    Compute feature importance as the drop in deviance explained when each term is removed.
    This is analogous to unique variance explained (semi-partial R²).

    Returns importance as percentage of total deviance explained.
    """
    # Fit full model
    full_terms = (
        s(0, n_splines=n_splines) + s(1, n_splines=n_splines) +
        f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) +
        te(0, 7) + te(0, 8) + te(1, 7) + te(1, 8)
    )
    full_gam = LinearGAM(full_terms)
    full_gam.gridsearch(X, y, progress=False)
    full_deviance = full_gam.statistics_['pseudo_r2']['explained_deviance']

    importance = {}

    # For each main effect term (indices 0-8), compute importance
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
                    terms_list.append(s(i, n_splines=n_splines))
                else:
                    terms_list.append(f(i))

        # Add interaction terms (only if main effects are present)
        if idx not in [0, 7]:  # dataset_size or is_cnn
            terms_list.append(te(0, 7))
        if idx not in [0, 8]:  # dataset_size or is_transformer
            terms_list.append(te(0, 8))
        if idx not in [1, 7]:  # model_params or is_cnn
            terms_list.append(te(1, 7))
        if idx not in [1, 8]:  # model_params or is_transformer
            terms_list.append(te(1, 8))

        # Combine terms
        reduced_terms = terms_list[0]
        for t in terms_list[1:]:
            reduced_terms = reduced_terms + t

        reduced_gam = LinearGAM(reduced_terms)
        try:
            reduced_gam.gridsearch(X, y, progress=False)
            reduced_deviance = reduced_gam.statistics_['pseudo_r2']['explained_deviance']
            # Unique contribution = drop in deviance explained
            importance[name] = (full_deviance - reduced_deviance) * 100  # As percentage
        except:
            importance[name] = 0

    return importance, full_deviance


# Compute deviance-based importance for each target
print("\nComputing deviance-based feature importance...")
deviance_importance = {}
for target in targets:
    print(f"  Processing {target}...")
    y = df_clean[target].values
    importance, full_dev = compute_deviance_importance(X, y, feature_names)
    deviance_importance[target] = {
        'importance': importance,
        'full_deviance': full_dev
    }

# Aggregate importance into categories
# Architecture = Transformer + CNN
# Training Objectives = Finetuned + Contrastive + VLM + Generative + Classification

ARCHITECTURE_FEATURES = ['is_cnn', 'is_transformer']
TRAINING_OBJECTIVES_FEATURES = ['is_finetuned', 'is_contrastive', 'is_vlm', 'is_generative', 'is_classification']

aggregated_importance = {}
for target in targets:
    imp = deviance_importance[target]['importance']
    aggregated = {
        'dataset_size_log10': imp['dataset_size_log10'],
        'model_params_log10': imp['model_params_log10'],
        'Architecture': sum(imp[f] for f in ARCHITECTURE_FEATURES),
        'Training Objectives': sum(imp[f] for f in TRAINING_OBJECTIVES_FEATURES),
    }
    aggregated_importance[target] = aggregated

# Labels for aggregated predictors
AGGREGATED_LABELS = {
    'dataset_size_log10': 'Dataset Size (log10)',
    'model_params_log10': 'Model Params (log10)',
    'Architecture': 'Architecture',
    'Training Objectives': 'Training Objectives',
}

# Print results
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Unique Deviance Explained)")
print("=" * 80)

for target in targets:
    print(f"\n{TASK_LABELS[target]} (Total Pseudo R² = {deviance_importance[target]['full_deviance']:.4f})")
    print("-" * 50)
    sorted_imp = sorted(aggregated_importance[target].items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp:
        print(f"  {AGGREGATED_LABELS.get(name, name):25s}: {imp:6.2f}%")


# Plot deviance-based importance (aggregated)
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
    ax.set_xlabel('Unique Deviance Explained (%)', fontsize=18)
    ax.set_title(f'{TASK_LABELS[target]}\n(CV R² = {results[target]["cv_r2_mean"]:.3f})',
                 fontweight='bold', fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.invert_yaxis()

fig.suptitle('GAM Feature Importance (Unique Deviance Explained)',
             fontsize=24, fontweight='bold', y=1.02)

fig.text(0.5, -0.02, 'Architecture = CNN + Transformer | Training Objectives = Finetuned + Contrastive + VLM + Generative + Classification',
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig(f'{output_path}/gam_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\nSaved figure: {output_path}/gam_feature_importance.png")
plt.close()


# ============================================================================
# VISUALIZATION 2b: Individual Feature Importance (Non-Aggregated)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 10))

for ax, target in zip(axes, targets):
    color = TASK_COLORS[target]
    importance = deviance_importance[target]['importance']

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [PREDICTOR_LABELS.get(k, k) for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=color, alpha=0.7, edgecolor='black', linewidth=1.0)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=14)
    ax.set_xlabel('Unique Deviance Explained (%)', fontsize=18)
    ax.set_title(f'{TASK_LABELS[target]}\n(CV R² = {results[target]["cv_r2_mean"]:.3f})',
                 fontweight='bold', fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.invert_yaxis()

fig.suptitle('GAM Feature Importance (Individual Features)',
             fontsize=24, fontweight='bold', y=1.02)

fig.text(0.5, -0.02, 'Unique Deviance = Pseudo R²(full) - Pseudo R²(without term)',
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig(f'{output_path}/gam_feature_importance_individual.png', dpi=300, bbox_inches='tight')
print(f"Saved figure: {output_path}/gam_feature_importance_individual.png")
plt.close()


# ============================================================================
# VISUALIZATION 3: Comparison with OLS
# ============================================================================

# Also run simple OLS for comparison
from sklearn.linear_model import LinearRegression

print("\n" + "=" * 80)
print("COMPARISON: GAM vs OLS")
print("=" * 80)

comparison_data = []
for target in targets:
    y = df_clean[target].values

    # OLS CV R²
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ols_cv_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        y_pred = ols.predict(X_test_scaled)
        ols_cv_scores.append(r2_score(y_test, y_pred))

    comparison_data.append({
        'Target': TASK_LABELS[target],
        'OLS CV R²': f"{np.mean(ols_cv_scores):.4f} ± {np.std(ols_cv_scores):.4f}",
        'GAM CV R²': f"{results[target]['cv_r2_mean']:.4f} ± {results[target]['cv_r2_std']:.4f}",
        'Improvement': f"{(results[target]['cv_r2_mean'] - np.mean(ols_cv_scores))*100:.2f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison
with open(output_file, 'a') as outfile:
    outfile.write("\n\n" + "=" * 80 + "\n")
    outfile.write("COMPARISON: GAM vs OLS\n")
    outfile.write("=" * 80 + "\n")
    outfile.write(comparison_df.to_string(index=False))

print("\nDone!")
