"""
Multivariate Linear Regression Analysis with Visualization

Predicts three target variables:
1. sem_dist_effect
2. diff_symbolic_r
3. relational_bias

Using predictors (standardized):
- dataset_size_log10
- model_params_log10
- is_classification
- is_contrastive
- is_generative
- is_vlm
- is_finetuned
- is_cnn (binary)
- is_transformer (binary)

Outputs:
1. Regression results (text)
2. Standardized coefficient bar plot
3. Cross-validated unique variance explained plot (5-fold CV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
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
    'dataset_size_log10': 'Dataset Size\n(log10)',
    'model_params_log10': 'Model Params\n(log10)',
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
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 24,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

# Load data
df = pd.read_csv('/user_data/wenjiel2/abstraction/for_steve/all_metrics_deduplicated.csv')

print("=" * 80)
print("MULTIVARIATE LINEAR REGRESSION ANALYSIS")
print("=" * 80)
print(f"\nTotal samples: {len(df)}")

# Define predictors and targets
numeric_predictors = ['dataset_size_log10', 'model_params_log10']
binary_predictors = ['is_classification', 'is_contrastive', 'is_generative', 'is_vlm', 'is_finetuned', 'is_cnn', 'is_transformer']
all_predictors = numeric_predictors + binary_predictors
targets = ['sem_dist_effect', 'diff_symbolic_r', 'relational_bias']

# Convert boolean columns to numeric
for col in binary_predictors:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Print architecture distribution
print("\n--- Architecture Distribution ---")
print(f"  is_cnn: {df['is_cnn'].sum()} ({df['is_cnn'].mean()*100:.1f}%)")
print(f"  is_transformer: {df['is_transformer'].sum()} ({df['is_transformer'].mean()*100:.1f}%)")

# Check for missing values
print("\n" + "-" * 80)
print("DATA SUMMARY")
print("-" * 80)

for col in all_predictors + targets:
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing} missing values")

# Drop rows with missing values
cols_needed = all_predictors + targets
df_clean = df.dropna(subset=cols_needed).copy()
print(f"\nSamples after removing missing values: {len(df_clean)}")

# Standardize all predictors (z-score)
scaler = StandardScaler()
df_standardized = df_clean.copy()
df_standardized[all_predictors] = scaler.fit_transform(df_clean[all_predictors])
print("Predictors have been standardized (mean=0, sd=1)")

# Prepare X matrix (add constant for intercept)
X = df_standardized[all_predictors].astype(float)
X = sm.add_constant(X)

# Run regression for each target
results = {}

for target in targets:
    print("\n" + "=" * 80)
    print(f"MODEL: Predicting {target}")
    print("=" * 80)

    y = df_standardized[target]

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    results[target] = model

    # Print summary
    print(model.summary())

    # Print key statistics
    print("\n" + "-" * 40)
    print("KEY STATISTICS:")
    print("-" * 40)
    print(f"R-squared:           {model.rsquared:.4f}")
    print(f"Adjusted R-squared:  {model.rsquared_adj:.4f}")
    print(f"F-statistic:         {model.fvalue:.4f}")
    print(f"Prob (F-statistic):  {model.f_pvalue:.2e}")

    # Significant predictors (p < 0.05)
    print("\n" + "-" * 40)
    print("SIGNIFICANT PREDICTORS (p < 0.05):")
    print("-" * 40)
    sig_params = model.pvalues[model.pvalues < 0.05]
    if len(sig_params) > 0:
        for param in sig_params.index:
            if param != 'const':
                coef = model.params[param]
                pval = model.pvalues[param]
                print(f"  {param:25s}: β = {coef:+.4f}, p = {pval:.4e}")
    else:
        print("  None")

# Summary comparison across models
print("\n" + "=" * 80)
print("SUMMARY COMPARISON ACROSS MODELS")
print("=" * 80)

summary_data = []
for target in targets:
    model = results[target]
    summary_data.append({
        'Target': target,
        'R²': f"{model.rsquared:.4f}",
        'Adj R²': f"{model.rsquared_adj:.4f}",
        'F-stat': f"{model.fvalue:.2f}",
        'F p-value': f"{model.f_pvalue:.2e}",
        'N sig predictors': sum(model.pvalues < 0.05) - 1  # -1 for intercept
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Coefficient comparison for key predictors
print("\n" + "-" * 80)
print("STANDARDIZED COEFFICIENT COMPARISON (β)")
print("-" * 80)

coef_comparison = pd.DataFrame(index=all_predictors, columns=targets)

for target in targets:
    model = results[target]
    for pred in all_predictors:
        if pred in model.params.index:
            coef = model.params[pred]
            pval = model.pvalues[pred]
            sig = "*" if pval < 0.05 else ""
            sig += "*" if pval < 0.01 else ""
            sig += "*" if pval < 0.001 else ""
            coef_comparison.loc[pred, target] = f"{coef:+.4f}{sig}"

print(coef_comparison.to_string())
print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001")

# Save results to file
output_file = '/user_data/wenjiel2/abstraction/for_steve/regression_results.txt'
with open(output_file, 'w') as f:
    f.write("MULTIVARIATE LINEAR REGRESSION RESULTS (STANDARDIZED)\n")
    f.write("=" * 80 + "\n\n")

    for target in targets:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"MODEL: Predicting {target}\n")
        f.write(f"{'=' * 80}\n")
        f.write(results[target].summary().as_text())
        f.write("\n")

print(f"\nFull results saved to: {output_file}")


# ============================================================================
# VISUALIZATION 1: Standardized Coefficient Bar Plot
# ============================================================================

def plot_coefficients(ax, model, target, color, predictors):
    """Plot coefficient bar chart for a single model, sorted by coefficient magnitude."""
    # Get coefficients and confidence intervals (excluding const)
    coefs = model.params[predictors]
    conf_int = model.conf_int().loc[predictors]
    pvalues = model.pvalues[predictors]

    # Sort predictors by coefficient value (largest to smallest)
    sorted_indices = np.argsort(coefs.values)[::-1]
    sorted_predictors = [predictors[i] for i in sorted_indices]
    sorted_coefs = coefs.values[sorted_indices]
    sorted_pvalues = pvalues.values[sorted_indices]
    sorted_conf_int_low = conf_int[0].values[sorted_indices]
    sorted_conf_int_high = conf_int[1].values[sorted_indices]

    # Calculate error bars
    errors = np.array([
        sorted_coefs - sorted_conf_int_low,
        sorted_conf_int_high - sorted_coefs
    ])

    # Create bar positions
    y_pos = np.arange(len(sorted_predictors))

    # Create horizontal bar plot
    ax.barh(y_pos, sorted_coefs, xerr=errors,
            color=color, alpha=0.7, edgecolor='black', linewidth=1.0,
            capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    # Add significance markers
    for i, (coef, pval) in enumerate(zip(sorted_coefs, sorted_pvalues)):
        if pval < 0.001:
            marker = '***'
        elif pval < 0.01:
            marker = '**'
        elif pval < 0.05:
            marker = '*'
        else:
            marker = ''

        if marker:
            x_pos = coef + errors[1, i] + 0.02 if coef > 0 else coef - errors[0, i] - 0.02
            ha = 'left' if coef > 0 else 'right'
            ax.text(x_pos, i, marker, ha=ha, va='center', fontsize=16, fontweight='bold')

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([PREDICTOR_LABELS.get(p, p) for p in sorted_predictors])

    # Labels and title
    ax.set_xlabel('Standardized Coefficient (β)', fontsize=18)
    ax.set_title(f'{TASK_LABELS[target]}\n(R² = {model.rsquared:.3f})', fontweight='bold', fontsize=20)

    # Invert y-axis so largest is at top
    ax.invert_yaxis()

    return ax


# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 9))

for ax, target in zip(axes, targets):
    plot_coefficients(ax, results[target], target, TASK_COLORS[target], all_predictors)

# Add overall title
fig.suptitle('Standardized Regression Coefficients (No Interactions)',
             fontsize=24, fontweight='bold', y=1.02)

# Add significance legend
fig.text(0.5, -0.02, 'Significance: * p<0.05, ** p<0.01, *** p<0.001 | Predictors standardized (z-scored)',
         ha='center', fontsize=16, style='italic')

plt.tight_layout()

# Save figure
output_path = '/user_data/wenjiel2/abstraction/for_steve/figures'
os.makedirs(output_path, exist_ok=True)

plt.savefig(f'{output_path}/regression_coefficients_no_interactions.png', dpi=300, bbox_inches='tight')
print(f"\nSaved figure: {output_path}/regression_coefficients_no_interactions.png")

plt.close()


# ============================================================================
# VISUALIZATION 2: Cross-Validated Unique Variance Explained
# ============================================================================

def compute_cv_r2(X, y, n_folds=5):
    """Compute cross-validated R² score."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize within fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit and predict
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Compute R² on test set
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    return np.mean(r2_scores), np.std(r2_scores)


def compute_cv_unique_variance(df, predictors, target, n_folds=5):
    """
    Compute cross-validated unique variance explained for each predictor.

    For each predictor:
    CV Unique R² = CV_R²_full - CV_R²_reduced
    """
    X_full = df[predictors].values
    y = df[target].values

    # Full model CV R²
    r2_full_mean, r2_full_std = compute_cv_r2(X_full, y, n_folds)

    unique_r2 = {}
    unique_r2_std = {}

    for pred in predictors:
        # Reduced model (without this predictor)
        reduced_predictors = [p for p in predictors if p != pred]
        X_reduced = df[reduced_predictors].values

        r2_reduced_mean, r2_reduced_std = compute_cv_r2(X_reduced, y, n_folds)

        # Unique variance = difference
        unique_r2[pred] = r2_full_mean - r2_reduced_mean
        # Propagate uncertainty (approximate)
        unique_r2_std[pred] = np.sqrt(r2_full_std**2 + r2_reduced_std**2)

    return unique_r2, unique_r2_std, r2_full_mean, r2_full_std


def plot_unique_variance(ax, unique_r2, target, color, r2_full, predictors):
    """Plot unique variance explained as horizontal bar chart, sorted by magnitude."""
    # Convert to percentages
    values = np.array([unique_r2[p] * 100 for p in predictors])

    # Sort by value (largest to smallest)
    sorted_indices = np.argsort(values)[::-1]
    sorted_predictors = [predictors[i] for i in sorted_indices]
    sorted_values = values[sorted_indices]

    y_pos = np.arange(len(sorted_predictors))

    # Create horizontal bar plot
    ax.barh(y_pos, sorted_values, color=color, alpha=0.7,
            edgecolor='black', linewidth=1.0)

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([PREDICTOR_LABELS.get(p, p) for p in sorted_predictors])

    # Labels and title
    ax.set_xlabel('CV Unique Variance Explained (%)', fontsize=18)
    ax.set_title(f'{TASK_LABELS[target]}\n(CV R² = {r2_full:.3f})', fontweight='bold', fontsize=20)

    # Invert y-axis so largest is at top
    ax.invert_yaxis()

    # Set x-axis to start at 0 (or slightly negative for visibility)
    ax.set_xlim(left=-0.5)

    return ax


# Compute CV unique variance for each target
print("\nComputing cross-validated unique variance (5-fold CV)...")
cv_results = {}
for target in targets:
    print(f"  Processing {target}...")
    unique_r2, unique_r2_std, r2_full, r2_full_std = compute_cv_unique_variance(
        df_clean, all_predictors, target, n_folds=5
    )
    cv_results[target] = {
        'unique_r2': unique_r2,
        'unique_r2_std': unique_r2_std,
        'r2_full': r2_full,
        'r2_full_std': r2_full_std
    }

# Print CV results
print("\n" + "=" * 80)
print("CROSS-VALIDATED UNIQUE VARIANCE EXPLAINED")
print("=" * 80)

for target in targets:
    r2_full = cv_results[target]['r2_full']
    r2_std = cv_results[target]['r2_full_std']
    print(f"\n{TASK_LABELS[target]} (CV R² = {r2_full:.4f} ± {r2_std:.4f})")
    print("-" * 50)

    # Sort by unique R²
    sorted_preds = sorted(cv_results[target]['unique_r2'].items(), key=lambda x: x[1], reverse=True)

    for pred, unique_r2 in sorted_preds:
        std = cv_results[target]['unique_r2_std'][pred]
        pct = unique_r2 * 100
        pct_std = std * 100
        print(f"  {pred:25s}: {pct:6.3f}% ± {pct_std:.3f}%")

# Create combined figure for unique variance
fig, axes = plt.subplots(1, 3, figsize=(20, 9))

for ax, target in zip(axes, targets):
    plot_unique_variance(
        ax,
        cv_results[target]['unique_r2'],
        target,
        TASK_COLORS[target],
        cv_results[target]['r2_full'],
        all_predictors
    )

# Add overall title
fig.suptitle('Cross-Validated Unique Variance Explained (No Interactions)',
             fontsize=24, fontweight='bold', y=1.02)

# Add legend
fig.text(0.5, -0.02, '5-Fold Cross-Validation | Predictors standardized within each fold',
         ha='center', fontsize=16, style='italic')

plt.tight_layout()

plt.savefig(f'{output_path}/unique_variance_no_interactions.png', dpi=300, bbox_inches='tight')
print(f"\nSaved figure: {output_path}/unique_variance_no_interactions.png")

plt.close()
print("\nDone!")
