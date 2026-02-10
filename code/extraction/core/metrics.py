"""
Metric computation functions for each task
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Optional, Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

import traceback
from typing import Union, Tuple, Optional
from scipy.stats import spearmanr

# Then REPLACE the compute_task_metric method with this:

# def compute_task_metric(
#     self, 
#     task_name: str, 
#     task_df: pd.DataFrame, 
#     model_uid: str
# ) -> Union[float, Tuple[float, float]]:
#     """
#     Compute metric for a task.
    
#     For identity and geometry tasks, returns (correlation, p_value).
#     For relation tasks, returns correlation only.
    
#     Args:
#         task_name: Name of the task
#         task_df: DataFrame with model distances and human distances
#         model_uid: Model identifier
    
#     Returns:
#         For identity/geometry: (correlation, p_value) tuple
#         For relation: correlation float
#     """
    
#     if task_name == 'identity':
#         return self._compute_identity_metric(task_df, model_uid)
#     elif task_name == 'geometry':
#         return self._compute_geometry_metric(task_df, model_uid)
#     elif task_name in ['relation', 'relation_patternonly']:
#         return self._compute_relation_metric(task_df, model_uid)
#     else:
#         raise ValueError(f"Unknown task: {task_name}")


# ADD this NEW helper method for identity task:

# def _compute_identity_metric(
#     self, 
#     task_df: pd.DataFrame, 
#     model_uid: str
# ) -> Tuple[float, float]:
#     """
#     Compute semantic distance effect (Spearman correlation).
#     Returns both correlation and p-value.
#     """
#     # Column names - adjust these if your CSV uses different names
#     human_col = 'Human'  # or 'human_distance', 'semantic_distance'
#     model_col = f'{model_uid}_distance'
    
#     if human_col not in task_df.columns or model_col not in task_df.columns:
#         print(f"      Warning: Required columns not found for identity task")
#         print(f"      Available columns: {list(task_df.columns)}")
#         return None, None
    
#     # Remove NaN values
#     mask = ~(task_df[human_col].isna() | task_df[model_col].isna())
#     human_distances = task_df.loc[mask, human_col].values
#     model_distances = task_df.loc[mask, model_col].values
    
#     if len(human_distances) < 3:
#         print(f"      Warning: Insufficient data points for identity task")
#         return None, None
    
#     # Compute Spearman correlation and p-value
#     correlation, p_value = spearmanr(human_distances, model_distances)
    
#     return correlation, p_value


# # ADD this NEW helper method for geometry task:

# def _compute_geometry_metric(
#     self, 
#     task_df: pd.DataFrame, 
#     model_uid: str
# ) -> Tuple[float, float]:
#     """
#     Compute regularity/geometry effect (Spearman correlation).
#     Returns both correlation and p-value.
#     """
#     # Column names - adjust these if your CSV uses different names
#     symbolic_col = 'Symbolic'  # or 'symbolic', 'regularity', 'Regularity'
#     model_col = f'{model_uid}_distance'
    
#     if symbolic_col not in task_df.columns or model_col not in task_df.columns:
#         print(f"      Warning: Required columns not found for geometry task")
#         print(f"      Available columns: {list(task_df.columns)}")
#         return None, None
    
#     # Remove NaN values
#     mask = ~(task_df[symbolic_col].isna() | task_df[model_col].isna())
#     symbolic_scores = task_df.loc[mask, symbolic_col].values
#     model_distances = task_df.loc[mask, model_col].values
    
#     if len(symbolic_scores) < 3:
#         print(f"      Warning: Insufficient data points for geometry task")
#         return None, None
    
#     # Compute Spearman correlation and p-value
#     correlation, p_value = spearmanr(symbolic_scores, model_distances)
    
#     return correlation, p_value
    
def compute_sem_dist_effect(distances_df: pd.DataFrame, model_uid: str) -> float:
    """
    Compute semantic distance effect for identity task.
    Correlation between model distances and semantic similarity.
    
    Args:
        distances_df: DataFrame with distances and sem_sim column
        model_uid: Model identifier
    
    Returns:
        Spearman correlation coefficient
    """
    col_name = f"{model_uid}_distance"
    
    # Check if required columns exist
    if col_name not in distances_df.columns or 'sem_sim' not in distances_df.columns:
        raise ValueError(f"Required columns not found: {col_name} or sem_sim")
    
    # Filter out NaN values
    valid_data = distances_df[[col_name, 'sem_sim']].dropna()
    if len(valid_data) < 2:
        return np.nan

    corr, p_value = spearmanr(valid_data[col_name], -valid_data['sem_sim'])
    return float(corr), float(p_value)

def compute_diff_symbolic_r(distances_df: pd.DataFrame, model_uid: str) -> float:
    """
    Compute diff_symbolic_r for geometry task.
    Correlation between model distances and symbolic regularity difference.
    
    Args:
        distances_df: DataFrame with distances and diff_symbolic column
        model_uid: Model identifier
    
    Returns:
        Spearman correlation coefficient
    """
    col_name = f"{model_uid}_distance"
    
    # Check if required columns exist
    if col_name not in distances_df.columns or 'diff_symbolic' not in distances_df.columns:
        raise ValueError(f"Required columns not found: {col_name} or diff_symbolic")
    
    # For geometry, group by Sample,Correct,Incorrect,Condition and take mean
    if 'Condition' in distances_df.columns:
        grouped = distances_df.groupby(['Sample', 'Correct', 'Incorrect', 'Condition']).mean().reset_index()
    else:
        grouped = distances_df
    
    # Filter out NaN values
    valid_data = grouped[[col_name, 'diff_symbolic']].dropna()
    if len(valid_data) < 2:
        return np.nan

    # Compute correlation
    corr, p_value = spearmanr(valid_data[col_name], valid_data['diff_symbolic'])
    
    return float(corr), float(p_value)

def compute_relational_bias(distances_df: pd.DataFrame, model_uid: str) -> float:
    """
    Compute relational bias for relation task.
    Proportion of trials where model chooses the relationally correct answer.
    
    Args:
        distances_df: DataFrame with distances
        model_uid: Model identifier
    
    Returns:
        Relational bias score (0 to 1)
    """
    col_name = f"{model_uid}_distance"
    
    # Check if required column exists
    if col_name not in distances_df.columns:
        raise ValueError(f"Required column not found: {col_name}")
    
    # Filter out NaN values
    valid_data = distances_df[col_name].dropna()
    if len(valid_data) == 0:
        return np.nan

    # Relational bias: proportion where distance > 0 (correct choice is more similar)
    accuracy = (valid_data > 0).mean()
    
    return float(accuracy)

def compute_reference_comparisons(
    distances_df: pd.DataFrame, 
    model_uid: str,
    reference_groups: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Compute correlations with reference groups (human/animal data).
    
    Args:
        distances_df: DataFrame with model and reference distances
        model_uid: Model identifier
        reference_groups: List of (name, column_name) tuples for reference groups
    
    Returns:
        Dictionary of correlation values for each reference group
    """
    model_col = f"{model_uid}_distance"
    comparisons = {}
    
    for ref_name, ref_col in reference_groups:
        try:
            if model_col not in distances_df.columns or ref_col not in distances_df.columns:
                comparisons[ref_name] = np.nan
                continue
            
            # Filter out NaN values from both columns
            valid_data = distances_df[[model_col, ref_col]].dropna()

            if len(valid_data) < 2:
                comparisons[ref_name] = np.nan
                continue

            # Compute Spearman correlation
            corr, p_value = spearmanr(valid_data[model_col], valid_data[ref_col])
            comparisons[ref_name] = float(corr)
        except Exception as e:
            print(f"Error computing correlation for {ref_name} and {ref_col}: {e}")
            traceback.print_exc()

    return comparisons

# def compute_triplet_distances(
#     embeddings: np.ndarray,
#     sample_indices: List[int],
#     correct_indices: List[int],
#     incorrect_indices: List[int]
# ) -> np.ndarray:
#     """
#     Fixed version matching the old script exactly.
#     """
#     distances = []
    
#     for sample_idx, correct_idx, incorrect_idx in zip(
#         sample_indices, correct_indices, incorrect_indices
#     ):
#         sample_emb = embeddings[sample_idx].reshape(1, -1)
#         correct_emb = embeddings[correct_idx].reshape(1, -1)
#         incorrect_emb = embeddings[incorrect_idx].reshape(1, -1)
        
#         # Use cosine_distances directly, no epsilon!
#         dist_correct = cosine_distances(sample_emb, correct_emb)[0, 0]
#         dist_incorrect = cosine_distances(sample_emb, incorrect_emb)[0, 0]
        
#         distance_diff = dist_incorrect - dist_correct
#         distances.append(distance_diff)
    
#     return np.array(distances)

def compute_triplet_distances(
    embeddings: np.ndarray,
    sample_indices: List[int],
    correct_indices: List[int],
    incorrect_indices: List[int],
    validate: bool = True,
    allow_degenerate: bool = True  # NEW PARAMETER
) -> np.ndarray:
    """
    Compute distances for triplets of images.
    
    Args:
        embeddings: Array of embeddings [n_images, n_features]
        sample_indices: Indices of sample images
        correct_indices: Indices of correct choice images
        incorrect_indices: Indices of incorrect choice images
        validate: Whether to validate embeddings
        allow_degenerate: If True, compute distances even for near-zero embeddings
    
    Returns:
        Array of distance differences
    """
    
    # Validate embeddings
    if validate:
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = np.mean(embedding_norms)
        
        if mean_norm < 1e-10:
            if allow_degenerate:
                print(f"    WARNING: Near-zero embeddings (mean norm={mean_norm:.2e})")
                print(f"    This is expected for random/untrained models - proceeding with distance computation")
            else:
                warnings.warn(f"Degenerate embeddings (mean norm={mean_norm:.2e}). Returning NaN.")
                return np.full(len(sample_indices), np.nan)
    
    distances = []
    
    for sample_idx, correct_idx, incorrect_idx in zip(
        sample_indices, correct_indices, incorrect_indices
    ):
        sample_emb = embeddings[sample_idx].reshape(1, -1)
        correct_emb = embeddings[correct_idx].reshape(1, -1)
        incorrect_emb = embeddings[incorrect_idx].reshape(1, -1)
        
        # Add small epsilon to avoid numerical issues with zero vectors
        epsilon = 1e-8
        sample_emb = sample_emb + epsilon
        correct_emb = correct_emb + epsilon
        incorrect_emb = incorrect_emb + epsilon
        
        # Use cosine similarity (more stable than cosine_distances)
        sim_correct = cosine_similarity(sample_emb, correct_emb)[0, 0]
        sim_incorrect = cosine_similarity(sample_emb, incorrect_emb)[0, 0]
        
        # Convert to distance
        dist_correct = 1 - sim_correct
        dist_incorrect = 1 - sim_incorrect
        
        distance_diff = dist_incorrect - dist_correct
        distances.append(distance_diff)
    
    return np.array(distances)

class MetricsComputer:
    """Unified metrics computer for all tasks"""
    
    def __init__(self, config):
        self.config = config
        self.metric_functions = {
            'sem_dist_effect': compute_sem_dist_effect,
            'diff_symbolic_r': compute_diff_symbolic_r,
            'relational_bias': compute_relational_bias,
            'relational_bias_patternonly': compute_relational_bias
        }
    
    def compute_task_metric(
        self, 
        task_name: str,
        distances_df: pd.DataFrame,
        model_uid: str
    ) -> float:
        """
        Compute the metric for a specific task.
        
        Args:
            task_name: Name of the task
            distances_df: DataFrame with distance data
            model_uid: Model identifier
        
        Returns:
            Computed metric value
        """
        task_config = self.config.TASKS[task_name]
        metric_fn = self.metric_functions[task_config.metric_name]

        print(f"Computing metric '{task_config.metric_name}' for task '{task_name}' using model '{model_uid}'")
        
        return metric_fn(distances_df, model_uid)
    
    def compute_reference_correlations(
        self,
        task_name: str,
        distances_df: pd.DataFrame,
        model_uid: str
    ) -> Dict[str, float]:
        """
        Compute correlations with reference groups for a task.
        
        Args:
            task_name: Name of the task
            distances_df: DataFrame with distance data
            model_uid: Model identifier
        
        Returns:
            Dictionary of correlations with reference groups
        """
        task_config = self.config.TASKS[task_name]

        reference_tuples = [
            (ref.name, ref.column_name) 
            for ref in task_config.reference_groups
        ]

        return compute_reference_comparisons(
            distances_df, 
            model_uid,
            reference_tuples
        )