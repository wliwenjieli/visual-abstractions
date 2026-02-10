"""
Main processing pipeline for abstraction analysis
"""

import sys

# CRITICAL FIX: Compatibility patch for torchvision and robustness library
# Must be before any other imports that might load torchvision models
try:
    from torchvision.models.utils import load_state_dict_from_url
except (ImportError, ModuleNotFoundError):
    # For torchvision >= 0.13, the function moved to torch.hub
    try:
        from torch.hub import load_state_dict_from_url
        from types import ModuleType
        
        # Create the missing torchvision.models.utils module
        utils_module = ModuleType('torchvision.models.utils')
        utils_module.load_state_dict_from_url = load_state_dict_from_url
        sys.modules['torchvision.models.utils'] = utils_module
        
        print("Applied torchvision compatibility patch for robustness library")
    except ImportError:
        print("WARNING: Could not apply torchvision compatibility patch")
        pass

import os
import glob
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import traceback

from config.settings import Config
from core.cache_manager import CacheManager
from core.error_tracker import ErrorTracker
from core.data_io import DataIO, StatusTracker
from core.model_processor import ModelProcessor
from core.model_utils import (
    get_extraction_layer,
    should_skip_model,
    cleanup_model,
    emergency_cleanup
)
from core.metrics import MetricsComputer, compute_triplet_distances
from core.geometry_mapping import map_geometry_name


def count_model_parameters(model) -> Optional[int]:
    """
    Count total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters, or None if counting fails
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    except Exception as e:
        print(f"    Warning: Could not count parameters: {e}")
        return None


class AbstractionPipeline:
    """Main pipeline for processing models and computing metrics"""
    
    def __init__(self, config: Config = None, force_recompute: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object (creates default if None)
            force_recompute: Force recomputation even if results exist
        """
        self.config = config or Config()
        self.force_recompute = force_recompute

        # Initialize components
        self.cache_manager = CacheManager(
            max_size_gb=self.config.MAX_CACHE_SIZE_GB,
            cleanup_threshold_gb=self.config.CACHE_CLEANUP_THRESHOLD_GB
        )
        
        self.error_tracker = ErrorTracker(
            f"{self.config.RESULTS_DIR}/logs/errors.json"
        )
        
        self.status_tracker = StatusTracker(
            f"{self.config.RESULTS_DIR}/logs/processing_status.json"
        )
        
        self.metrics_computer = MetricsComputer(self.config)
        
        self.model_processor = ModelProcessor(self.config)
        
        # Load metadata
        self.load_metadata()
        
        # Prepare results directories
        self.prepare_directories()
    
    def load_metadata(self):
        """Load and clean model metadata"""
        print("Loading model metadata...")
        
        if not Path(self.config.MODEL_METADATA_PATH).exists():
            raise FileNotFoundError(f"Metadata file not found: {self.config.MODEL_METADATA_PATH}")
        
        self.metadata_df = pd.read_csv(self.config.MODEL_METADATA_PATH)
        
        print(f"  Loaded {len(self.metadata_df)} models")
    
    def prepare_directories(self):
        """Create necessary directories"""
        dirs = [
            f"{self.config.RESULTS_DIR}/distances",
            f"{self.config.RESULTS_DIR}/metrics",
            f"{self.config.RESULTS_DIR}/logs",
            f"{self.config.RESULTS_DIR}/backups"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def should_process(self, model_uid: str, task: str) -> bool:
        """
        Check if a model-task combination needs processing.
        
        Args:
            model_uid: Model identifier
            task: Task name
        
        Returns:
            True if processing is needed
        """
        if self.force_recompute:
            return True
        
        # Check if already completed in status tracker
        if self.status_tracker.is_completed(model_uid, task):
            return False
        
        # Check if had errors before (might want to retry)
        if self.error_tracker.has_error(model_uid, task):
            print(f"  Note: {model_uid}-{task} had previous errors, retrying...")
            return True
        
        return True
    
    def load_task_stimuli(self, task_name: str) -> Tuple[List[str], pd.DataFrame]:
        """
        Load stimuli for a task.
        
        Args:
            task_name: Name of the task
        
        Returns:
            Tuple of (image_paths, task_info_df)
        """
        task_config = self.config.TASKS[task_name]
        
        # Map task names to CSV file names
        task_to_csv = {
            'identity': 'identity_distance.csv',
            'geometry': 'geometry_distance.csv', 
            'relation': 'relation_distance.csv',
            'relation_patternonly': 'patterns_only_distance.csv'
        }
        
        # Load existing distance CSV if it exists
        csv_filename = task_to_csv.get(task_name)
        csv_path = f"{self.config.RESULTS_DIR}/{csv_filename}"
        
        if not Path(csv_path).exists():
            print(f"    Warning: {csv_path} not found - skipping task")
            return [], pd.DataFrame()
        
        # Load the CSV with task structure
        task_df = pd.read_csv(csv_path)

        # Handle both task name formats
        base_task = task_name.replace('_patternonly', '')
        stimulus_dir = f"{self.config.STIMULUS_DIR}/{base_task}"
        
        # NEW: Use glob.glob to get all images (matching old script)
        all_images = glob.glob(os.path.join(stimulus_dir, '*'))
        
        # NEW: Create index mapping based on basename (matching old script)
        image_name_to_idx = {os.path.basename(img): idx 
                            for idx, img in enumerate(all_images)}
        
        # Apply geometry mapping if this is a geometry task
        if 'geometry' in task_name.lower():
            print(f"    Applying geometry name mapping...")
            
            # Map names based on column context
            for col in ['Sample', 'Correct', 'Incorrect']:
                if col in task_df.columns:
                    task_df[f'{col}_original'] = task_df[col]
                    task_df[col] = task_df[col].apply(
                        lambda x: map_geometry_name(str(x), column=col) if pd.notna(x) else x
                    )
        
        # ALWAYS recreate index mapping based on current glob order
        # FIX: Previously only created indices if not present, but CSV indices
        # may have been created from a DIFFERENT glob order, causing embeddings[idx]
        # to retrieve wrong image embeddings.
        # The indices are only used during distance computation to map image names
        # to embedding positions - they don't affect stored parquet data which
        # contains computed distance values aligned with task_df rows.
        print(f"    Creating index mapping based on current glob order...")

        task_df['sample_idx'] = task_df['Sample'].apply(
            lambda x: image_name_to_idx.get(x, -1) if pd.notna(x) else -1
        )
        task_df['correct_idx'] = task_df['Correct'].apply(
            lambda x: image_name_to_idx.get(x, -1) if pd.notna(x) else -1
        )
        task_df['incorrect_idx'] = task_df['Incorrect'].apply(
            lambda x: image_name_to_idx.get(x, -1) if pd.notna(x) else -1
        )

        # Check for missing indices
        missing = (task_df[['sample_idx', 'correct_idx', 'incorrect_idx']] == -1).any(axis=1)
        if missing.any():
            print(f"    Warning: {missing.sum()} trials have missing images")
            # Show examples of missing images
            missing_trials = task_df[missing].head(2)
            for _, row in missing_trials.iterrows():
                print(f"      Sample: {row['Sample']} (idx: {row['sample_idx']})")
                print(f"      Correct: {row['Correct']} (idx: {row['correct_idx']})")
                print(f"      Incorrect: {row['Incorrect']} (idx: {row['incorrect_idx']})")
        
        print(f"    Found {len(all_images)} unique images")
        
        if len(all_images) == 0:
            print(f"    Warning: No images found in {stimulus_dir}")
            return [], pd.DataFrame()
        
        # CHANGED: Return unsorted image list (matching old script)
        return all_images, task_df  # Don't sort!

    # def load_task_stimuli(self, task_name: str) -> Tuple[List[str], pd.DataFrame]:
    #     """
    #     Load stimuli for a task.
        
    #     Args:
    #         task_name: Name of the task
        
    #     Returns:
    #         Tuple of (image_paths, task_info_df)
    #     """
    #     task_config = self.config.TASKS[task_name]
        
    #     # Map task names to CSV file names
    #     task_to_csv = {
    #         'identity': 'identity_distance.csv',
    #         'geometry': 'geometry_distance.csv', 
    #         'relation': 'relation_distance.csv',
    #         'relation_patternonly': 'patterns_only_distance.csv'  # Use the pattern-only CSV directly
    #     }
        
    #     # Load existing distance CSV if it exists
    #     csv_filename = task_to_csv.get(task_name)
    #     csv_path = f"{self.config.RESULTS_DIR}/{csv_filename}"
        
    #     if not Path(csv_path).exists():
    #         print(f"    Warning: {csv_path} not found - skipping task")
    #         return [], pd.DataFrame()
        
    #     # Load the CSV with task structure
    #     task_df = pd.read_csv(csv_path)

    #     # Get unique image names from the task
    #     image_names = set()
    #     for col in ['Sample', 'Correct', 'Incorrect']:
    #         if col in task_df.columns:
    #             image_names.update(task_df[col].unique())
        
    #     # Apply geometry mapping if this is a geometry task
    #     if 'geometry' in task_name.lower():
    #         print(f"    Applying geometry name mapping...")
            
    #         # Map names based on column context
    #         for col in ['Sample', 'Correct', 'Incorrect']:
    #             if col in task_df.columns:
    #                 task_df[f'{col}_original'] = task_df[col]
    #                 task_df[col] = task_df[col].apply(
    #                     lambda x: map_geometry_name(str(x), column=col) if pd.notna(x) else x
    #                 )
            
    #         # Update image_names with mapped values
    #         image_names = set()
    #         for col in ['Sample', 'Correct', 'Incorrect']:
    #             if col in task_df.columns:
    #                 image_names.update(task_df[col].unique())
        
    #     # Build image paths
    #     image_paths = []
    #     image_name_to_idx = {}
        
    #     # Handle both task name formats
    #     base_task = task_name.replace('_patternonly', '')
    #     stimulus_dir = f"{self.config.STIMULUS_DIR}/{base_task}"
        
    #     # First, try to find images without adding extension (they might already have it)
    #     for img_name in image_names:
    #         if pd.isna(img_name):
    #             continue
                
    #         img_name = str(img_name)
    #         img_path = f"{stimulus_dir}/{img_name}"
            
    #         # Check if the path exists as-is
    #         if Path(img_path).exists():
    #             if img_path not in image_paths:
    #                 idx = len(image_paths)
    #                 image_paths.append(img_path)
    #                 image_name_to_idx[img_name] = idx
    #         else:
    #             # Try adding common extensions if not found
    #             found = False
    #             for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']:
    #                 img_path_ext = f"{stimulus_dir}/{img_name}{ext}"
    #                 if Path(img_path_ext).exists():
    #                     if img_path_ext not in image_paths:
    #                         idx = len(image_paths)
    #                         image_paths.append(img_path_ext)
    #                         image_name_to_idx[img_name] = idx
    #                     found = True
    #                     break
                
    #             if not found:
    #                 # Try removing extension if the name already has one
    #                 base_name = Path(img_name).stem
    #                 img_path_base = f"{stimulus_dir}/{base_name}"
    #                 if Path(img_path_base).exists():
    #                     if img_path_base not in image_paths:
    #                         idx = len(image_paths)
    #                         image_paths.append(img_path_base)
    #                         image_name_to_idx[img_name] = idx
    #                 else:
    #                     # Try with extensions on the base name
    #                     for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']:
    #                         img_path_ext = f"{stimulus_dir}/{base_name}{ext}"
    #                         if Path(img_path_ext).exists():
    #                             if img_path_ext not in image_paths:
    #                                 idx = len(image_paths)
    #                                 image_paths.append(img_path_ext)
    #                                 image_name_to_idx[img_name] = idx
    #                             break
        
    #     print(f"    Found {len(image_paths)} unique images")
        
    #     if len(image_paths) == 0:
    #         print(f"    Warning: No images found in {stimulus_dir}")
    #         print(f"    Sample image names from CSV: {list(image_names)[:5]}")
    #         # Check what files actually exist in the directory
    #         if Path(stimulus_dir).exists():
    #             actual_files = list(Path(stimulus_dir).glob("*"))
    #             print(f"    Files in directory: {[f.name for f in actual_files[:5]]}")
    #         return [], pd.DataFrame()
        
    #     # Create index mapping if not present
    #     if 'sample_idx' not in task_df.columns:
    #         print(f"    Creating index mapping...")
            
    #         # Add index columns
    #         task_df['sample_idx'] = task_df['Sample'].map(
    #             lambda x: image_name_to_idx.get(str(x) if pd.notna(x) else '', -1)
    #         )
    #         task_df['correct_idx'] = task_df['Correct'].map(
    #             lambda x: image_name_to_idx.get(str(x) if pd.notna(x) else '', -1)
    #         )
    #         task_df['incorrect_idx'] = task_df['Incorrect'].map(
    #             lambda x: image_name_to_idx.get(str(x) if pd.notna(x) else '', -1)
    #         )
            
    #         # Check for missing indices
    #         missing = (task_df[['sample_idx', 'correct_idx', 'incorrect_idx']] == -1).any(axis=1)
    #         if missing.any():
    #             print(f"    Warning: {missing.sum()} trials have missing images")
    #             # Show examples of missing images
    #             missing_trials = task_df[missing].head(2)
    #             for _, row in missing_trials.iterrows():
    #                 print(f"      Sample: {row['Sample']}, Correct: {row['Correct']}, Incorrect: {row['Incorrect']}")
                
    #             # # For geometry, this is expected - we only keep trials with all images found
    #             # # For other tasks, this might be an error
    #             # if 'geometry' in task_name.lower():
    #             #     print(f"    Keeping only trials with all images found")
    #             #     task_df = task_df[~missing].reset_index(drop=True)
    #             # else:
    #             #     task_df = task_df[~missing]
        
    #         # save updated task structure back to CSV
    #         task_df.to_csv(csv_path, index=False)
            
    #     return sorted(image_paths), task_df
    
    def create_task_structure(self, task_name: str, image_paths: List[str]) -> pd.DataFrame:
        """
        Create basic task structure DataFrame.
        This should be customized based on your actual task CSVs.
        """
        # This is a placeholder - you'll need to adapt this based on your actual data
        print(f"  Warning: Creating new task structure for {task_name}")
        
        # Create image name to index mapping
        img_names = [Path(p).name for p in image_paths]
        img_to_idx = {name: idx for idx, name in enumerate(img_names)}
        
        # You'll need to load the actual triplet information from somewhere
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def extract_embeddings(
        self, 
        model: Any,
        preprocess: Any,
        image_paths: List[str],
        layer_idx: int,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract embeddings from a model for given images.
        
        Args:
            model: Loaded model
            preprocess: Preprocessing function
            image_paths: List of image paths
            layer_idx: Layer index to extract from
            batch_size: Batch size for processing
        
        Returns:
            Array of embeddings [n_images, n_features]
        """
        try:
            features = self.model_processor.extract_features(
            model, preprocess, image_paths, layer_idx, batch_size
            )
            # are all features identical
            if np.all(features == features[0]):
                raise ValueError("All extracted features are identical")
            # check if there is nan in features
            if np.isnan(features).any():
                raise ValueError("NaN values found in extracted features")

        except Exception as e:
            print(f"      Error during feature extraction: {e}")
            print(image_paths)
            emergency_cleanup()
            raise e
        return features
    
    def save_task_metric(
        self, 
        model_uid: str, 
        task_name: str, 
        metric_value: float,
        p_value: Optional[float] = None,  # NEW parameter
        reference_correlations: dict = None,
        layer_idx: int = None
    ):
        """
        Save computed metric to consolidated metrics file.
        UPDATED to save p-values for identity and geometry tasks.
        """
        metrics_file = Path(self.config.RESULTS_DIR) / "metrics" / "all_metrics.csv"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Map task names to metric columns
        metric_columns = {
            'identity': 'sem_dist_effect',
            'geometry': 'diff_symbolic_r', 
            'relation': 'relational_bias',
            'relation_patternonly': 'relational_bias_patternonly'
        }
        
        # NEW: Map task names to p-value columns
        p_value_columns = {
            'identity': 'sem_dist_effect_p',
            'geometry': 'diff_symbolic_p',
        }
        
        metric_col = metric_columns.get(task_name)
        if not metric_col:
            print(f"      Warning: Unknown task name {task_name}")
            return
        
        # Load existing metrics or create new dataframe
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
        else:
            df = pd.DataFrame()
        
        # Ensure model exists in dataframe
        if df.empty or 'model_uid' not in df.columns or model_uid not in df['model_uid'].values:
            print(model_uid, "does not exist in metrics dataframe, adding new row")
            new_row = pd.DataFrame({'model_uid': [model_uid]})
            df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
        else:
            print("model already exists in metrics dataframe")
        
        # Update the metric value (correlation)
        df.loc[df['model_uid'] == model_uid, metric_col] = metric_value
        
        # NEW: Update the p-value if applicable
        if task_name in p_value_columns and p_value is not None:
            p_col = p_value_columns[task_name]
            df.loc[df['model_uid'] == model_uid, p_col] = p_value
            print(f"      ✓ Saved {metric_col} = {metric_value:.4f}, p = {p_value:.4e}")
        else:
            print(f"      ✓ Saved {metric_col} = {metric_value:.4f}")
        
        # Add extraction layer info
        if layer_idx is not None:
            df.loc[df['model_uid'] == model_uid, f'{metric_col}_layer'] = layer_idx
        
        # Save reference correlations if provided
        if reference_correlations:
            print("saving reference correlations")
            self.save_reference_correlations(model_uid, task_name, reference_correlations)
        
        # Save the updated metrics
        df.to_csv(metrics_file, index=False)
    
    def save_reference_correlations(self, model_uid: str, task_name: str, correlations: dict):
        """Save reference group correlations."""
        ref_file = Path(self.config.RESULTS_DIR) / "metrics" / "reference_comparisons.csv"
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        
        ref_rows = []
        for ref_group, correlation in correlations.items():
            ref_rows.append({
                'model_uid': model_uid,
                'task': task_name,
                'reference_group': ref_group,
                'correlation': correlation
            })
        
        ref_df = pd.DataFrame(ref_rows)
        
        # Check if file exists and is not empty
        if ref_file.exists() and ref_file.stat().st_size > 0:
            try:
                existing_ref = pd.read_csv(ref_file)
                # Remove old entries for this model/task combo
                existing_ref = existing_ref[~((existing_ref['model_uid'] == model_uid) & 
                                            (existing_ref['task'] == task_name))]
                ref_df = pd.concat([existing_ref, ref_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                # File exists but is empty or corrupted
                print(f"Warning: {ref_file} is empty or corrupted. Creating new file.")
            except Exception as e:
                print(f"Warning: Error reading {ref_file}: {e}. Creating new file.")
        
        ref_df.to_csv(ref_file, index=False)
        print(f"Saved {len(ref_rows)} reference correlations for {model_uid}/{task_name}")

    def save_model_params(self, model_uid: str, total_params: int):
        """
        Save model parameter count to consolidated metrics file.

        Args:
            model_uid: Model identifier
            total_params: Total number of parameters in the model
        """
        metrics_file = Path(self.config.RESULTS_DIR) / "metrics" / "all_metrics.csv"

        if not metrics_file.exists():
            print(f"    Warning: Metrics file does not exist, cannot save params")
            return

        try:
            df = pd.read_csv(metrics_file)

            # Check if model exists in dataframe
            if 'model_uid' not in df.columns or model_uid not in df['model_uid'].values:
                print(f"    Warning: {model_uid} not in metrics file, cannot save params")
                return

            # Check if params already exist and are valid
            existing_params = df.loc[df['model_uid'] == model_uid, 'model_params'].values
            if len(existing_params) > 0 and pd.notna(existing_params[0]) and existing_params[0] > 0:
                # Already has valid params, skip
                return

            # Update model_params and model_params_log10
            df.loc[df['model_uid'] == model_uid, 'model_params'] = total_params
            df.loc[df['model_uid'] == model_uid, 'model_params_log10'] = np.log10(total_params) if total_params > 0 else np.nan
            df.loc[df['model_uid'] == model_uid, 'model_params_millions'] = total_params / 1e6

            # Save
            df.to_csv(metrics_file, index=False)
            print(f"    ✓ Saved model_params = {total_params:,} ({total_params/1e6:.1f}M)")

        except Exception as e:
            print(f"    Warning: Could not save model params: {e}")

    def process_task(
        self,
        model: Any,
        preprocess: Any,
        model_uid: str,
        task_name: str,
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        Process a single task for a model.
        UPDATED to handle p-values for identity and geometry tasks.
        """
        print(f"    Processing {task_name}...")
        
        # Load stimuli
        image_paths, task_df = self.load_task_stimuli(task_name)
        
        if len(task_df) == 0:
            print(f"      Warning: No task structure for {task_name}")
            return {'status': 'skipped', 'reason': 'no_task_structure'}
        
        # Check if this is a problematic model
        if 'dinov2' in model_uid.lower() and '_ld' in model_uid:
            print(f"      Using special extraction for {model_uid}")
            from core.dinov2_fix import extract_features_with_fallback
            embeddings = extract_features_with_fallback(
                model, preprocess, image_paths, model_uid,
                layer_idx, self.config.BATCH_SIZE, self.config.DEVICE
            )
        else:
            # Extract embeddings
            embeddings = self.extract_embeddings(
                model, preprocess, image_paths, layer_idx
            )

        # Compute distances
        if 'sample_idx' in task_df.columns:
            distances = compute_triplet_distances(
                embeddings,
                task_df['sample_idx'].values,
                task_df['correct_idx'].values,
                task_df['incorrect_idx'].values
            )
            if len(distances) != len(task_df):
                print(f"      Warning: Distance length mismatch: {len(distances)} vs {len(task_df)}")
                valid_indices = task_df.index[:len(distances)]
                task_df = task_df.loc[valid_indices].copy()
            
            task_df[f"{model_uid}_distance"] = distances
        else:
            print(f"      Warning: No indices in task structure")
            return {'status': 'skipped', 'reason': 'no_indices'}
        
        # Save distances
        distance_file = f"{self.config.RESULTS_DIR}/distances/{task_name}.parquet"
        if Path(distance_file).exists():
            existing_df = pd.read_parquet(distance_file)
            if f"{model_uid}_distance" in existing_df.columns:
                existing_df[f"{model_uid}_distance"] = distances
            else:
                existing_df = pd.concat([existing_df, task_df[[f"{model_uid}_distance"]]], axis=1)
            DataIO.save_parquet(existing_df, distance_file)
        else:
            DataIO.save_parquet(task_df, distance_file)
        
        # Compute metric - UPDATED to handle p-values
        metric_result = self.metrics_computer.compute_task_metric(
            task_name, task_df, model_uid
        )
        
        # Parse metric result based on task type
        if task_name in ['identity', 'geometry']:
            # These tasks return (correlation, p_value)
            if metric_result is not None and isinstance(metric_result, tuple):
                metric_value, p_value = metric_result
            else:
                print("      Warning: Invalid metric result format", metric_result)
                metric_value, p_value = None, None
        else:
            # Relation tasks return only correlation
            metric_value = metric_result
            p_value = None
        
        # Compute reference correlations
        ref_correlations = self.metrics_computer.compute_reference_correlations(
            task_name, task_df, model_uid
        )
        
        # Save the metric with p-value
        if metric_value is not None:
            self.save_task_metric(
                model_uid=model_uid,
                task_name=task_name,
                metric_value=metric_value,
                p_value=p_value,  # NEW: Pass p-value
                reference_correlations=ref_correlations,
                layer_idx=layer_idx
            )

        return {
            'status': 'completed',
            'metric_value': metric_value,
            'p_value': p_value,  # NEW: Include in return dict
            'reference_correlations': ref_correlations,
            'layer': layer_idx
        }
    
    def consolidate_with_metadata(self):
        """Consolidate metrics with original metadata."""
        print("\n" + "="*70)
        print("CONSOLIDATING METRICS WITH METADATA")
        print("="*70)
        
        metrics_file = Path(self.config.RESULTS_DIR) / "metrics" / "consolidated_metrics.csv"
        
        if not metrics_file.exists():
            print("No metrics file found - no new metrics were computed")
            return None
        
        # Load computed metrics
        metrics_df = pd.read_csv(metrics_file)
        print(f"Found metrics for {len(metrics_df)} models")
        
        # Load original metadata if exists
        if Path(self.config.MODEL_METADATA_PATH).exists():
            metadata_df = pd.read_csv(self.config.MODEL_METADATA_PATH)
            
            # Select essential columns
            essential_cols = [
                'model_uid', 'source', 'model_name', 'architecture', 
                'training_objective', 'dataset_name', 'dataset_size_log10',
                'model_params_log10', 'has_language_component', 'is_video_model',
                'is_quantized', 'architecture_detailed', 'training_paradigm',
                'model_size_category', 'is_modern', 'is_finetuned'
            ]
            
            keep_cols = [col for col in essential_cols if col in metadata_df.columns]
            metadata_df = metadata_df[keep_cols]
            
            # Merge with metrics
            final_df = metadata_df.merge(metrics_df, on='model_uid', how='outer')
        else:
            final_df = metrics_df
        
        # Save final consolidated file
        output_file = Path(self.config.RESULTS_DIR) / "model_metadata_with_metrics.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Final results saved to: {output_file}")
        print(f"  Total models: {len(final_df)}")
        
        # Print summary of metrics
        print("\nMetrics summary:")
        for col in ['sem_dist_effect', 'diff_symbolic_r', 'relational_bias', 'relational_bias_patternonly']:
            if col in final_df.columns:
                count = final_df[col].notna().sum()
                if count > 0:
                    mean_val = final_df[col].mean()
                    std_val = final_df[col].std()
                    print(f"  {col}: {count} models (mean={mean_val:.3f}, std={std_val:.3f})")
        
        return final_df
    
    def process_model(self, model_uid: str) -> Dict[str, Any]:
        """
        Process a single model across all tasks.
        
        Args:
            model_uid: Model identifier
        
        Returns:
            Dictionary with processing results
        """
        results = {'model_uid': model_uid, 'tasks': {}}
        
        # Check if should skip
        if should_skip_model(model_uid, self.config):
            print(f"  Skipping {model_uid}: in skip list")
            results['status'] = 'skipped'
            return results

        # skip if sem_dist_effect, diff_symbolic_r, relational_bias, relational_bias_patternonly 
        # already filled up for this model_uid in model_metadata
        tasks_to_process = []
        for task_name in ['identity', 'geometry', 'relation', 'relation_patternonly']:
            if self.should_process(model_uid, task_name):
                tasks_to_process.append(task_name)
        if len(tasks_to_process) == 0:
            print(f"  Skipping {model_uid}: all tasks completed")
            results['status'] = 'skipped'
            return results

        
        # Get metadata for this model
        model_meta = self.metadata_df[self.metadata_df['model_uid'] == model_uid]
        if len(model_meta) == 0:
            print(f"  Warning: No metadata for {model_uid}")
            model_meta = pd.Series({'model_uid': model_uid})
        else:
            model_meta = model_meta.iloc[0]
        
        # ========================================================================
        # CRITICAL FIX: PROACTIVE CACHE CLEANUP BEFORE MODEL LOADING
        # ========================================================================
        print(f"  Checking cache before loading model...")
        cache_summary = self.cache_manager.get_cache_summary()
        current_cache_size = cache_summary['total_size_gb']
        
        # Define aggressive threshold (70% of cleanup threshold)
        # This ensures we have enough space BEFORE torch.hub tries to download
        aggressive_threshold = self.config.CACHE_CLEANUP_THRESHOLD_GB * 0.7
        
        if current_cache_size > aggressive_threshold:
            print(f"  ⚠️  Cache size ({current_cache_size:.1f}GB) exceeds safe threshold ({aggressive_threshold:.1f}GB)")
            print(f"  Performing aggressive cleanup before loading {model_uid}...")
            
            # Clean to a very conservative target (20% of max cache size)
            # This gives plenty of room for the new model download
            target_size = self.config.MAX_CACHE_SIZE_GB * 0.2
            
            cleanup_result = self.cache_manager.cleanup_cache(
                force=True, 
                target_size_gb=target_size
            )
            
            if cleanup_result['cleaned']:
                print(f"  ✓ Freed {cleanup_result['cleaned_size_gb']:.2f} GB")
                print(f"  ✓ New cache size: {cleanup_result['final_size_gb']:.2f} GB")
            else:
                print(f"  ℹ️  Cleanup result: {cleanup_result.get('reason', 'Unknown')}")
        else:
            print(f"  ✓ Cache size OK ({current_cache_size:.1f}GB < {aggressive_threshold:.1f}GB)")
        
        # Additional safety check: verify we have enough disk space
        import shutil
        try:
            # Get the torch cache directory from the cache manager
            cache_dir = self.cache_manager.torch_cache.parent
            stat = shutil.disk_usage(str(cache_dir))
            available_gb = stat.free / (1024**3)
            
            # Require at least 10GB free space before attempting download
            if available_gb < 10:
                print(f"  ⚠️  WARNING: Only {available_gb:.1f}GB free disk space available!")
                print(f"  Attempting emergency cleanup...")
                
                # Emergency cleanup: clean to absolute minimum
                emergency_result = self.cache_manager.cleanup_cache(
                    force=True,
                    target_size_gb=5  # Keep only 5GB in cache
                )
                
                if emergency_result['cleaned']:
                    print(f"  ✓ Emergency cleanup freed {emergency_result['cleaned_size_gb']:.2f} GB")
                
                # Re-check disk space
                stat = shutil.disk_usage(str(cache_dir))
                available_gb = stat.free / (1024**3)
                
                if available_gb < 5:
                    error_msg = f"Insufficient disk space: only {available_gb:.1f}GB available. Need at least 5GB."
                    print(f"  ✗ {error_msg}")
                    self.error_tracker.log_error(model_uid, 'disk_space', Exception(error_msg), 'pre_loading')
                    results['status'] = 'error'
                    results['error'] = error_msg
                    return results
            else:
                print(f"  ✓ Disk space OK ({available_gb:.1f}GB available)")
        except Exception as e:
            print(f"  ⚠️  Could not check disk space: {e}")
        
        # ========================================================================
        # NOW PROCEED WITH MODEL LOADING
        # ========================================================================
        
        model = None
        try:
            # Load model
            print(f"  Loading model...")
            model, preprocess = self.model_processor.load_model(model_uid)

            # Count and save model parameters (if not already recorded)
            total_params = count_model_parameters(model)
            if total_params is not None:
                self.save_model_params(model_uid, total_params)

            # Determine extraction layer
            layer_idx = get_extraction_layer(model, model_meta, self.config)

            # Process each task
            for task_name in tasks_to_process:
                try:
                    task_results = self.process_task(
                        model, preprocess, model_uid, task_name, layer_idx
                    )
                    results['tasks'][task_name] = task_results
                    
                    # Mark as completed
                    if task_results['status'] == 'completed':
                        self.status_tracker.mark_completed(
                            model_uid, task_name, 
                            layer=layer_idx,
                            metric_value=task_results['metric_value']
                        )
                
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[-1]  # last traceback entry
                    print(f"    Error in {task_name}: {e} (File \"{tb.filename}\", line {tb.lineno})")

                    self.error_tracker.log_error(model_uid, task_name, e, 'processing')
                    results['tasks'][task_name] = {'status': 'error', 'error': str(e)}
                # task completed
                print(f"    Finished {task_name}")
            
            results['status'] = 'completed'
            print(f"  Finished processing {model_uid}")
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            self.error_tracker.log_error(model_uid, 'model_loading', e, 'loading')
            results['status'] = 'error'
            results['error'] = str(e)
            
        finally:
            # Clean up
            if model is not None:
                cleanup_model(model)
        
        return results
    
    def run(self, model_uids: Optional[List[str]] = None, test_mode: bool = False):
        """
        Run the pipeline.
        
        Args:
            model_uids: Optional list of specific models to process
            test_mode: If True, only process first 3 models
        """
        print("\n" + "="*70)
        print("ABSTRACTION ANALYSIS PIPELINE")
        print("="*70)
        
        # Show initial cache status
        cache_summary = self.cache_manager.get_cache_summary()
        print(f"\nInitial cache status:")
        print(f"  Total cache size: {cache_summary['total_size_gb']:.2f} GB")
        print(f"  Torch models: {cache_summary['torch_models']}")
        print(f"  HuggingFace models: {cache_summary['huggingface_models']}")
        
        # Show disk space
        try:
            import shutil
            cache_dir = self.cache_manager.torch_cache.parent
            stat = shutil.disk_usage(str(cache_dir))
            available_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            percent_used = (stat.used / stat.total) * 100
            
            print(f"\nDisk space at {cache_dir}:")
            print(f"  Total: {total_gb:.1f} GB")
            print(f"  Available: {available_gb:.1f} GB ({100-percent_used:.1f}% free)")
            
            if available_gb < 20:
                print(f"  ⚠️  WARNING: Low disk space!")
        except Exception as e:
            print(f"\n  Could not check disk space: {e}")
        
        # Initial cleanup if recommended
        if cache_summary['cleanup_recommended']:
            print("\n  Cache cleanup recommended, performing initial cleanup...")
            cleanup_result = self.cache_manager.cleanup_cache()
            if cleanup_result['cleaned']:
                print(f"  ✓ Freed {cleanup_result['cleaned_size_gb']:.2f} GB")
        
        # Get models to process
        if model_uids is None:
            model_uids = self.metadata_df['model_uid'].tolist()
        
        if test_mode:
            model_uids = model_uids[:3]
            print("\n⚠️  TEST MODE: Processing only first 3 models")
        
        print(f"\nProcessing {len(model_uids)} models")
        print("-" * 50)
        
        # Process models with enhanced monitoring
        failed_models = []
        skipped_models = []
        successful_models = []
        
        for i, model_uid in enumerate(model_uids):
            print(f"\n[{i+1}/{len(model_uids)}] {model_uid}")
            print(f"  Progress: {i}/{len(model_uids)} completed")
            
            try:
                results = self.process_model(model_uid)
                
                # Track results
                if results.get('status') == 'completed':
                    successful_models.append(model_uid)
                elif results.get('status') == 'skipped':
                    skipped_models.append(model_uid)
                elif results.get('status') == 'error':
                    failed_models.append(model_uid)
                
                # More aggressive periodic cache monitoring
                # Check cache every 5 models instead of 10
                if (i + 1) % 5 == 0:
                    print(f"\n  === Periodic cache check ({i+1}/{len(model_uids)} models processed) ===")
                    current_cache = self.cache_manager.get_cache_size_gb()['total']
                    print(f"  Current cache size: {current_cache:.1f} GB")
                    
                    # More aggressive threshold for periodic cleanup
                    if current_cache > self.config.CACHE_CLEANUP_THRESHOLD_GB * 0.6:
                        print(f"  Cache exceeds 60% threshold, performing cleanup...")
                        cleanup_result = self.cache_manager.cleanup_cache(
                            force=True,
                            target_size_gb=self.config.MAX_CACHE_SIZE_GB * 0.3
                        )
                        if cleanup_result['cleaned']:
                            print(f"  ✓ Freed {cleanup_result['cleaned_size_gb']:.2f} GB")
                    
                    # Also check disk space
                    try:
                        cache_dir = self.cache_manager.torch_cache.parent
                        stat = shutil.disk_usage(str(cache_dir))
                        available_gb = stat.free / (1024**3)
                        print(f"  Available disk space: {available_gb:.1f} GB")
                        
                        if available_gb < 15:
                            print(f"  ⚠️  Low disk space - performing emergency cleanup...")
                            emergency_result = self.cache_manager.cleanup_cache(
                                force=True,
                                target_size_gb=5
                            )
                            if emergency_result['cleaned']:
                                print(f"  ✓ Emergency cleanup freed {emergency_result['cleaned_size_gb']:.2f} GB")
                    except Exception as e:
                        print(f"  Could not check disk space: {e}")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Pipeline interrupted by user")
                print(f"\nProgress before interruption:")
                print(f"  Processed: {i+1}/{len(model_uids)} models")
                print(f"  Successful: {len(successful_models)}")
                print(f"  Failed: {len(failed_models)}")
                print(f"  Skipped: {len(skipped_models)}")
                raise
                
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                self.error_tracker.log_error(model_uid, 'pipeline', e, 'pipeline')
                failed_models.append(model_uid)
        
        # Final summary
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"\nTotal models: {len(model_uids)}")
        print(f"  ✓ Successful: {len(successful_models)}")
        print(f"  ⊘ Skipped: {len(skipped_models)}")
        print(f"  ✗ Failed: {len(failed_models)}")
        
        if failed_models:
            print(f"\nFailed models:")
            for model_uid in failed_models[:10]:  # Show first 10
                print(f"  - {model_uid}")
            if len(failed_models) > 10:
                print(f"  ... and {len(failed_models) - 10} more")
        
        # Show final cache and disk status
        final_cache = self.cache_manager.get_cache_size_gb()
        print(f"\nFinal cache size: {final_cache['total']:.2f} GB")
        
        try:
            cache_dir = self.cache_manager.torch_cache.parent
            stat = shutil.disk_usage(str(cache_dir))
            available_gb = stat.free / (1024**3)
            print(f"Final disk space available: {available_gb:.1f} GB")
        except:
            pass
        
        # Print full error summary
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        
        # Completion status
        completed = self.status_tracker.get_completed_count()
        print("\nCompleted tasks:")
        for task, count in completed.items():
            print(f"  {task}: {count} models")
        
        # Error summary
        self.error_tracker.print_summary()
        
        # Check if consolidated metrics file was created
        final_file = Path(self.config.RESULTS_DIR) / "model_metadata_with_metrics.csv"
        if final_file.exists():
            print(f"\n✓ Results saved to: {final_file}")