"""
Data I/O utilities for robust file operations
"""
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
import fcntl
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class DataIO:
    """Handles all data input/output operations"""
    
    @staticmethod
    def save_csv_atomic(df: pd.DataFrame, filepath: str):
        """
        Save DataFrame to CSV atomically to prevent corruption
        
        Args:
            df: DataFrame to save
            filepath: Target file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file in the same directory
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent, 
            suffix='.tmp'
        )
        
        try:
            # Write to temporary file
            df.to_csv(temp_path, index=False)
            
            # Verify the write was successful
            test_df = pd.read_csv(temp_path)
            if len(test_df) != len(df):
                raise ValueError(f"Row count mismatch: {len(test_df)} vs {len(df)}")
            
            # Atomically replace the original file
            os.replace(temp_path, filepath)
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
        finally:
            os.close(temp_fd)
    
    @staticmethod
    def save_parquet(df: pd.DataFrame, filepath: str):
        """
        Save DataFrame to Parquet format
        
        Args:
            df: DataFrame to save
            filepath: Target file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with compression
        df.to_parquet(filepath, compression='snappy', index=False)
    
    @staticmethod
    def read_csv_with_lock(filepath: str, max_retries: int = 5) -> pd.DataFrame:
        """
        Read CSV with file locking for concurrent access
        
        Args:
            filepath: Path to CSV file
            max_retries: Maximum number of retry attempts
        
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        for attempt in range(max_retries):
            try:
                with open(filepath, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                    df = pd.read_csv(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return df
            except IOError:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
    
    @staticmethod
    def read_parquet(filepath: str) -> pd.DataFrame:
        """
        Read Parquet file
        
        Args:
            filepath: Path to Parquet file
        
        Returns:
            Loaded DataFrame
        """
        return pd.read_parquet(filepath)
    
    @staticmethod
    def backup_file(filepath: str, backup_dir: Optional[str] = None):
        """
        Create timestamped backup of a file
        
        Args:
            filepath: File to backup
            backup_dir: Optional backup directory (default: same dir + /backups)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return
        
        if backup_dir is None:
            backup_dir = filepath.parent / 'backups'
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(filepath, backup_path)
        
        # Keep only last 10 backups
        pattern = f"{filepath.stem}_backup_*{filepath.suffix}"
        backups = sorted(backup_dir.glob(pattern))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                old_backup.unlink()
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """
        Save dictionary to JSON file
        
        Args:
            data: Dictionary to save
            filepath: Target file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """
        Load JSON file
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Loaded dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {}
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

class StatusTracker:
    """Tracks processing status for models and tasks"""
    
    def __init__(self, status_file: str):
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.status = self.load_status()
    
    def load_status(self) -> Dict:
        """Load existing status or create new"""
        if self.status_file.exists():
            return DataIO.load_json(self.status_file)
        return {'models': {}, 'last_updated': None}
    
    def save_status(self):
        """Save current status to file"""
        self.status['last_updated'] = datetime.now().isoformat()
        DataIO.save_json(self.status, self.status_file)
    
    def is_completed(self, model_uid: str, task: str) -> bool:
        """
        Check if a model-task combination has been completed
        
        Args:
            model_uid: Model identifier
            task: Task name
        
        Returns:
            True if completed
        """
        # if model_uid not in self.status['models']:
        #     return False
        
        # if task not in self.status['models'][model_uid]:
        #     return False
        
        # return self.status['models'][model_uid][task].get('completed', False)
        df = pd.read_csv('/user_data/wenjiel2/abstraction/results/metrics/all_metrics.csv')

        task_metrics = {
            'identity': 'sem_dist_effect',
            'geometry': 'diff_symbolic_r',
            'relation': 'relational_bias',
            'relation_patternonly': 'relational_bias_patternonly',

        }
        # True if df[model_uid][task_metrics[task]] is not NaN
        if task not in task_metrics:
            raise ValueError(f"Unknown task: {task}")
        metric = task_metrics[task]
        if model_uid not in df['model_uid'].values:
            # error
            print(f"Model {model_uid} not found in metrics file")
            return False
            
        value = df[df['model_uid'] == model_uid][metric].values[0]
        return not pd.isna(value)
    
    def mark_completed(self, model_uid: str, task: str, 
                      layer: int = None, metric_value: float = None):
        """
        Mark a model-task combination as completed
        
        Args:
            model_uid: Model identifier
            task: Task name
            layer: Extraction layer used
            metric_value: Computed metric value
        """
        if model_uid not in self.status['models']:
            self.status['models'][model_uid] = {}
        
        self.status['models'][model_uid][task] = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'layer': layer,
            'metric_value': metric_value
        }
        
        self.save_status()
    
    def get_completed_count(self) -> Dict[str, int]:
        """Get count of completed models per task"""
        counts = {}
        
        for model_uid, tasks in self.status['models'].items():
            for task, info in tasks.items():
                if info.get('completed', False):
                    if task not in counts:
                        counts[task] = 0
                    counts[task] += 1
        
        return counts
    
    def reset_model(self, model_uid: str):
        """Reset status for a specific model"""
        if model_uid in self.status['models']:
            del self.status['models'][model_uid]
            self.save_status()
    
    def reset_task(self, task: str):
        """Reset status for all models on a specific task"""
        for model_uid in self.status['models']:
            if task in self.status['models'][model_uid]:
                del self.status['models'][model_uid][task]
        self.save_status()