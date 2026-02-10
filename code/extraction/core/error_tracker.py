"""
Error tracking system for pipeline processing
"""
import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class ErrorTracker:
    """Tracks and logs errors during model processing"""
    
    def __init__(self, error_log_path: str):
        self.error_log_path = Path(error_log_path)
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.errors = self.load_errors()
    
    def load_errors(self) -> Dict:
        """Load existing error log"""
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load error log, starting fresh")
                return {'models': {}, 'summary': {}}
        return {'models': {}, 'summary': {}}
    
    def log_error(self, model_uid: str, task: str, error: Exception, 
                  phase: str = 'processing'):
        """
        Log an error with detailed information
        
        Args:
            model_uid: Model identifier
            task: Task name (identity, geometry, relation)
            error: The exception that occurred
            phase: Phase where error occurred ('loading', 'processing', 'saving')
        """
        if model_uid not in self.errors['models']:
            self.errors['models'][model_uid] = {}
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        if task not in self.errors['models'][model_uid]:
            self.errors['models'][model_uid][task] = []
        
        self.errors['models'][model_uid][task].append(error_entry)
        
        # Update summary
        self._update_summary()
        
        # Save immediately
        self.save_errors()
    
    def _update_summary(self):
        """Update error summary statistics"""
        self.errors['summary'] = {
            'last_updated': datetime.now().isoformat(),
            'total_models_with_errors': len(self.errors['models']),
            'errors_by_task': {},
            'errors_by_type': {},
            'errors_by_phase': {}
        }
        
        for model_uid, tasks in self.errors['models'].items():
            for task, errors in tasks.items():
                # Count by task
                if task not in self.errors['summary']['errors_by_task']:
                    self.errors['summary']['errors_by_task'][task] = 0
                self.errors['summary']['errors_by_task'][task] += len(errors)
                
                for error in errors:
                    # Count by error type
                    error_type = error['error_type']
                    if error_type not in self.errors['summary']['errors_by_type']:
                        self.errors['summary']['errors_by_type'][error_type] = 0
                    self.errors['summary']['errors_by_type'][error_type] += 1
                    
                    # Count by phase
                    phase = error['phase']
                    if phase not in self.errors['summary']['errors_by_phase']:
                        self.errors['summary']['errors_by_phase'][phase] = 0
                    self.errors['summary']['errors_by_phase'][phase] += 1
    
    def save_errors(self):
        """Save error log to file"""
        try:
            with open(self.error_log_path, 'w') as f:
                json.dump(self.errors, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save error log: {e}")
    
    def get_failed_models(self, task: Optional[str] = None) -> List[str]:
        """
        Get list of models that failed
        
        Args:
            task: Optional task name to filter by
        
        Returns:
            List of model UIDs that failed
        """
        failed = []
        for model_uid, tasks in self.errors['models'].items():
            if task:
                if task in tasks and tasks[task]:
                    failed.append(model_uid)
            elif tasks:  # Any task failed
                failed.append(model_uid)
        return failed
    
    def get_error_summary(self) -> Dict:
        """Get summary statistics of errors"""
        if 'summary' in self.errors:
            return self.errors['summary']
        
        # Generate summary if not present
        self._update_summary()
        return self.errors['summary']
    
    def has_error(self, model_uid: str, task: Optional[str] = None) -> bool:
        """
        Check if a model has errors
        
        Args:
            model_uid: Model identifier
            task: Optional task name to check specifically
        
        Returns:
            True if model has errors
        """
        if model_uid not in self.errors['models']:
            return False
        
        if task:
            return task in self.errors['models'][model_uid] and \
                   len(self.errors['models'][model_uid][task]) > 0
        
        return len(self.errors['models'][model_uid]) > 0
    
    def clear_errors(self, model_uid: Optional[str] = None, task: Optional[str] = None):
        """
        Clear errors for a specific model/task or all
        
        Args:
            model_uid: Optional model to clear errors for
            task: Optional task to clear errors for
        """
        if model_uid and task:
            if model_uid in self.errors['models'] and task in self.errors['models'][model_uid]:
                del self.errors['models'][model_uid][task]
                if not self.errors['models'][model_uid]:
                    del self.errors['models'][model_uid]
        elif model_uid:
            if model_uid in self.errors['models']:
                del self.errors['models'][model_uid]
        else:
            self.errors['models'] = {}
        
        self._update_summary()
        self.save_errors()
    
    def print_summary(self):
        """Print a formatted error summary"""
        summary = self.get_error_summary()
        
        print("\n" + "="*50)
        print("ERROR SUMMARY")
        print("="*50)
        
        if summary.get('total_models_with_errors', 0) == 0:
            print("No errors recorded")
            return
        
        print(f"Total models with errors: {summary['total_models_with_errors']}")
        
        if summary['errors_by_task']:
            print("\nErrors by task:")
            for task, count in summary['errors_by_task'].items():
                print(f"  {task}: {count}")
        
        if summary['errors_by_type']:
            print("\nErrors by type:")
            for error_type, count in summary['errors_by_type'].items():
                print(f"  {error_type}: {count}")
        
        if summary['errors_by_phase']:
            print("\nErrors by phase:")
            for phase, count in summary['errors_by_phase'].items():
                print(f"  {phase}: {count}")
        
        print(f"\nLast updated: {summary.get('last_updated', 'Unknown')}")