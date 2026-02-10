"""
Cache management system for model downloads
"""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import time

class CacheManager:
    """Manages model cache from torch hub and huggingface"""
    
    def __init__(self, max_size_gb: int = 50, cleanup_threshold_gb: int = 40):
        self.max_size_gb = max_size_gb
        self.cleanup_threshold_gb = cleanup_threshold_gb
        
        # CRITICAL FIX: Use environment variables if set, otherwise use default
        torch_home = os.environ.get('TORCH_HOME')
        hf_home = os.environ.get('HF_HOME')
        
        if torch_home:
            self.torch_cache = Path(torch_home) / 'hub'
        else:
            self.torch_cache = Path.home() / '.cache' / 'torch' / 'hub'
        
        if hf_home:
            self.huggingface_cache = Path(hf_home) / 'hub'
        else:
            self.huggingface_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
        
        # Ensure cache directories exist
        self.torch_cache.mkdir(parents=True, exist_ok=True)
        self.huggingface_cache.mkdir(parents=True, exist_ok=True)
        
        # All cache directories to monitor
        self.cache_dirs = [self.torch_cache, self.huggingface_cache]
    
    def get_cache_size_gb(self) -> Dict[str, float]:
        """Get size of each cache directory and total in GB"""
        sizes = {}
        total_size = 0
        
        for cache_path in self.cache_dirs:
            if cache_path.exists():
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                size_gb = size / (1024**3)
                sizes[str(cache_path)] = size_gb
                total_size += size_gb
        
        sizes['total'] = total_size
        return sizes
    
    def get_model_cache_info(self) -> List[Tuple[str, float, float, str]]:
        """
        Get info about cached models: (path, size_gb, last_accessed, cache_type)
        Returns sorted by last accessed (oldest first)
        """
        cache_info = []
        
        # Check Torch hub cache
        if self.torch_cache.exists():
            for model_dir in self.torch_cache.iterdir():
                if model_dir.is_dir():
                    try:
                        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                        # Get most recent access time from all files
                        access_times = [f.stat().st_atime for f in model_dir.rglob('*') if f.is_file()]
                        if access_times:
                            last_accessed = max(access_times)
                            cache_info.append((
                                str(model_dir), 
                                size / (1024**3), 
                                last_accessed,
                                'torch'
                            ))
                    except Exception as e:
                        print(f"Error getting info for {model_dir}: {e}")
        
        # Check HuggingFace hub cache
        if self.huggingface_cache.exists():
            # HuggingFace structure is different - models are in subdirectories
            for subdir in self.huggingface_cache.iterdir():
                if subdir.is_dir():
                    try:
                        size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                        access_times = [f.stat().st_atime for f in subdir.rglob('*') if f.is_file()]
                        if access_times:
                            last_accessed = max(access_times)
                            cache_info.append((
                                str(subdir),
                                size / (1024**3),
                                last_accessed,
                                'huggingface'
                            ))
                    except Exception as e:
                        print(f"Error getting info for {subdir}: {e}")
        
        # Sort by last accessed time (oldest first for LRU cleanup)
        return sorted(cache_info, key=lambda x: x[2])
    
    def cleanup_cache(self, force: bool = False, target_size_gb: float = None) -> Dict:
        """
        Clean up cache if needed or forced
        Args:
            force: Force cleanup regardless of current size
            target_size_gb: Target size after cleanup (default: 50% of threshold)
        """
        cache_sizes = self.get_cache_size_gb()
        current_total = cache_sizes['total']
        
        print(f"\nCurrent cache sizes:")
        for path, size in cache_sizes.items():
            if path != 'total':
                print(f"  {Path(path).name}: {size:.2f} GB")
        print(f"  Total: {current_total:.2f} GB")
        
        # Check if cleanup needed
        if not force and current_total < self.cleanup_threshold_gb:
            return {
                'cleaned': False,
                'reason': f'Cache size ({current_total:.1f}GB) below threshold ({self.cleanup_threshold_gb}GB)',
                'current_size_gb': current_total,
                'sizes_by_dir': cache_sizes
            }
        
        # Determine target size
        if target_size_gb is None:
            target_size_gb = self.cleanup_threshold_gb * 0.5  # Default to 50% of threshold
        
        # Get cached models sorted by last access time (oldest first)
        cache_info = self.get_model_cache_info()
        
        if not cache_info:
            return {
                'cleaned': False,
                'reason': 'No cached models found',
                'current_size_gb': current_total
            }
        
        cleaned_models = []
        cleaned_size = 0
        failed_cleanups = []
        
        print(f"\nCleaning cache to target size: {target_size_gb:.1f} GB")
        
        # Clean oldest models first until we reach target size
        for model_path, size_gb, last_accessed, cache_type in cache_info:
            if current_total - cleaned_size <= target_size_gb:
                break
            
            # Format last accessed time for display
            last_accessed_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(last_accessed))
            model_name = Path(model_path).name
            
            print(f"  Cleaning {model_name} ({cache_type}): {size_gb:.2f} GB, last used: {last_accessed_str}")
            
            try:
                shutil.rmtree(model_path)
                cleaned_models.append({
                    'name': model_name,
                    'size_gb': size_gb,
                    'cache_type': cache_type,
                    'last_accessed': last_accessed_str
                })
                cleaned_size += size_gb
            except Exception as e:
                print(f"    Failed to clean: {e}")
                failed_cleanups.append(model_name)
        
        final_size = current_total - cleaned_size
        
        return {
            'cleaned': True,
            'cleaned_models': cleaned_models,
            'cleaned_size_gb': cleaned_size,
            'initial_size_gb': current_total,
            'final_size_gb': final_size,
            'failed_cleanups': failed_cleanups,
            'target_size_gb': target_size_gb
        }
    
    def clear_all_cache(self, confirm: bool = False) -> Dict:
        """
        Completely clear all cache directories
        Args:
            confirm: Safety flag to prevent accidental deletion
        """
        if not confirm:
            return {
                'error': 'Must set confirm=True to clear all cache',
                'cache_dirs': [str(d) for d in self.cache_dirs]
            }
        
        initial_sizes = self.get_cache_size_gb()
        cleared = []
        failed = []
        
        for cache_path in self.cache_dirs:
            if cache_path.exists():
                try:
                    print(f"Clearing {cache_path}...")
                    shutil.rmtree(cache_path)
                    cache_path.mkdir(parents=True, exist_ok=True)
                    cleared.append(str(cache_path))
                except Exception as e:
                    print(f"Failed to clear {cache_path}: {e}")
                    failed.append(str(cache_path))
        
        return {
            'cleared_paths': cleared,
            'failed_paths': failed,
            'freed_space_gb': initial_sizes['total'],
            'initial_sizes': initial_sizes
        }
    
    def get_cache_summary(self) -> Dict:
        """Get detailed summary of cache status"""
        sizes = self.get_cache_size_gb()
        cache_info = self.get_model_cache_info()
        
        # Group by cache type
        torch_models = [m for m in cache_info if m[3] == 'torch']
        hf_models = [m for m in cache_info if m[3] == 'huggingface']
        
        # Find largest models
        largest_models = sorted(cache_info, key=lambda x: x[1], reverse=True)[:5]
        
        # Find oldest models (candidates for cleanup)
        oldest_models = cache_info[:5] if len(cache_info) >= 5 else cache_info
        
        return {
            'total_size_gb': sizes['total'],
            'sizes_by_directory': sizes,
            'total_models': len(cache_info),
            'torch_models': len(torch_models),
            'huggingface_models': len(hf_models),
            'largest_models': [
                {'name': Path(m[0]).name, 'size_gb': m[1], 'type': m[3]} 
                for m in largest_models
            ],
            'oldest_models': [
                {
                    'name': Path(m[0]).name, 
                    'size_gb': m[1], 
                    'type': m[3],
                    'last_accessed': time.strftime('%Y-%m-%d %H:%M', time.localtime(m[2]))
                } 
                for m in oldest_models
            ],
            'cleanup_recommended': sizes['total'] > self.cleanup_threshold_gb
        }