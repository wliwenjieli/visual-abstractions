# core/__init__.py
"""Core functionality modules"""
from .cache_manager import CacheManager
from .error_tracker import ErrorTracker
from .data_io import DataIO, StatusTracker
from .model_processor import ModelProcessor
from .model_utils import (
    get_extraction_layer,
    should_skip_model,
    cleanup_model,
    emergency_cleanup
)
from .metrics import MetricsComputer
from .pipeline import AbstractionPipeline
from .dinov2_fix import extract_dinov2_features_fixed, load_dinov2_base_model

__all__ = [
    'CacheManager',
    'ErrorTracker', 
    'DataIO',
    'StatusTracker',
    'ModelProcessor',
    'get_extraction_layer',
    'should_skip_model',
    'cleanup_model',
    'emergency_cleanup',
    'MetricsComputer',
    'AbstractionPipeline'
]

# ===== Save this as core/__init__.py =====