"""
Model utilities including layer selection logic
"""
import numpy as np
import pandas as pd
import torch
import gc
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

def get_extraction_layer(model: Any, model_metadata: pd.Series, config: Any) -> int:
    """
    Determine which layer to extract features from.

    Logic:
    1. Check if training objective is Classification
    2. Check if last layer has classification output dimension (most reliable)
    3. Check if last layer name contains classification indicators AND is a real layer

    Args:
        model: The loaded model
        model_metadata: Row from metadata DataFrame
        config: Configuration object

    Returns:
        -1 for last layer, -2 for penultimate layer
    """
    model_uid = model_metadata.get('model_uid', '')

    # Feature-only models: no classification head, use last layer
    feature_only_indicators = ['_features', '_feat', 'feature_extractor']
    if any(indicator in model_uid.lower() for indicator in feature_only_indicators):
        print(f"  Layer selection: Using last layer (-1) - Feature-only model variant")
        return -1

    # Method 1: Check training objective in metadata
    if pd.notna(model_metadata.get('training_objective')):
        if model_metadata['training_objective'] == 'Classification':
            print(f"  Layer selection: Using penultimate (-2) - Classification objective")
            return -2

    # Method 2: Check if last layer has classification output dimension
    # This is more reliable than just checking layer names (Identity layers can have 'fc' in name)
    try:
        last_layer_dim = get_layer_output_dim(model, -1)
        if last_layer_dim is not None and last_layer_dim in config.CLASSIFICATION_DIMS:
            print(f"  Layer selection: Using penultimate (-2) - Dimension {last_layer_dim} matches classification")
            return -2
    except Exception as e:
        print(f"  Warning: Could not get last layer dimension: {e}")

    # Method 3: Check layer name for classification head patterns
    # Only if the layer is a real layer (Linear, Conv, etc.), not Identity
    try:
        last_layer_name = get_last_layer_name(model)
        last_layer_type = get_last_layer_type(model)

        # Skip if it's an Identity layer (placeholder, no real classification head)
        if last_layer_type and 'Identity' in last_layer_type:
            print(f"  Layer selection: Using last layer (-1) - Last layer is Identity (no classification head)")
            return -1

        if last_layer_name:
            for indicator in config.CLASSIFICATION_HEAD_INDICATORS:
                if indicator in last_layer_name.lower():
                    print(f"  Layer selection: Using penultimate (-2) - Found '{indicator}' in layer name")
                    return -2
    except Exception as e:
        print(f"  Warning: Could not get last layer name: {e}")

    # Default: use last layer
    print(f"  Layer selection: Using last layer (-1) - No classification head detected")
    return -1


def get_last_layer_type(model: Any) -> Optional[str]:
    """Get the class name of the last layer in the model."""
    try:
        if hasattr(model, 'named_modules'):
            last_module = None
            for name, module in model.named_modules():
                if name:  # Skip empty names
                    last_module = module
            if last_module is not None:
                return last_module.__class__.__name__
    except Exception:
        pass
    return None

def get_last_layer_name(model: Any) -> Optional[str]:
    """
    Get the name of the last layer in the model
    
    Args:
        model: The loaded model
    
    Returns:
        Name of the last layer or None if cannot determine
    """
    try:
        # For PyTorch models
        if hasattr(model, 'named_modules'):
            last_name = None
            for name, module in model.named_modules():
                if name:  # Skip empty names
                    last_name = name
            return last_name
        
        # For models with different structure
        if hasattr(model, 'layers'):
            if isinstance(model.layers, list) and len(model.layers) > 0:
                last_layer = model.layers[-1]
                if hasattr(last_layer, 'name'):
                    return last_layer.name
                elif hasattr(last_layer, '__class__'):
                    return last_layer.__class__.__name__
    except Exception as e:
        print(f"    Could not determine last layer name: {e}")
    
    return None

def get_layer_output_dim(model: Any, layer_idx: int) -> Optional[int]:
    """
    Get the output dimension of a specific layer
    
    Args:
        model: The loaded model
        layer_idx: Layer index (-1 for last, -2 for penultimate)
    
    Returns:
        Output dimension or None if cannot determine
    """
    try:
        # Try to get from model architecture
        if hasattr(model, 'fc') and layer_idx == -1:
            # Common pattern for classification heads
            if hasattr(model.fc, 'out_features'):
                return model.fc.out_features
            elif hasattr(model.fc, 'weight'):
                return model.fc.weight.shape[0]
        
        if hasattr(model, 'classifier') and layer_idx == -1:
            if hasattr(model.classifier, 'out_features'):
                return model.classifier.out_features
            elif isinstance(model.classifier, torch.nn.Sequential):
                last = model.classifier[-1]
                if hasattr(last, 'out_features'):
                    return last.out_features
        
        if hasattr(model, 'head') and layer_idx == -1:
            if hasattr(model.head, 'out_features'):
                return model.head.out_features
        
        # Try to infer from a forward pass with dummy input
        # This is more expensive but more general
        return None  # Skip this for now to avoid loading models unnecessarily
        
    except Exception as e:
        print(f"    Could not determine output dimension: {e}")
    
    return None

def should_skip_model(model_uid: str, config: Any) -> bool:
    """
    Check if a model should be skipped based on configuration
    
    Args:
        model_uid: Model identifier
        config: Configuration object
    
    Returns:
        True if model should be skipped
    """
    model_uid_lower = model_uid.lower()
    
    for skip_pattern in config.SKIP_MODELS:
        if skip_pattern in model_uid_lower:
            return True
    
    return False

def clean_model_cache(model_uid: str):
    """
    Clean up model-specific cache and memory
    
    Args:
        model_uid: Model identifier
    """
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()

def cleanup_model(model: Any):
    """
    Clean up a loaded model from memory
    
    Args:
        model: The model to clean up
    """
    try:
        # Delete the model
        del model
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        print(f"Warning: Error during model cleanup: {e}")

def emergency_cleanup():
    """
    Emergency cleanup when memory is critically low
    """
    print("  Running emergency cleanup...")
    
    # Clear all CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Aggressive garbage collection
    for _ in range(3):
        gc.collect()
    
    # Check memory status
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")