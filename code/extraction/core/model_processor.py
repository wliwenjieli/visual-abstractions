"""
Model processor for loading and extracting features from models
"""
import sys
import torch
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, List, Optional
from pathlib import Path
from torchvision import transforms
from PIL import Image

# Add deepjuice path
sys.path.append('/user_data/wenjiel2/')

from deepjuice import (
    get_deepjuice_model,
    get_data_loader,
    FeatureExtractor,
    get_model_options
)

from core.model_utils import get_extraction_layer, cleanup_model


def wrap_transform_with_resize(preprocess, target_size=224):
    """
    Wrap a preprocessing transform to ensure images are resized to a consistent size.
    This handles models whose preprocess doesn't include a resize (e.g., segmentation models).

    Args:
        preprocess: Original preprocessing transform
        target_size: Target size for images (default 224x224)

    Returns:
        Wrapped transform that resizes then applies original preprocess
    """
    class ResizeWrapper:
        def __init__(self, original_transform, size):
            self.original_transform = original_transform
            self.resize = transforms.Resize((size, size), antialias=True)

        def __call__(self, img):
            # First resize to consistent size
            if isinstance(img, Image.Image):
                img = self.resize(img)
            elif torch.is_tensor(img):
                img = self.resize(img)
            # Then apply original preprocessing
            return self.original_transform(img)

    return ResizeWrapper(preprocess, target_size)

class ModelProcessor:
    """Handles model loading and feature extraction"""
    
    def __init__(self, config):
        """
        Initialize model processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.DEVICE
        
    def load_model(self, model_uid: str) -> Tuple[Any, Any]:
        """
        Load a model from DeepJuice.
        
        Args:
            model_uid: Model identifier
            
        Returns:
            Tuple of (model, preprocess_function)
        """
        print(f"    Loading {model_uid}...")
        
        try:
            # DeepJuice doesn't take device parameter - it handles it internally
            model, preprocess = get_deepjuice_model(model_uid)
            
            # Move model to device if needed
            if torch.cuda.is_available() and self.device == 'cuda':
                model = model.cuda()
            
            # Set to eval mode
            if hasattr(model, 'eval'):
                model.eval()
                
            return model, preprocess
            
        except Exception as e:
            print(f"    Error loading model: {e}")
            raise
    

    def extract_features(
        self,
        model: Any,
        preprocess: Any,
        image_paths: List[str],
        layer_idx: int = -1,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract features from a model for given images.
        """
        # Wrap preprocess with resize to ensure consistent image sizes
        # This fixes issues with segmentation/detection models that don't resize
        wrapped_preprocess = wrap_transform_with_resize(preprocess, target_size=224)

        # Create data loader with wrapped preprocessing
        dataloader = get_data_loader(
            image_paths,
            wrapped_preprocess,
            batch_size=batch_size
        )
        # Check if model requires img_metas
        model_needs_meta = False
        if hasattr(model, 'forward'):
            import inspect
            sig = inspect.signature(model.forward)
            if 'img_metas' in sig.parameters:
                model_needs_meta = True
        
        if model_needs_meta:
            # Create dummy img_metas for each batch
            # This is a workaround for models that require metadata
            features_list = []
            for batch in dataloader:
                if torch.is_tensor(batch):
                    batch_size_actual = batch.shape[0]
                    # Create minimal img_metas
                    img_metas = [{'img_shape': (224, 224, 3), 
                                'ori_shape': (224, 224, 3),
                                'pad_shape': (224, 224, 3),
                                'scale_factor': 1.0} 
                                for _ in range(batch_size_actual)]
                    
                    # Extract features with img_metas
                    with torch.no_grad():
                        if self.device == 'cuda' and batch.device != 'cuda':
                            batch = batch.cuda()
                        output = model(batch, img_metas=img_metas)
                        # Get intermediate features if needed
                        # You might need to modify this based on model architecture
                        features_list.append(output)
            
            features = torch.cat(features_list, dim=0)
            if torch.is_tensor(features):
                features = features.cpu().numpy()
        else:
            # Use standard FeatureExtractor for models that don't need img_metas
            print("to load feature extractor")
            extractor = FeatureExtractor(
                model=model,
                inputs=dataloader,
                n_samples=len(image_paths),
                device=self.device,
                vectorize=True,
                flatten=True
            )
            print("feature extractor loaded")

            with torch.no_grad():
                features = extractor.get(layer_idx)

            if torch.is_tensor(features):
                features = features.cpu().numpy()
            
        
        # Ensure 2D array [n_images, n_features]
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        
        return features

    # def extract_features(
    #     self,
    #     model: Any,
    #     preprocess: Any,
    #     image_paths: List[str],
    #     layer_idx: int = -1,
    #     batch_size: int = 8
    # ) -> np.ndarray:
    #     """
    #     Extract features from a model for given images.
        
    #     Args:
    #         model: Loaded model
    #         preprocess: Preprocessing function
    #         image_paths: List of image paths
    #         layer_idx: Layer index to extract from (-1 for last, -2 for penultimate)
    #         batch_size: Batch size for processing
            
    #     Returns:
    #         Array of features [n_images, n_features]
    #     """
    #     # Create data loader
    #     dataloader = get_data_loader(
    #         image_paths, 
    #         preprocess, 
    #         batch_size=batch_size
    #     )
        
    #     # Initialize feature extractor
    #     # The extractor automatically extracts features during initialization
    #     extractor = FeatureExtractor(
    #         model=model,
    #         inputs=dataloader,
    #         n_samples=len(image_paths),
    #         device=self.device,
    #         vectorize=True,  # This flattens spatial dimensions
    #         flatten=True  # Flatten to 2D: [n_samples, n_features]
    #     )
        
    #     # Get features for the specified layer using the .get() method
    #     # DeepJuice supports negative indexing: -1 for last layer, -2 for penultimate
    #     with torch.no_grad():
    #         features = extractor.get(layer_idx)
        
    #     # Convert to numpy if needed
    #     if torch.is_tensor(features):
    #         features = features.cpu().numpy()
        
    #     # Ensure 2D array [n_images, n_features]
    #     if features.ndim == 1:
    #         features = features.reshape(-1, 1)
    #     elif features.ndim > 2:
    #         # Flatten extra dimensions if not already flattened
    #         features = features.reshape(features.shape[0], -1)
        
    #     return features
    
    def process_model_for_task(
        self,
        model_uid: str,
        task_name: str,
        image_paths: List[str],
        model_metadata: Optional[pd.Series] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process a model for a specific task.
        
        Args:
            model_uid: Model identifier
            task_name: Task name
            image_paths: List of image paths
            model_metadata: Optional metadata for the model
            
        Returns:
            Dictionary with extracted features
        """
        # Load model
        model, preprocess = self.load_model(model_uid)
        
        try:
            # Determine extraction layer
            if model_metadata is not None:
                layer_idx = get_extraction_layer(model, model_metadata, self.config)
            else:
                layer_idx = -1  # Default to last layer
                
            print(f"    Extracting features from layer {layer_idx}...")
            
            # Extract features
            features = self.extract_features(
                model,
                preprocess,
                image_paths,
                layer_idx=layer_idx,
                batch_size=self.config.BATCH_SIZE
            )
            
            return {
                'features': features,
                'layer_idx': layer_idx,
                'feature_dim': features.shape[1]
            }
            
        finally:
            # Always clean up
            cleanup_model(model)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from DeepJuice.
        
        Returns:
            List of model UIDs
        """
        try:
            return get_model_options()
        except Exception as e:
            print(f"Error getting model options: {e}")
            return []
    
    def validate_model(self, model_uid: str) -> bool:
        """
        Check if a model can be loaded.
        
        Args:
            model_uid: Model identifier
            
        Returns:
            True if model can be loaded
        """
        try:
            model, preprocess = self.load_model(model_uid)
            cleanup_model(model)
            return True
        except Exception as e:
            print(f"    Model validation failed: {e}")
            return False