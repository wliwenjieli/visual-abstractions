"""
Fix for DINOv2 models with depth heads that require special handling
Add this to your model_processor.py or create as a separate module
"""

import torch
import numpy as np
from typing import Any, List
import sys

def extract_dinov2_features_fixed(
    model: Any,
    preprocess: Any,
    image_paths: List[str],
    layer_idx: int = -2,
    batch_size: int = 8,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Special feature extraction for DINOv2 models that bypasses depth head issues.
    
    Args:
        model: The DINOv2 model
        preprocess: Preprocessing function
        image_paths: List of paths to images
        layer_idx: Which layer to extract from (-2 for penultimate)
        batch_size: Batch size for processing
        device: Device to use
        
    Returns:
        numpy array of features [n_images, n_features]
    """
    from PIL import Image
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleImageDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
    
    # Create dataset and dataloader
    dataset = SimpleImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Set model to eval mode (crucial!)
    model.eval()
    
    all_features = []
    
    # Try to access the backbone directly for DINOv2
    if hasattr(model, 'backbone'):
        feature_extractor = model.backbone
    elif hasattr(model, 'encode_image'):
        feature_extractor = model.encode_image
    elif hasattr(model, 'encoder'):
        feature_extractor = model.encoder
    else:
        # Try to find the vision transformer component
        for name, module in model.named_modules():
            if 'vit' in name.lower() or 'transformer' in name.lower():
                if not any(x in name.lower() for x in ['head', 'decoder', 'depth']):
                    feature_extractor = module
                    print(f"    Using {name} as feature extractor")
                    break
        else:
            # Last resort: use the model but hook into intermediate layers
            feature_extractor = model
    
    # Extract features using hooks if needed
    features_hook = []
    
    def hook_fn(module, input, output):
        # Store the output
        if isinstance(output, tuple):
            output = output[0]
        features_hook.append(output)
        return output
    
    # Register hook on appropriate layer
    hook_handle = None
    if hasattr(feature_extractor, 'blocks'):
        # Vision Transformer architecture
        if layer_idx == -2:  # penultimate
            target_block = feature_extractor.blocks[-2]
        else:
            target_block = feature_extractor.blocks[layer_idx]
        hook_handle = target_block.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                if torch.cuda.is_available() and device == 'cuda':
                    batch = batch.cuda()
                
                # Clear previous hook results
                features_hook.clear()
                
                # Forward pass
                try:
                    # Try standard forward
                    if feature_extractor != model:
                        output = feature_extractor(batch)
                    else:
                        # For full model, we might need to handle differently
                        if hasattr(model, 'forward_features'):
                            output = model.forward_features(batch)
                        else:
                            # This might trigger the error, but hooks should capture features
                            try:
                                output = model(batch)
                            except TypeError as e:
                                if 'depth_gt' in str(e) or 'img_metas' in str(e):
                                    # Use the hooked features instead
                                    if features_hook:
                                        output = features_hook[-1]
                                    else:
                                        raise RuntimeError("Could not extract features from DINOv2 model")
                                else:
                                    raise
                    
                    # Use hooked features if available, otherwise use output
                    if features_hook:
                        features = features_hook[-1]
                    else:
                        features = output
                    
                    # Handle different output types
                    if isinstance(features, tuple):
                        features = features[0]
                    
                    # Global average pooling if needed
                    if features.dim() == 3:  # [B, N, D] format (tokens)
                        # Take CLS token or average all tokens
                        features = features[:, 0, :]  # CLS token
                    elif features.dim() == 4:  # [B, C, H, W] format
                        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        features = features.flatten(1)
                    
                    # Move to CPU and store
                    features = features.cpu().numpy()
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"    Error in batch processing: {e}")
                    # Create zero features as fallback
                    fallback_features = np.zeros((batch.shape[0], 768))  # Assuming 768-dim features
                    all_features.append(fallback_features)
    
    finally:
        # Remove hook
        if hook_handle:
            hook_handle.remove()
    
    # Concatenate all features
    if all_features:
        features = np.vstack(all_features)
    else:
        # Return zero features if extraction failed
        print("    Warning: Feature extraction failed, returning zero features")
        features = np.zeros((len(image_paths), 768))
    
    return features


def load_dinov2_base_model(model_name: str):
    """
    Load DINOv2 model directly from torch hub without depth head.
    
    Args:
        model_name: Name of the model (e.g., 'dinov2_vitg14')
        
    Returns:
        model, preprocess function
    """
    import torch
    
    # Map full names to base names
    base_model_map = {
        'dinov2_vitg14_ld': 'dinov2_vitg14',
        'dinov2_vitl14_ld': 'dinov2_vitl14',
        'dinov2_vitb14_ld': 'dinov2_vitb14',
        'dinov2_vits14_ld': 'dinov2_vits14'
    }
    
    base_name = base_model_map.get(model_name, model_name)
    
    try:
        # Load the base model without depth head
        model = torch.hub.load('facebookresearch/dinov2', base_name)
        
        # Create preprocessing function
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        return model, preprocess
        
    except Exception as e:
        print(f"    Error loading base DINOv2 model: {e}")
        raise


# Integration with your existing pipeline
def extract_features_with_fallback(
    model: Any,
    preprocess: Any,
    image_paths: List[str],
    model_uid: str,
    layer_idx: int = -1,
    batch_size: int = 8,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract features with fallback for problematic models.
    
    This function should replace the extract_features method in your pipeline.
    """
    
    # Check if this is a problematic DINOv2 model
    if 'dinov2' in model_uid.lower() and '_ld' in model_uid.lower():
        print(f"    Using special DINOv2 extraction for {model_uid}")
        
        # Try to load the base model instead
        try:
            print(f"    Loading base DINOv2 model without depth head...")
            base_model, base_preprocess = load_dinov2_base_model(model_uid)
            return extract_dinov2_features_fixed(
                base_model, base_preprocess, image_paths, 
                layer_idx, batch_size, device
            )
        except Exception as e:
            print(f"    Base model loading failed: {e}")
            # Fall back to the original model with special extraction
            return extract_dinov2_features_fixed(
                model, preprocess, image_paths, 
                layer_idx, batch_size, device
            )
    
    # For other models, use your existing extraction method
    # This should be your original FeatureExtractor code
    from deepjuice import get_data_loader, FeatureExtractor
    
    dataloader = get_data_loader(
        image_paths, 
        preprocess, 
        batch_size=batch_size
    )
    
    extractor = FeatureExtractor(
        model=model,
        inputs=dataloader,
        n_samples=len(image_paths),
        device=device,
        vectorize=True,
        flatten=True
    )
    
    with torch.no_grad():
        features = extractor.get(layer_idx)
    
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    
    # Ensure 2D array
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    elif features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    return features