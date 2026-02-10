"""
Configuration settings for the abstraction analysis pipeline
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path

@dataclass
class ReferenceGroup:
    """Reference group configuration for comparison"""
    name: str
    column_name: str
    
@dataclass
class TaskConfig:
    """Configuration for a specific task"""
    name: str
    metric_name: str
    stimulus_dir: str
    reference_groups: List[ReferenceGroup] = field(default_factory=list)
    has_conditions: bool = False
    
@dataclass
class Config:
    """Main configuration class"""
    
    # ========== Paths ==========
    PROJECT_ROOT: str = '/user_data/wenjiel2/abstraction'
    
    @property
    def STIMULUS_DIR(self) -> str:
        return f'{self.PROJECT_ROOT}/stimulus'
    
    @property
    def RESULTS_DIR(self) -> str:
        return f'{self.PROJECT_ROOT}/results'
    
    @property
    def MODEL_METADATA_PATH(self) -> str:
        # return f'{self.PROJECT_ROOT}/model_metadata/model_metadata_enhanced_with_results_with_last_layer.csv'
        return f'{self.PROJECT_ROOT}/results/metrics/all_metrics.csv'
    
    @property
    def MODEL_METADATA_PATTERNONLY_PATH(self) -> str:
        return f'{self.PROJECT_ROOT}/model_metadata/model_metadata_with_patternonly.csv'

    @property
    def PATTERNS_CSV_PATH(self) -> str:
        return f'{self.PROJECT_ROOT}/results/patterns_only_combo.csv'
    
    # ========== Essential Metadata Columns ==========
    METADATA_COLUMNS: List[str] = field(default_factory=lambda: [
        'model_uid', 'source', 'model_name', 'weights', 'pretrained',
        'architecture', 'training_objective', 'dataset_name', 
        'is_video_model', 'has_language_component', 'is_quantized',
        'dataset_size_log10', 'training_data_size', 'architecture_detailed',
        'training_paradigm', 'model_size_category', 'model_params',
        'model_params_millions', 'model_params_log10', 'is_modern',
        'is_finetuned', 'is_developmental'
    ])
    
    # ========== Model Processing ==========
    SKIP_MODELS: Set[str] = field(default_factory=lambda: {
        'r2plus1d', '3d', 'video', 'xlarge', 'giant', 'gigantic',
        'beit_large', 'swin_v2_large', 'deit3_huge', 'omnivore', 
        'mvit', 'slowfast', 'gpt', 'bert', 'llama'
    })
    
    CLASSIFICATION_DIMS: Set[int] = field(default_factory=lambda: {
        1000, 21843, 21841, 10450, 11221, 11821,  # ImageNet variants
        365, 205,  # Places
        100, 10,   # CIFAR
        200,       # Tiny-ImageNet
        397,       # SUN397
        102,       # Flowers102
        120,       # Stanford Dogs
        196,       # Cars196
        101        # Food101
    })
    
    CLASSIFICATION_HEAD_INDICATORS: List[str] = field(default_factory=lambda: [
        'fc', 'classifier', 'head', 'logits', 'output', 'linear'
    ])
    
    # ========== Task Configurations ==========
    def __post_init__(self):
        """Initialize task configurations after dataclass initialization"""
        self.TASKS: Dict[str, TaskConfig] = {
            'identity': TaskConfig(
                name='identity',
                metric_name='sem_dist_effect',
                stimulus_dir=f'{self.STIMULUS_DIR}/identity',
                reference_groups=[
                    ReferenceGroup('USADULT', 'USADULT_Distance'),
                    ReferenceGroup('MONKEY', 'MONKEY_Distance')
                ]
            ),
            'geometry': TaskConfig(
                name='geometry',
                metric_name='diff_symbolic_r',
                stimulus_dir=f'{self.STIMULUS_DIR}/geometry',
                reference_groups=[
                    ReferenceGroup('MONKEY', 'MONKEY_Distance'),
                    ReferenceGroup('KID', 'KID_Distance'),
                    ReferenceGroup('TSIADULT', 'TSIADULT_Accuracy')
                ],
                has_conditions=True
            ),
            'relation': TaskConfig(
                name='relation',
                metric_name='relational_bias',
                stimulus_dir=f'{self.STIMULUS_DIR}/relation',
                reference_groups=[
                    ReferenceGroup('MONKEY', 'MONKEY_Distance'),
                    ReferenceGroup('KID', 'KID_Distance'),
                    ReferenceGroup('USADULT', 'USADULT_Distance'),
                    ReferenceGroup('TSIADULT', 'TSIADULT_Distance')
                ]
            ),
            'relation_patternonly': TaskConfig(
                name='relation_patternonly',
                metric_name='relational_bias_patternonly',
                stimulus_dir=f'{self.STIMULUS_DIR}/relation',
                reference_groups=[
                    ReferenceGroup('MONKEY', 'MONKEY_Distance'),
                    ReferenceGroup('KID', 'KID_Distance'),
                    ReferenceGroup('USADULT', 'USADULT_Distance'),
                    ReferenceGroup('TSIADULT', 'TSIADULT_Distance')
                ]
            )
        }
    
    # ========== Processing Parameters ==========
    BATCH_SIZE: int = 8
    CHECKPOINT_FREQUENCY: int = 5
    FORCE_RECOMPUTE: bool = False
    
    # ========== Cache Settings ==========
    MAX_CACHE_SIZE_GB: int = 50
    CACHE_CLEANUP_THRESHOLD_GB: int = 40
    
    # ========== Device Settings ==========
    @property
    def DEVICE(self) -> str:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'