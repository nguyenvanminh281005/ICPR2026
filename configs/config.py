"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    MODEL_TYPE: str = "restran"  # "crnn" or "restran"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network
    
    # Data paths
    DATA_ROOT: str = "data/old/train"
    TEST_DATA_ROOT: str = "data/test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "prediction.txt"
    
    IMG_HEIGHT: int = 64
    IMG_WIDTH: int = 256
    
    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Training hyperparameters
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 50
    SEED: int = 23520945
    NUM_WORKERS: int = 16
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = True
    
    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25
    
    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    
    # ===== NEW FEATURES =====
    
    # Temporal Fusion settings
    TEMPORAL_FUSION_TYPE: str = "quality"  # "attention", "quality", "transformer", "hybrid"
    
    # Super-Resolution settings
    USE_SR: bool = True  # Enable Super-Resolution module
    SR_TYPE: str = "lightweight"  # "lightweight" or "rcan_lite"
    SR_NUM_BLOCKS: int = 4
    SR_NUM_FEATURES: int = 64
    SR_UPSCALE: int = 2
    SR_LOSS_WEIGHT: float = 0.1
    
    # Contrastive Learning settings
    USE_CONTRASTIVE: bool = True  # Enable contrastive loss
    CONTRASTIVE_WEIGHT: float = 0.1
    CONTRASTIVE_TEMPERATURE: float = 0.07
    
    # Loss function settings
    LOSS_TYPE: str = "combined"  # "ctc", "combined", "focal", "smoothing"
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.1
    
    # Curriculum Learning settings
    USE_CURRICULUM: bool = True  # Enable curriculum learning
    CURRICULUM_TYPE: str = "curriculum"  # "curriculum", "self_paced", "anti", "mixed"
    CURRICULUM_START_RATIO: float = 0.3  # Start with 30% easiest samples
    CURRICULUM_END_EPOCH_RATIO: float = 0.7  # Use all samples by 70% of training
    
    # Test-Time Augmentation settings
    USE_TTA: bool = True  # Enable TTA during inference
    TTA_TYPE: str = "standard"  # "standard" or "multiscale"
    TTA_AUGMENTATIONS: List[str] = field(default_factory=lambda: [
        "original", "brightness_up", "brightness_down", "contrast"
    ])
    TTA_AGGREGATION: str = "mean"  # "mean" or "max"
    
    # ===== END NEW FEATURES =====
    
    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"
    
    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()
