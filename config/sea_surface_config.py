"""
Configuration for sea surface elevation prediction using FNO.

This configuration is optimized for: 
- 128x128 grid resolution
- 60s prediction horizon at 0.25s intervals
- Real-time inference (<10s)
- <10% prediction error
"""

from typing import List, Optional
from zencfg import ConfigBase


class DataConfig(ConfigBase):
    """Data configuration for sea surface prediction."""
    folder: str = "~/neuraloperator/neuralop/data/train_data_mat"
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 4
    test_batch_size: int = 4
    input_steps: int = 40  # 10s of history at 0.25s
    output_steps:  int = 240  # 60s prediction at 0.25s
    observation_size: List[int] = [640, 640]
    prediction_size: List[int] = [640, 640]
    grid_size: int = 128
    dt: float = 0.25
    encode_input: bool = True
    encode_output:  bool = True
    num_workers: int = 4


class ModelConfig(ConfigBase):
    """FNO model configuration optimized for wave prediction."""
    model_arch: str = "tfno"  # 2D Temporal FNO
    
    # Input:  (batch, input_steps, H, W) -> treat input_steps as channels
    data_channels: int = 40  # input_steps
    out_channels: int = 240  # output_steps
    
    # FNO architecture parameters
    n_modes: List[int] = [32, 32]  # Fourier modes
    hidden_channels: int = 128
    n_layers: int = 4
    
    # Factorization for efficiency (important for real-time inference)
    factorization: str = "tucker"
    rank: float = 0.42
    
    # Normalization and activation
    norm:  str = "instance_norm"
    
    # Skip connections for better gradient flow
    fno_skip:  str = "soft-gating"
    
    # Domain padding for handling boundaries
    domain_padding: float = 0.125


class OptConfig(ConfigBase):
    """Optimizer and training configuration."""
    n_epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler: str = "CosineAnnealingLR"
    scheduler_T_max: int = 500
    
    # Mixed precision for faster training
    mixed_precision: bool = False
    
    # Training loss
    training_loss:  str = "h1"  # H1 loss for smoother predictions
    
    # Evaluation interval
    eval_interval: int = 10
    
    # Gradient clipping
    clip_grad_norm:  float = 1.0


class DistributedConfig(ConfigBase):
    """Distributed training configuration."""
    use_distributed: bool = False
    seed: Optional[int] = None
    model_parallel_size: int = 1


class WandbConfig(ConfigBase):
    """Weights & Biases logging configuration."""
    log:  bool = True
    name: str = "sea_surface_fno"
    project: str = "sea-surface-prediction"
    entity:  Optional[str] = None
    group: str = "fno_experiments"
    log_output: bool = True
    sweep:  bool = False


class PatchingConfig(ConfigBase):
    """Multi-grid patching configuration."""
    levels: int = 0  # No patching for this task
    padding: float = 0.0
    stitching: float = 0.0


class Default(ConfigBase):
    """Default configuration combining all sub-configs."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    opt: OptConfig = OptConfig()
    distributed: DistributedConfig = DistributedConfig()
    wandb: WandbConfig = WandbConfig()
    patching:  PatchingConfig = PatchingConfig()
    
    verbose: bool = True
    n_params_baseline: Optional[int] = None
