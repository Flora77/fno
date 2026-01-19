'''
Training script for sea surface elevation prediction using FNO.

This script trains a Fourier Neural Operator to predict future sea surface
heights from observed snapshots.  Optimized for: 
- 128x128 grid over 640m x 640m observation area
- Predicting full 640m x 640m region
- 60s prediction horizon from 10s history
- Real-time inference (<10s)
- <10% prediction error
'''

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch. distributed as dist
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop. data.datasets. sea_surface import load_sea_surface_data
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop. utils import get_wandb_api_key, count_model_params
from neuralop.mpu. comm import get_local_rank
from neuralop. training import setup, AdamW

# Configuration setup
from dataclasses import asdict

# Add project root to sys.path to allow importing from config regardless of CWD
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.sea_surface_config import Default

# Parse configuration from CLI or use defaults
try:
    from zencfg import make_config_from_cli
    config = make_config_from_cli(Default)
except ImportError: 
    config = Default()

config_dict = config.to_dict() if hasattr(config, 'to_dict') else asdict(config)

# Distributed training setup
device, is_logger = setup(config_dict)

# WandB logging configuration
if config. wandb.log and is_logger:
    try:
        wandb.login(key=get_wandb_api_key())
    except Exception as e:
        print(f"WandB login failed: {e}. Disabling WandB logging.")
        config.wandb.log = False

if config.wandb.log and is_logger:
    wandb_init_args = dict(
        config=config_dict,
        name=config.wandb.name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config. wandb.entity,
    )
    wandb.init(**wandb_init_args)

# Print configuration details
if config.verbose and is_logger: 
    print("=" * 60)
    print("SEA SURFACE ELEVATION PREDICTION - FNO TRAINING")
    print("=" * 60)
    print(f"\n### CONFIGURATION ###")
    print(f"Observation area: {config. data.observation_size[0]}m x {config. data.observation_size[1]}m")
    print(f"Prediction area:  {config.data. prediction_size[0]}m x {config.data.prediction_size[1]}m")
    print(f"Grid size:  {config.data. grid_size} x {config.data.grid_size}")
    print(f"Time step: {config.data.dt}s")
    print(f"Input history: {config.data.input_steps * config.data.dt}s ({config.data.input_steps} steps)")
    print(f"Prediction horizon: {config.data.output_steps * config.data.dt}s ({config. data.output_steps} steps)")
    print(f"\nModel: {config.model.model_arch}")
    print(f"Fourier modes: {config.model.n_modes}")
    print(f"Hidden channels: {config.model.hidden_channels}")
    print(f"Layers: {config.model.n_layers}")
    print("=" * 60)

# Data directory setup
data_dir = Path(config. data.folder).expanduser()

# Load the sea surface dataset
train_loader, test_loader, data_processor = load_sea_surface_data(
    data_root=data_dir,
    n_train=config.data. n_train,
    n_test=config.data.n_test,
    batch_size=config.data.batch_size,
    test_batch_size=config.data.test_batch_size,
    input_steps=config.data. input_steps,
    output_steps=config.data.output_steps,
    observation_size=tuple(config.data. observation_size),
    prediction_size=tuple(config.data.prediction_size),
    grid_size=config.data. grid_size,
    dt=config. data.dt,
    stride=1, # Use stride 1 to maximize training data (overlapping windows)
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
    num_workers=config. data.num_workers,
)

test_loaders = {"sea_surface": test_loader}

# Model initialization
model = get_model(config_dict)
model = model.to(device)

if hasattr(data_processor, "to"):
    try:
        data_processor = data_processor.to(device)
    except AttributeError:
        # Fallback if specific attributes (like std) are None
        pass

# Distributed data parallel setup
if config.distributed.use_distributed:
    train_sampler = DistributedSampler(train_loader. dataset, rank=get_local_rank())
    train_loader = DataLoader(
        dataset=train_loader. dataset,
        batch_size=config. data.batch_size,
        sampler=train_sampler,
        num_workers=config. data.num_workers,
    )
    
    test_sampler = DistributedSampler(test_loader.dataset, rank=get_local_rank())
    test_loaders["sea_surface"] = DataLoader(
        dataset=test_loader. dataset,
        batch_size=config. data.test_batch_size,
        shuffle=False,
        sampler=test_sampler,
    )

# Optimizer setup
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config. opt.weight_decay,
)

# Learning rate scheduler
if config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler. CosineAnnealingLR(
        optimizer, T_max=config. opt.scheduler_T_max
    )
elif config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, mode="min"
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )

# Loss function configuration
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else: 
    train_loss = h1loss

eval_losses = {"h1": h1loss, "l2":  l2loss}

if config.verbose and is_logger: 
    print("\n### MODEL ###")
    print(model)
    print(f"\n### OPTIMIZER:  {optimizer.__class__.__name__} ###")
    print(f"Learning rate: {config.opt.learning_rate}")
    print(f"Weight decay: {config.opt.weight_decay}")
    print(f"\n### SCHEDULER: {config.opt.scheduler} ###")
    print(f"\n### LOSS FUNCTIONS ###")
    print(f"Training:  {config.opt. training_loss}")
    print(f"Evaluation: {list(eval_losses. keys())}")
    sys.stdout.flush()

# Create trainer
trainer = Trainer(
    model=model,
    n_epochs=config. opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config. opt.mixed_precision,
    eval_interval=config.opt. eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config. distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log,
)

# Log model parameter count
if is_logger:
    n_params = count_model_params(model)
    print(f"\n### MODEL PARAMETERS:  {n_params: ,} ###")
    
    if config.wandb. log:
        wandb.log({"n_params": n_params}, commit=False)
        wandb.watch(model)

# Start training
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60 + "\n")

trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# Save the trained model
if is_logger:
    model_save_path = Path("checkpoints/sea_surface_fno.pt")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # helper to clean config
    def clean_config(cfg):
        if hasattr(cfg, 'to_dict'):
            return cfg.to_dict()
        if isinstance(cfg, dict):
            return {k: clean_config(v) for k, v in cfg.items()}
        return cfg

    try:
        # Try to save with full config
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': clean_config(config_dict),
        }
        if hasattr(data_processor, 'in_normalizer'):
            save_dict['in_normalizer'] = data_processor.in_normalizer
        if hasattr(data_processor, 'out_normalizer'):
            save_dict['out_normalizer'] = data_processor.out_normalizer
            
        torch.save(save_dict, model_save_path)
    except Exception as e:
        print(f"Warning: Failed to save config with model: {e}")
        # Fallback: save only model state
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_save_path)

    print(f"\nModel saved to {model_save_path}")

# Benchmark inference time
if is_logger:
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK")
    print("=" * 60)
    
    model.eval()
    
    # Create dummy input matching expected shape
    in_ch = getattr(config.model, 'in_channels', getattr(config.model, 'data_channels', None))
    if in_ch is None:
        in_ch = 40 # Default fallback
        
    dummy_input = torch.randn(
        1, in_ch, config.data.grid_size, config.data.grid_size
    ).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time. time()
    
    n_runs = 100
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time. time() - start_time
    
    avg_inference_time = elapsed_time / n_runs
    print(f"Average inference time: {avg_inference_time * 1000:.2f} ms")
    print(f"Inference time for 60s prediction: {avg_inference_time * 1000:.2f} ms")
    
    if avg_inference_time < 10.0:
        print("✓ Meets real-time requirement (<10s)")
    else:
        print("✗ Does not meet real-time requirement")

# Finalize
if config.wandb.log and is_logger: 
    wandb.finish()

if dist.is_initialized():
    dist.destroy_process_group()

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)