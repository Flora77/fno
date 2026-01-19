import torch
import sys
from pathlib import Path
from dataclasses import asdict

# path setup
sys.path.append(".")
from neuralop.data.datasets.sea_surface import load_sea_surface_data
from config.sea_surface_config import Default

def clean_config(cfg):
    if hasattr(cfg, 'to_dict'):
        return clean_config(cfg.to_dict())
    if isinstance(cfg, dict):
        return {k: clean_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [clean_config(v) for v in cfg]
    return cfg

def fix():
    config = Default()
    data_dir = Path("neuralop/data/train_data_mat") # Force correct path
    
    print("Calculing normalizers from data...")
    try:
        # This will compute stats
        _, _, data_processor = load_sea_surface_data(
            data_root=data_dir,
            n_train=config.data.n_train,
            n_test=config.data.n_test,
            batch_size=config.data.batch_size,
            test_batch_size=config.data.test_batch_size,
            input_steps=config.data.input_steps,
            output_steps=config.data.output_steps,
            observation_size=tuple(config.data.observation_size),
            prediction_size=tuple(config.data.prediction_size),
            grid_size=config.data.grid_size,
            dt=config.data.dt,
            stride=10, 
            encode_input=config.data.encode_input,
            encode_output=config.data.encode_output,
            num_workers=0, # avoid spawn issues
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Loading existing checkpoint...")
    ckpt_path = Path("checkpoints/sea_surface_fno.pt")
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    ckpt = torch.load(ckpt_path)
    
    print("Patching checkpoint...")
    ckpt['in_normalizer'] = data_processor.in_normalizer
    ckpt['out_normalizer'] = data_processor.out_normalizer
    
    # Remove config if it exists to verify it doesn't cause issues
    if 'config' in ckpt:
        del ckpt['config']

    torch.save(ckpt, ckpt_path)
    print("Checkpoint patched and saved.")

if __name__ == "__main__":
    fix()
