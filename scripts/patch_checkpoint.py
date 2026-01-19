import torch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from neuralop.data.datasets.sea_surface import load_sea_surface_data
from neuralop.data.transforms.data_processors import DefaultDataProcessor

def patch_checkpoint():
    model_path = Path("checkpoints/sea_surface_fno.pt")
    data_dir = Path("neuralop/data/train_data_mat")
    
    print("Loading existing checkpoint...")
    checkpoint = torch.load(model_path)
    
    print("Re-computing normalization statistics from training data (this may take a moment)...")
    # Re-use the loading logic to get the processor with fitted normalizers
    # We use a small subset or minimal settings just to trigger the fitting, 
    # but we need ALL training data to fit correctly unless we accept approximation.
    # The load_sea_surface_data computes stats on the `train_indices` subset.
    # We should match the config used for training.
    
    # Quick config reconstruction based on user's last run
    train_loader, _, data_processor = load_sea_surface_data(
        data_root=data_dir,
        n_train=1000, 
        n_test=200,
        batch_size=4,
        test_batch_size=4,
        input_steps=40,
        output_steps=240,
        observation_size=(640, 640),
        prediction_size=(640, 640),
        grid_size=128,
        dt=0.25,
        stride=1,
        encode_input=True,
        encode_output=True,
        num_workers=0
    )
    
    print("Injecting normalizers into checkpoint...")
    checkpoint['in_normalizer'] = data_processor.in_normalizer
    checkpoint['out_normalizer'] = data_processor.out_normalizer
    
    print("Saving patched checkpoint...")
    torch.save(checkpoint, model_path)
    print("Done! Checkpoint patched.")

if __name__ == "__main__":
    patch_checkpoint()
