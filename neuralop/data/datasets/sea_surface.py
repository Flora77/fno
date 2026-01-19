"""
Sea surface elevation dataset for wave prediction using neural operators. 

This module handles loading and processing of sea surface height data
stored in . mat format for training FNO models to predict wave evolution.
"""

import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
import scipy.io as sio
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from neuralop.data.transforms. data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

logger = logging.getLogger(__name__)


class SeaSurfaceDataset(Dataset):
    """
    Dataset for sea surface elevation prediction. 
    
    Loads . mat files containing sea surface height snapshots h(t,x,y)
    and creates input-output pairs for temporal prediction.
    
    Parameters
    ----------
    data_files : List[Path]
        List of . mat files containing sea surface data
    input_steps : int
        Number of input time steps (history length)
    output_steps : int
        Number of output time steps to predict (60s / 0.25s = 240 steps)
    observation_size :  Tuple[int, int]
        Size of observation area in meters (1500, 1500)
    prediction_size : Tuple[int, int]
        Size of prediction area in meters (500, 500)
    grid_size : int
        Grid resolution (128)
    dt : float
        Time step in seconds (0.25)
    stride : int
        Stride for sliding window sampling (default: 1)
    transform : Optional[callable]
        Optional transform to apply to samples
    """
    
    def __init__(
        self,
        data_files: List[Path],
        input_steps: int = 40,  # 10s of history at 0.25s intervals
        output_steps: int = 240,  # 60s prediction at 0.25s intervals
        observation_size: Tuple[int, int] = (1500, 1500),
        prediction_size:  Tuple[int, int] = (500, 500),
        grid_size: int = 128,
        dt: float = 0.25,
        stride: int = 1,
        transform: Optional[callable] = None,
    ):
        super().__init__()
        self.data_files = data_files
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.observation_size = observation_size
        self. prediction_size = prediction_size
        self.grid_size = grid_size
        self.dt = dt
        self.stride = stride
        self.transform = transform
        
        # Calculate prediction region indices (center 500m of 1500m)
        # For 128 grid points over 1500m, 500m corresponds to ~43 points
        obs_resolution = observation_size[0] / grid_size  # meters per grid point
        pred_grid_size = int(prediction_size[0] / obs_resolution)
        
        # Center the prediction region
        self.pred_start = (grid_size - pred_grid_size) // 2
        self.pred_end = self.pred_start + pred_grid_size
        
        # Load all data
        self. samples = []
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess all . mat files."""
        for file_path in self.data_files:
            try:
                data = sio.loadmat(str(file_path))
                
                # Assuming . mat file contains 'h' variable with shape (T, X, Y)
                # Adjust key name based on actual data structure
                if 'h' in data: 
                    h = data['h']
                elif 'eta' in data: 
                    h = data['eta']
                elif 'elevation' in data: 
                    h = data['elevation']
                else:
                    # Try to find the main data array
                    keys = [k for k in data.keys() if not k.startswith('__')]
                    if len(keys) == 1:
                        h = data[keys[0]]
                    else:
                        raise ValueError(f"Cannot identify data variable in {file_path}")
            except NotImplementedError:
                # Fallback for MATLAB v7.3 files which use HDF5 format
                with h5py.File(file_path, 'r') as f:
                    if 'h' in f:
                        h = f['h'][()]
                    elif 'eta' in f:
                        h = f['eta'][()]
                    elif 'elevation' in f:
                        h = f['elevation'][()]
                    else:
                        keys = [k for k in f.keys() if k != '#refs#']
                        if len(keys) >= 1:
                            h = f[keys[0]][()]
                        else:
                            raise ValueError(f"Cannot identify data variable in {file_path} (HDF5)")
                    
                    # MATLAB v7.3 saves arrays in column-major order, h5py reads as row-major.
                    # This results in transposed dimensions compared to loadmat.
                    h = np.transpose(h)

            h = np.array(h, dtype=np.float32)
            
            # Ensure shape is (T, X, Y)
            if h.ndim == 2:
                h = h[np.newaxis, :, :]
            
            # Create input-output pairs with sliding window
            total_steps = h.shape[0]
            required_steps = self. input_steps + self.output_steps
            
            for start_idx in range(0, total_steps - required_steps + 1, self.stride):
                input_seq = h[start_idx:start_idx + self. input_steps]
                output_seq = h[start_idx + self.input_steps: 
                              start_idx + self.input_steps + self.output_steps]
                
                # Extract prediction region for output
                output_seq = output_seq[
                    : , 
                    self. pred_start:self.pred_end, 
                    self.pred_start:self.pred_end
                ]
                
                self.samples.append({
                    'x': torch.from_numpy(input_seq),
                    'y': torch.from_numpy(output_seq),
                })
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.data_files)} files")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self. samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def load_sea_surface_data(
    data_root: Union[str, Path],
    n_train: int,
    n_test: int,
    batch_size: int,
    test_batch_size: int,
    input_steps: int = 40,
    output_steps: int = 240,
    observation_size:  Tuple[int, int] = (1500, 1500),
    prediction_size: Tuple[int, int] = (500, 500),
    grid_size: int = 128,
    dt: float = 0.25,
    stride: int = 1,
    encode_input: bool = True,
    encode_output: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DefaultDataProcessor]:
    """
    Load sea surface elevation data for training and testing.
    
    Parameters
    ----------
    data_root :  Union[str, Path]
        Root directory containing .mat files
    n_train : int
        Number of training samples to use
    n_test : int
        Number of test samples to use
    batch_size : int
        Training batch size
    test_batch_size :  int
        Test batch size
    input_steps : int
        Number of input time steps
    output_steps : int
        Number of output time steps to predict
    observation_size : Tuple[int, int]
        Observation area size in meters
    prediction_size : Tuple[int, int]
        Prediction area size in meters
    grid_size : int
        Grid resolution
    dt : float
        Time step in seconds
    stride : int
        Stride for sliding window sampling
    encode_input : bool
        Whether to normalize inputs
    encode_output : bool
        Whether to normalize outputs
    num_workers : int
        Number of data loading workers
        
    Returns
    -------
    train_loader : DataLoader
    test_loader : DataLoader
    data_processor : DefaultDataProcessor
    """
    data_root = Path(data_root)
    
    # Find all .mat files
    mat_files = sorted(data_root.glob("*.mat"))
    if len(mat_files) == 0:
        raise ValueError(f"No .mat files found in {data_root}")
    
    logger.info(f"Found {len(mat_files)} .mat files in {data_root}")
    
    # Split files for train/test (or use all files and split samples)
    # Here we create dataset from all files and split samples
    full_dataset = SeaSurfaceDataset(
        data_files=mat_files,
        input_steps=input_steps,
        output_steps=output_steps,
        observation_size=observation_size,
        prediction_size=prediction_size,
        grid_size=grid_size,
        dt=dt,
        stride=stride,
    )
    
    # Limit samples
    total_samples = len(full_dataset)
    n_train = min(n_train, int(total_samples * 0.8))
    n_test = min(n_test, total_samples - n_train)
    
    # Create train/test split indices
    indices = torch.randperm(total_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train: n_train + n_test]
    
    train_dataset = torch.utils.data. Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Compute normalization statistics from training data
    train_inputs = torch.stack([full_dataset. samples[i]['x'] for i in train_indices])
    train_outputs = torch.stack([full_dataset.samples[i]['y'] for i in train_indices])
    
    in_normalizer = None
    out_normalizer = None
    
    if encode_input:
        in_normalizer = UnitGaussianNormalizer(dim=None)
        in_normalizer.fit(train_inputs)
    if encode_output: 
        out_normalizer = UnitGaussianNormalizer(dim=None)
        out_normalizer.fit(train_outputs)
    
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader, data_processor