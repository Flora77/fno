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
        obs_resolution = observation_size[0] / grid_size  # meters per grid point
        pred_grid_size = int(prediction_size[0] / obs_resolution)
        
        # Center the prediction region
        self.pred_start = (grid_size - pred_grid_size) // 2
        self.pred_end = self.pred_start + pred_grid_size
        
        # Metadata storage for lazy loading: (file_idx, start_time_idx)
        self.sample_metadata = []
        # Cache for loaded full arrays to avoid repeated I/O but share memory
        self.data_cache = []
        
        self._load_metadata()
        
    def _load_metadata(self):
        """
        Load logical structure of data processing without duplicating tensors.
        Stores raw full-sequence data in self.data_cache and indices in self.sample_metadata.
        """
        for file_idx, file_path in enumerate(self.data_files):
            try:
                # Load the full file only once
                data = sio.loadmat(str(file_path))
                
                # Find variable
                if 'h' in data: h = data['h']
                elif 'eta' in data: h = data['eta']
                elif 'elevation' in data: h = data['elevation']
                else:
                    keys = [k for k in data.keys() if not k.startswith('__')]
                    if len(keys) == 1: h = data[keys[0]]
                    else: raise ValueError(f"Cannot identify data variable in {file_path}")
            
            except NotImplementedError:
                # HDF5 Fallback
                with h5py.File(file_path, 'r') as f:
                    if 'h' in f: h = f['h'][()]
                    elif 'eta' in f: h = f['eta'][()]
                    elif 'elevation' in f: h = f['elevation'][()]
                    else:
                        keys = [k for k in f.keys() if k != '#refs#']
                        if len(keys) >= 1: h = f[keys[0]][()]
                        else: raise ValueError(f"Cannot identify data variable in {file_path} (HDF5)")
                    h = np.transpose(h)

            h = np.array(h, dtype=np.float32)
            
            # DEBUG: Print shape
            print(f"DEBUG: Loaded file {file_path.name}, shape={h.shape}, size={h.nbytes / 1024**3:.2f} GB")

            if h.ndim == 2:
                h = h[np.newaxis, :, :]
            
            # Store the raw big array in cache (efficient shared memory)
            self.data_cache.append(h)
            
            # Calculate valid start indices
            total_steps = h.shape[0]
            required_steps = self.input_steps + self.output_steps
            
            # Create metadata tuple (file_idx, start_time_idx)
            for start_idx in range(0, total_steps - required_steps + 1, self.stride):
                self.sample_metadata.append((file_idx, start_idx))
        
        logger.info(f"Loaded metadata for {len(self.sample_metadata)} samples from {len(self.data_files)} files. Data kept in shared cache.")
    
    def __len__(self):
        return len(self.sample_metadata)
    
    def __getitem__(self, idx):
        file_idx, start_idx = self.sample_metadata[idx]
        
        # Access raw data from cache
        h = self.data_cache[file_idx]
        
        # Slice on-the-fly (Zero-Copy usually for numpy slice)
        input_seq = h[start_idx : start_idx + self.input_steps]
        output_seq = h[start_idx + self.input_steps : start_idx + self.input_steps + self.output_steps]
        
        # Extract prediction region
        output_seq = output_seq[:, self.pred_start:self.pred_end, self.pred_start:self.pred_end]
        
        sample = {
            'x': torch.from_numpy(input_seq), # Creates tensor, might copy
            'y': torch.from_numpy(output_seq), 
        }
        
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
    # IMPORTANT: Do NOT stack all samples, as this duplicates memory (OOM). 
    # Use a random subset to estimate statistics.
    # 100 samples is enough for decent mean/std estimation
    subset_size = min(100, len(train_indices))
    subset_indices = train_indices[:subset_size] # Since indices are already randomized by randperm
    
    # Efficient loading for stats
    train_inputs = []
    train_outputs = []
    for i in subset_indices:
        # Manually get item to avoid transform overhead if any, though here transform is None usually
        sample = full_dataset[i]
        train_inputs.append(sample['x'])
        train_outputs.append(sample['y'])
    
    train_inputs = torch.stack(train_inputs)
    train_outputs = torch.stack(train_outputs)
    
    logger.info(f"Computed Normalizer stats using {subset_size} samples.")
    
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