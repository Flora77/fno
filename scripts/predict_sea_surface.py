"""
Real-time inference script for sea surface elevation prediction. 

This script loads a trained FNO model and performs real-time predictions
of sea surface heights.  Designed to meet <10s inference requirement for
60s predictions.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import asdict

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.sea_surface_config import Default

import numpy as np
import scipy.io as sio
import h5py
import torch
import torch.nn as nn


class SeaSurfacePredictor:
    """
    Real-time sea surface elevation predictor using trained FNO. 
    
    Parameters
    ----------
    model_path : str or Path
        Path to trained model checkpoint
    device : str
        Device to run inference on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model_path: str,
        device:  str = 'cuda' if torch.cuda. is_available() else 'cpu',
    ):
        self.device = torch. device(device)
        model_path = Path(model_path)
        
        # Check if model_path is a directory (Resume style checkpoint)
        if model_path.is_dir():
            print(f"Loading checkpoint from directory: {model_path}")
            # Try to find model file
            if (model_path / "best_model_state_dict.pt").exists():
                model_file = model_path / "best_model_state_dict.pt"
            elif (model_path / "model_state_dict.pt").exists():
                model_file = model_path / "model_state_dict.pt"
            else:
                raise FileNotFoundError(f"No model state dict found in {model_path}")
            
            # Load weights
            state_dict = torch.load(model_file, map_location=self.device, weights_only=False)
            
            # Load config (Not usually in resume dir, use default)
            print("Using default configuration for resume checkpoint.")
            config_obj = Default()
            self.config = config_obj.to_dict() if hasattr(config_obj, 'to_dict') else asdict(config_obj)
            
            # Load normalizers if available
            self.in_normalizer = None
            self.out_normalizer = None
            if (model_path / "data_processor.pt").exists():
                dp_state = torch.load(model_path / "data_processor.pt", map_location=self.device, weights_only=False)
                self.in_normalizer = dp_state.get('in_normalizer')
                self.out_normalizer = dp_state.get('out_normalizer')
                print("Loaded normalizers from data_processor.pt")
            else:
                print("Warning: data_processor.pt not found. Normalization might be missing!")
                
        else:
            # Single file checkpoint (Legacy or Final style)
            print(f"Loading checkpoint from file: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                print("Warning: Config not found in checkpoint. Using default configuration.")
                config_obj = Default()
                self.config = config_obj.to_dict() if hasattr(config_obj, 'to_dict') else asdict(config_obj)
            
            # Helper to get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint # Assume raw state dict if key missing
            
            # Load normalization parameters if available
            self.in_normalizer = checkpoint.get('in_normalizer', None)
            self.out_normalizer = checkpoint.get('out_normalizer', None)
        
        # Reconstruct model
        from neuralop import get_model
        self.model = get_model(self.config)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract config parameters
        self. input_steps = self. config['data']['input_steps']
        self.output_steps = self. config['data']['output_steps']
        self.grid_size = self. config['data']['grid_size']
        self.dt = self.config['data']['dt']
        
        # Prediction region (center 500m of 1500m)
        self.obs_size = self.config['data']['observation_size'][0]
        self.pred_size = self.config['data']['prediction_size'][0]
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Input steps: {self. input_steps} ({self.input_steps * self.dt}s)")
        print(f"Output steps: {self.output_steps} ({self.output_steps * self.dt}s)")
    
    def predict(
        self,
        input_data: np.ndarray,
        return_full_grid: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict future sea surface elevation. 
        
        Parameters
        ----------
        input_data : np. ndarray
            Input sea surface heights, shape (T, H, W) where T >= input_steps
        return_full_grid :  bool
            If True, return prediction for full grid; else return center region
            
        Returns
        -------
        prediction : np.ndarray
            Predicted sea surface heights, shape (output_steps, H', W')
        inference_time : float
            Time taken for inference in seconds
        """
        # Validate input
        if input_data.shape[0] < self.input_steps:
            raise ValueError(
                f"Input must have at least {self.input_steps} time steps, "
                f"got {input_data.shape[0]}"
            )
        
        # Take last input_steps frames
        input_seq = input_data[-self.input_steps:]
        
        # Convert to tensor
        x = torch.from_numpy(input_seq).float()
        
        # Add batch dimension:  (T, H, W) -> (1, T, H, W)
        x = x.unsqueeze(0).to(self.device)

        # Normalize if normalizer available
        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        
        # Run inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            y_pred = self.model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.time() - start_time
        
        # Denormalize output
        if self.out_normalizer is not None: 
            y_pred = self.out_normalizer.inverse_transform(y_pred)
        
        # Convert to numpy
        prediction = y_pred.squeeze(0).cpu().numpy()
        
        # Extract center region if not returning full grid
        if not return_full_grid:
            _, _, h, w = x.shape
            # Calculate slicing based on actual input resolution
            obs_res = self.obs_size / h
            pred_grid_h = int(self.pred_size / obs_res)
            
            # Assuming square grid for now
            start_x = (h - pred_grid_h) // 2
            end_x = start_x + pred_grid_h
            
            start_y = (w - pred_grid_h) // 2
            end_y = start_y + pred_grid_h
            
            prediction = prediction[:, start_x:end_x, start_y:end_y]
        
        return prediction, inference_time
    
    def predict_from_mat(
        self,
        mat_file: str,
        variable_name: str = 'h',
        output_file: Optional[str] = None,
        return_full_grid: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Load data from . mat file and predict. 
        
        Parameters
        ----------
        mat_file : str
            Path to input .mat file
        variable_name : str
            Name of the variable containing sea surface data
        output_file : str, optional
            If provided, save predictions to this . mat file
        return_full_grid : bool
            If True, return prediction for full grid
            
        Returns
        -------
        prediction : np.ndarray
        inference_time : float
        """
        # Load . mat file
        try:
            data = sio.loadmat(mat_file)
            # Try specified variable or common names
            if variable_name in data:
                input_data = data[variable_name]
            elif 'h' in data:
                input_data = data['h']
            elif 'eta' in data:
                input_data = data['eta']
            elif 'elevation' in data:
                input_data = data['elevation']
            else:
                # Try finding valid key
                keys = [k for k in data.keys() if not k.startswith('__')]
                if len(keys) == 1:
                    input_data = data[keys[0]]
                else:
                    raise ValueError(f"Variable {variable_name} not found in {mat_file}")
            
            input_data = np.array(input_data, dtype=np.float32)

        except NotImplementedError:
            # Fallback for MATLAB v7.3 files
            with h5py.File(mat_file, 'r') as f:
                if variable_name in f:
                    input_data = f[variable_name][()]
                elif 'h' in f:
                    input_data = f['h'][()]
                elif 'eta' in f:
                    input_data = f['eta'][()]
                elif 'elevation' in f:
                    input_data = f['elevation'][()]
                else:
                    keys = [k for k in f.keys() if k != '#refs#']
                    if len(keys) >= 1:
                        input_data = f[keys[0]][()]
                    else:
                        raise ValueError(f"Variable {variable_name} not found in {mat_file}")
                
                # Transpose for v7.3
                input_data = np.transpose(input_data)
                input_data = np.array(input_data, dtype=np.float32)

        # Run prediction
        prediction, inference_time = self.predict(input_data, return_full_grid=return_full_grid)
        
        # Save if output file specified
        if output_file:
            output_data = {
                'h_predicted': prediction,
                'dt': self.dt,
                'prediction_horizon': self.output_steps * self.dt,
            }
            sio.savemat(output_file, output_data)
            print(f"Predictions saved to {output_file}")
        
        return prediction, inference_time
    
    def benchmark(self, n_runs: int = 100) -> dict:
        """
        Benchmark inference performance.
        
        Parameters
        ----------
        n_runs :  int
            Number of inference runs for benchmarking
            
        Returns
        -------
        results : dict
            Benchmark results including mean, std, min, max times
        """
        # Create dummy input
        dummy_input = np.random.randn(
            self.input_steps, self.grid_size, self. grid_size
        ).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _, _ = self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            _, inference_time = self. predict(dummy_input)
            times.append(inference_time)
        
        times = np.array(times)
        results = {
            'mean_ms': times.mean() * 1000,
            'std_ms': times.std() * 1000,
            'min_ms': times. min() * 1000,
            'max_ms': times.max() * 1000,
            'meets_realtime': times.mean() < 10.0,
        }
        
        print("\n=== Inference Benchmark Results ===")
        print(f"Mean:  {results['mean_ms']:.2f} ms")
        print(f"Std:   {results['std_ms']:.2f} ms")
        print(f"Min:  {results['min_ms']:.2f} ms")
        print(f"Max:  {results['max_ms']:.2f} ms")
        print(f"Meets <10s requirement: {'✓' if results['meets_realtime'] else '✗'}")
        
        return results


def main():
    """Example usage of the predictor."""
    import argparse
    
    parser = argparse. ArgumentParser(description='Sea Surface Prediction')
    parser.add_argument('--model', type=str, default=None, required=False,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .mat file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions')
    parser.add_argument('--variable', type=str, default='h',
                        help='Variable name in .mat file')
    parser.add_argument('--full-grid', action='store_true',
                        help='Return predictions for the full observation grid')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark')
    
    args = parser. parse_args()
    
    model_path = args.model
    
    # Auto-detect latest checkpoint if not provided
    if model_path is None:
        print("No model specified. Searching for latest checkpoint...")
        base_dir = Path(__file__).resolve().parent.parent
        resume_dir = base_dir / "checkpoints/sea_surface_resume"
        final_model = base_dir / "checkpoints/sea_surface_fno.pt"
        
        candidates = []
        
        # Check Final Model
        if final_model.exists():
            candidates.append(final_model)
            
        # Check Resume Epochs
        if resume_dir.exists():
            for p in resume_dir.glob("epoch_*"):
                if p.is_dir() and (p / "model_state_dict.pt").exists():
                    candidates.append(p)
        
        if not candidates:
            print(f"Error: No checkpoints found in {resume_dir} or {final_model}")
            sys.exit(1)
            
        # Sort by modification time
        latest_model = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        model_path = str(latest_model)
        print(f"Selected latest checkpoint: {model_path}")

    # Initialize predictor
    predictor = SeaSurfacePredictor(model_path)
    
    # Run benchmark if requested
    if args.benchmark:
        predictor.benchmark()
    
    # Run prediction
    prediction, inference_time = predictor.predict_from_mat(
        args.input,
        variable_name=args. variable,
        output_file=args. output,
        return_full_grid=args.full_grid,
    )
    
    print(f"\nPrediction shape: {prediction.shape}")
    print(f"Inference time: {inference_time * 1000:.2f} ms")
    print(f"Prediction horizon: {predictor.output_steps * predictor. dt}s")


if __name__ == '__main__':
    main()