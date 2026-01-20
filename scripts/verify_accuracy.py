import torch
import numpy as np
import scipy.io as sio
import h5py
import sys
from pathlib import Path

# Imports from existing scripts
sys.path.append(".")
from scripts.predict_sea_surface import SeaSurfacePredictor

def load_mat_data(file_path, variable='h'):
    try:
        data = sio.loadmat(file_path)
        if variable in data:
            return data[variable]
        for k in data.keys():
            if not k.startswith('__'):
                return data[k]
    except:
        with h5py.File(file_path, 'r') as f:
            if variable in f:
                return np.transpose(f[variable][()])
            for k in f.keys():
                if k != '#refs#':
                    return np.transpose(f[k][()])
    return None

def check_error():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="checkpoints/sea_surface_fno.pt", help="Path to model checkpoint")
    parser.add_argument('--data', type=str, default="neuralop/data/train_data_mat/PM_Tp7d5_Hs1_length640_x128_T140_to_300.mat", help="Path to test data")
    args = parser.parse_args()

    # Config
    model_path = args.model
    # 使用之前确认存在的文件
    data_path = args.data 
    
    print(f"Loading model from {model_path}...")
    try:
        # Init Predictor
        predictor = SeaSurfacePredictor(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from {data_path}...")
    # Load Data
    full_data = load_mat_data(data_path) 
    if full_data is None:
        print("Failed to load data")
        return
        
    print(f"Data shape: {full_data.shape}")
    
    # Input: 40 steps, Output: 240 steps
    t_input = 40
    t_output = 240
    
    # 为了验证泛化性，我们选取数据集中最后的一段样本（通常训练集和测试集是按比例划分的）
    # 假设最后一段数据没有参与训练或者至少能代表模型性能
    start_idx = full_data.shape[0] - t_input - t_output
    
    input_data = full_data[start_idx : start_idx + t_input]
    target_data = full_data[start_idx + t_input : start_idx + t_input + t_output]
    
    # Predict
    print("Running inference...")
    # 注意：这里我们使用 return_full_grid=True 因为我们已经设置了 target_data 为全尺寸
    pred, time_taken = predictor.predict(input_data, return_full_grid=True)
    
    # Check shape mismatch
    if pred.shape != target_data.shape:
        print(f"Shape mismatch: Pred {pred.shape} vs Target {target_data.shape}")
        # 如果预测的是中心区域，我们需要裁剪 target
        if pred.shape[1] != target_data.shape[1]:
            diff_h = (target_data.shape[1] - pred.shape[1]) // 2
            diff_w = (target_data.shape[2] - pred.shape[2]) // 2
            target_data = target_data[:, diff_h:-diff_h, diff_w:-diff_w]
            print(f"Cropped target to {target_data.shape}")

    # Calculate Error
    # Relative L2 = ||Pred - True|| / ||True||
    # 计算整个时空场的误差
    diff = pred - target_data
    l2_diff = np.linalg.norm(diff)
    l2_true = np.linalg.norm(target_data)
    rel_l2 = l2_diff / l2_true
    
    print("-" * 30)
    print(f"Evaluating on sample starting at index {start_idx}")
    print(f"Prediction Horizon: {t_output * 0.25}s")
    print(f"Inference Time: {time_taken*1000:.2f} ms")
    print(f"Relative L2 Error: {rel_l2:.4f} ({rel_l2*100:.2f}%)")
    print("-" * 30)
    
    if rel_l2 < 0.10:
        print("RESULT: SUCCESS (Error < 10%)")
    else:
        print("RESULT: FAIL (Error >= 10%)")

if __name__ == "__main__":
    check_error()
