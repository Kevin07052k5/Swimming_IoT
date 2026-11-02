"""
src/data_generator.py
Tạo tập dữ liệu kiểm thử mô phỏng hoặc nạp từ ETH Zurich Swimming Dataset (Brunner et al., ISWC 2019).
Mô phỏng 2 nhóm hành vi:
1. Bơi bình thường (4 kiểu bơi tiêu chuẩn: sải, ếch, ngửa, bướm và thả nổi).
2. Phản xạ đuối nước tự nhiên (IDR - Instinctive Drowning Response) hỗn loạn, mất chu kỳ.
"""
import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
from src.config import CONFIG, SystemConfig
from src.dsp_pipeline import SwimmingDSPPipeline

def generate_synthetic_normal_swimming(
    n_windows: int = 400,
    sampling_rate: int = CONFIG.SAMPLING_RATE_HZ,
    window_size: int = CONFIG.WINDOW_SIZE,
    random_seed: int = 42
) -> np.ndarray:
    """
    Giả lập động học cử động quạt tay bơi tuần hoàn ổn định (chu kỳ 1.0s - 1.5s).
    Các kiểu bơi: Sải (Freestyle), Ếch (Breaststroke), Ngửa (Backstroke), Thả nổi (Treading).
    Trả về mảng thô (N, 6): [Ax, Ay, Az, Gx, Gy, Gz]
    Đơn vị: Gia tốc (m/s^2 hoặc g), Con quay (deg/s)
    """
    np.random.seed(random_seed)
    total_samples = n_windows * window_size
    t = np.linspace(0, total_samples / sampling_rate, total_samples, endpoint=False)
    
    # Chia thời gian thành các phân đoạn kiểu bơi khác nhau
    segment_len = total_samples // 4
    
    acc_x = np.zeros(total_samples)
    acc_y = np.zeros(total_samples)
    acc_z = np.zeros(total_samples)
    gyro_x = np.zeros(total_samples)
    gyro_y = np.zeros(total_samples)
    gyro_z = np.zeros(total_samples)
    
    # 1. Bơi sải (Freestyle): quạt tay luân phiên, tần số ~ 0.85 Hz
    idx1 = slice(0, segment_len)
    t1 = t[idx1]
    acc_x[idx1] = 1.1 * np.sin(2 * np.pi * 0.85 * t1) + np.random.normal(0, 0.12, len(t1))
    acc_y[idx1] = 1.8 * np.cos(2 * np.pi * 0.85 * t1) + np.random.normal(0, 0.15, len(t1))
    acc_z[idx1] = 9.8 + 0.6 * np.sin(4 * np.pi * 0.85 * t1) + np.random.normal(0, 0.10, len(t1))
    gyro_x[idx1] = 65.0 * np.sin(2 * np.pi * 0.85 * t1) + np.random.normal(0, 3.0, len(t1))
    gyro_y[idx1] = 45.0 * np.cos(2 * np.pi * 0.85 * t1) + np.random.normal(0, 2.5, len(t1))
    gyro_z[idx1] = 25.0 * np.sin(2 * np.pi * 0.85 * t1 + np.pi / 4) + np.random.normal(0, 2.0, len(t1))
    
    # 2. Bơi ếch (Breaststroke): động tác đồng thời, tần số ~ 0.70 Hz, biên độ đạp mạnh
    idx2 = slice(segment_len, 2 * segment_len)
    t2 = t[idx2]
    acc_x[idx2] = 2.4 * np.sin(2 * np.pi * 0.70 * t2) * (np.sin(2 * np.pi * 0.70 * t2) > 0) + np.random.normal(0, 0.15, len(t2))
    acc_y[idx2] = 0.9 * np.sin(2 * np.pi * 0.70 * t2) + np.random.normal(0, 0.12, len(t2))
    acc_z[idx2] = 9.8 + 1.2 * np.cos(2 * np.pi * 0.70 * t2) + np.random.normal(0, 0.12, len(t2))
    gyro_x[idx2] = 30.0 * np.sin(2 * np.pi * 0.70 * t2) + np.random.normal(0, 2.0, len(t2))
    gyro_y[idx2] = 55.0 * np.sin(2 * np.pi * 0.70 * t2 + np.pi / 2) + np.random.normal(0, 2.5, len(t2))
    gyro_z[idx2] = 18.0 * np.cos(2 * np.pi * 0.70 * t2) + np.random.normal(0, 1.8, len(t2))
    
    # 3. Bơi ngửa (Backstroke): quạt tay hướng lên, tần số ~ 0.80 Hz
    idx3 = slice(2 * segment_len, 3 * segment_len)
    t3 = t[idx3]
    acc_x[idx3] = -1.2 * np.sin(2 * np.pi * 0.80 * t3) + np.random.normal(0, 0.12, len(t3))
    acc_y[idx3] = -1.5 * np.cos(2 * np.pi * 0.80 * t3) + np.random.normal(0, 0.15, len(t3))
    acc_z[idx3] = -9.8 + 0.5 * np.sin(4 * np.pi * 0.80 * t3) + np.random.normal(0, 0.10, len(t3))
    gyro_x[idx3] = -60.0 * np.sin(2 * np.pi * 0.80 * t3) + np.random.normal(0, 3.0, len(t3))
    gyro_y[idx3] = -35.0 * np.cos(2 * np.pi * 0.80 * t3) + np.random.normal(0, 2.0, len(t3))
    gyro_z[idx3] = 20.0 * np.sin(2 * np.pi * 0.80 * t3) + np.random.normal(0, 1.5, len(t3))
    
    # 4. Thả nổi / Đứng nước (Treading water): tần số ~ 1.1 Hz, biên độ nhẹ
    idx4 = slice(3 * segment_len, total_samples)
    t4 = t[idx4]
    acc_x[idx4] = 0.4 * np.sin(2 * np.pi * 1.10 * t4) + np.random.normal(0, 0.08, len(t4))
    acc_y[idx4] = 0.5 * np.cos(2 * np.pi * 1.10 * t4) + np.random.normal(0, 0.08, len(t4))
    acc_z[idx4] = 9.8 + 0.3 * np.sin(2 * np.pi * 1.10 * t4) + np.random.normal(0, 0.08, len(t4))
    gyro_x[idx4] = 20.0 * np.sin(2 * np.pi * 1.10 * t4) + np.random.normal(0, 1.5, len(t4))
    gyro_y[idx4] = 22.0 * np.cos(2 * np.pi * 1.10 * t4) + np.random.normal(0, 1.5, len(t4))
    gyro_z[idx4] = 12.0 * np.sin(2 * np.pi * 1.10 * t4) + np.random.normal(0, 1.0, len(t4))
    
    raw_imu = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1).astype(np.float32)
    return raw_imu

def generate_synthetic_drowning_anomaly(
    n_windows: int = 100,
    sampling_rate: int = CONFIG.SAMPLING_RATE_HZ,
    window_size: int = CONFIG.WINDOW_SIZE,
    random_seed: int = 99
) -> np.ndarray:
    """
    Giả lập phản xạ đuối nước tự nhiên (IDR - Instinctive Drowning Response):
    Giai đoạn 1: Quẫy đạp hỗn loạn tần số cao, tỳ đè mặt nước mất chu kỳ tuần hoàn (0-30s).
    Giai đoạn 2: Chìm dần và bất động dưới nước (>30s).
    """
    np.random.seed(random_seed)
    total_samples = n_windows * window_size
    half_samples = int(total_samples * 0.75)
    
    # Giai đoạn 1: Quẫy đạp hỗn loạn tần số cao (IDR tích cực)
    acc_x_panic = np.random.uniform(-4.5, 4.5, half_samples)
    acc_y_panic = np.random.uniform(-5.0, 5.0, half_samples)
    acc_z_panic = 9.8 + np.random.uniform(-4.0, 3.5, half_samples)
    gyro_x_panic = np.random.uniform(-190.0, 190.0, half_samples)
    gyro_y_panic = np.random.uniform(-160.0, 160.0, half_samples)
    gyro_z_panic = np.random.uniform(-140.0, 140.0, half_samples)
    
    # Giai đoạn 2: Nạn nhân kiệt sức, chìm sâu và bất động (Immobility / Submerged)
    quiescent_samples = total_samples - half_samples
    acc_x_sink = np.random.normal(0.0, 0.05, quiescent_samples)
    acc_y_sink = np.random.normal(0.0, 0.05, quiescent_samples)
    acc_z_sink = 9.8 + np.random.normal(0.0, 0.05, quiescent_samples)
    gyro_x_sink = np.random.normal(0.0, 0.5, quiescent_samples)
    gyro_y_sink = np.random.normal(0.0, 0.5, quiescent_samples)
    gyro_z_sink = np.random.normal(0.0, 0.5, quiescent_samples)
    
    acc_x = np.concatenate([acc_x_panic, acc_x_sink])
    acc_y = np.concatenate([acc_y_panic, acc_y_sink])
    acc_z = np.concatenate([acc_z_panic, acc_z_sink])
    gyro_x = np.concatenate([gyro_x_panic, gyro_x_sink])
    gyro_y = np.concatenate([gyro_y_panic, gyro_y_sink])
    gyro_z = np.concatenate([gyro_z_panic, gyro_z_sink])
    
    raw_imu = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1).astype(np.float32)
    return raw_imu

def prepare_and_process_datasets(config: SystemConfig = CONFIG) -> Dict[str, Any]:
    """
    Quy trình chuẩn hóa và chuẩn bị toàn diện dữ liệu:
    1. Sinh hoặc đọc dữ liệu bơi bình thường
    2. Sinh dữ liệu đuối nước bất thường
    3. Chạy qua DSP Pipeline (Butterworth + Z-Score + Cửa sổ trượt)
    4. Lưu vào thư mục data/processed/ (.npy và .json thống kê)
    """
    dsp = SwimmingDSPPipeline(config)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    
    print("[*] 1. Đang chuẩn bị dữ liệu bơi lội bình thường...")
    raw_normal = generate_synthetic_normal_swimming(n_windows=500, random_seed=42)
    
    # Lọc và tính mean/std chuẩn từ tập huấn luyện bơi bình thường
    filtered_normal = dsp.apply_butterworth_filter(raw_normal)
    norm_normal, means, stds = dsp.z_score_normalize(filtered_normal)
    normal_windows = dsp.create_sliding_windows(norm_normal)
    print(f"    - Kích thước cửa sổ bơi bình thường: {normal_windows.shape}")
    
    print("[*] 2. Đang chuẩn bị dữ liệu kiểm thử bất thường (IDR)...")
    raw_anomaly = generate_synthetic_drowning_anomaly(n_windows=120, random_seed=101)
    filtered_anomaly = dsp.apply_butterworth_filter(raw_anomaly)
    # Dùng mean và std CỦA TẬP BÌNH THƯỜNG để chuẩn hóa tập dị thường (đúng chuẩn triển khai nhúng)
    norm_anomaly, _, _ = dsp.z_score_normalize(filtered_anomaly, means=means, stds=stds)
    anomaly_windows = dsp.create_sliding_windows(norm_anomaly)
    print(f"    - Kích thước cửa sổ đuối nước (IDR): {anomaly_windows.shape}")
    
    # Lưu các mảng .npy
    normal_file = config.DATA_PROCESSED_DIR / "normal_swimming.npy"
    anomaly_file = config.DATA_PROCESSED_DIR / "anomaly_swimming.npy"
    stats_file = config.DATA_PROCESSED_DIR / "sensor_stats.json"
    
    np.save(str(normal_file), normal_windows)
    np.save(str(anomaly_file), anomaly_windows)
    
    stats = {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        "window_size": config.WINDOW_SIZE,
        "window_stride": config.WINDOW_STRIDE
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"[+] Đã lưu dữ liệu tiền xử lý tại:")
    print(f"    - {normal_file}")
    print(f"    - {anomaly_file}")
    print(f"    - {stats_file}")
    
    return {
        "normal_windows": normal_windows,
        "anomaly_windows": anomaly_windows,
        "means": means,
        "stds": stds,
        "normal_file": str(normal_file),
        "anomaly_file": str(anomaly_file),
        "stats_file": str(stats_file)
    }

if __name__ == "__main__":
    prepare_and_process_datasets()
