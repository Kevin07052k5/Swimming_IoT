import os
import numpy as np
import pandas as pd

os.makedirs("datasets", exist_ok=True)
np.random.seed(42)

# Chuẩn hóa cấu hình: 50 Hz, 128 mẫu/cửa sổ (2.56 giây)
FS = 50
NUM_SAMPLES = 128
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

def generate_normal_swimming(n_windows=300):
    """Giả lập chuyển động quạt tay bơi tuần hoàn ổn định (chu kỳ 1.2s)."""
    data = []
    t = np.linspace(0, n_windows * (NUM_SAMPLES / FS), n_windows * NUM_SAMPLES)
    
    # Sóng tuần hoàn kết hợp nhiễu nước nhẹ
    acc_x = 0.8 * np.sin(2 * np.pi * 0.8 * t) + np.random.normal(0, 0.1, len(t))
    acc_y = 1.2 * np.cos(2 * np.pi * 0.8 * t) + np.random.normal(0, 0.15, len(t))
    acc_z = 9.8 + 0.5 * np.sin(2 * np.pi * 1.6 * t) + np.random.normal(0, 0.1, len(t))
    gyro_x = 45.0 * np.sin(2 * np.pi * 0.8 * t) + np.random.normal(0, 2.0, len(t))
    gyro_y = 30.0 * np.cos(2 * np.pi * 0.8 * t) + np.random.normal(0, 2.0, len(t))
    gyro_z = 15.0 * np.sin(2 * np.pi * 0.8 * t + np.pi/4) + np.random.normal(0, 1.5, len(t))

    df = pd.DataFrame(np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1), columns=CHANNELS)
    return df

def generate_drowning_anomaly(n_windows=50):
    """Giả lập phản xạ đuối nước tự nhiên (IDR): hỗn loạn tần số cao, mất chu kỳ."""
    n_points = n_windows * NUM_SAMPLES
    acc_x = np.random.uniform(-3.5, 3.5, n_points)
    acc_y = np.random.uniform(-4.0, 4.0, n_points)
    acc_z = 9.8 + np.random.uniform(-2.5, 2.5, n_points)
    gyro_x = np.random.uniform(-180.0, 180.0, n_points)
    gyro_y = np.random.uniform(-150.0, 150.0, n_points)
    gyro_z = np.random.uniform(-120.0, 120.0, n_points)

    df = pd.DataFrame(np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1), columns=CHANNELS)
    return df

# Tạo file
train_df = generate_normal_swimming(400)
test_normal_df = generate_normal_swimming(60)
test_anomaly_df = generate_drowning_anomaly(60)

train_df.to_csv("datasets/train_normal.csv", index=False)
test_normal_df.to_csv("datasets/test_normal.csv", index=False)
test_anomaly_df.to_csv("datasets/test_anomaly.csv", index=False)
print("-> [OK] Đã tạo xong tập dữ liệu kiểm thử mẫu trong thư mục datasets/")