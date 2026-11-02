"""
src/config.py
Cấu hình các hằng số hệ thống, thông số phần cứng, mô hình và trạm Gateway.
Dự án: AIoT Wearable Water Anomaly Detection (TinyML)
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Đảm bảo KERAS_HOME trỏ vào thư mục cục bộ tránh lỗi quyền sandbox trên Windows
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))

@dataclass(frozen=True)
class SystemConfig:
    # Thông số lấy mẫu cảm biến IMU (ICM-42688-P)
    SAMPLING_RATE_HZ: int = 50         # Tần số lấy mẫu 50 Hz
    WINDOW_SIZE: int = 128             # 128 mẫu = 2.56 giây
    WINDOW_STRIDE: int = 64            # 64 mẫu = 50% overlap (1.28 giây)
    NUM_CHANNELS: int = 6              # Ax, Ay, Az, Gx, Gy, Gz
    
    # Bộ lọc số DSP (Butterworth thông thấp bậc 2)
    BUTTER_ORDER: int = 2
    BUTTER_CUTOFF_HZ: float = 10.0
    
    # Tham số học máy Autoencoder bán giám sát
    LATENT_DIM: int = 16               # Kích thước nút cổ chai (16, 1)
    BATCH_SIZE: int = 64
    EPOCHS: int = 60
    LEARNING_RATE: float = 1e-3
    ANOMALY_THRESHOLD_K: float = 3.5   # tau = mean + K * std (K trong khoảng 3.0 - 4.0)
    VOTING_CONSECUTIVE_WINDOWS: int = 3 # Yêu cầu 3 cửa sổ liên tiếp vượt ngưỡng để xác nhận IDR
    
    # Ngưỡng vật lý độ sâu & thời gian (MS5837-02BA)
    DEPTH_SURFACE_THRESHOLD_M: float = 0.15 # Dưới 15cm xem như ngoi mặt nước
    DEPTH_DANGER_THRESHOLD_M: float = 1.20  # Chìm sâu nguy hiểm (> 1.2m)
    MAX_SUBMERGED_TIME_SEC: float = 30.0   # Báo động đỏ khẩn cấp nếu chìm quá 30s
    
    # Giới hạn phần cứng Nordic nRF52840 (ARM Cortex-M4F)
    MAX_FLASH_SIZE_KB: float = 40.0        # Flash ROM cho model TinyML (< 40 KB, mục tiêu ~32.4 KB)
    MAX_TENSOR_ARENA_KB: int = 26          # Static RAM Tensor Arena (26 KB)
    MAX_LATENCY_MS: float = 15.0           # Độ trễ suy luận biên (< 15 ms)
    
    # Cấu hình đường dẫn dữ liệu & mô hình
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw_ethz"
    DATA_CUSTOM_DIR: Path = BASE_DIR / "data" / "custom_swimming"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    FIRMWARE_INC_DIR: Path = BASE_DIR / "firmware" / "include"
    
    # Cấu hình MQTT Broker cục bộ (Mosquitto trên RPi CM4)
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    TOPIC_TELEMETRY: str = "swim/telemetry"
    TOPIC_ALERT_LOCAL: str = "swim/alert/local"
    TOPIC_ALERT_HAPTIC: str = "swim/alert/haptic"
    TOPIC_ALERT_UI: str = "swim/alert/ui"
    TOPIC_ALERT_SMS: str = "swim/alert/sms"

CONFIG = SystemConfig()

