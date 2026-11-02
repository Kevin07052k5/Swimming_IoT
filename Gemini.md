# HƯỚNG DẪN KỸ THUẬT PHÁT TRIỂN & PROMPT MẪU CHO ANTIGRAVITY IDE
## DỰ ÁN: AIoT WEARABLE WATER ANOMALY DETECTION (TINYML)

---

### 1. MASTER PROMPT CHO ANTIGRAVITY IDE

Khi tương tác với Antigravity IDE hoặc Agentic Coding Assistant, sử dụng câu lệnh tổng thể (Master Prompt) chuẩn kỹ sư AI dưới đây:

```markdown
Bạn là Kiến trúc sư Hệ thống AIoT và Chuyên gia TinyML cấp cao. 
Nhiệm vụ của bạn là triển khai mã nguồn Python chuẩn công nghiệp cho đề tài: "XÂY DỰNG THIẾT KẾ MÔ HÌNH AIoT PHÁT HIỆN BẤT THƯỜNG KHI BƠI (TINYML)".

YÊU CẦU KỸ THUẬT CỐT LÕI:
1. Tiếp nhận cấu trúc dữ liệu IMU 6 trục (Gia tốc Ax, Ay, Az; Con quay Gx, Gy, Gz) kế thừa định dạng từ ETH Zurich Swimming Dataset (Brunner et al., ISWC 2019).
2. Xây dựng DSP Pipeline chuẩn: Lọc số Butterworth bậc 2 (fc=10Hz, fs=50Hz) sử dụng Scipy, chuẩn hóa Z-Score, chia cửa sổ trượt 128 mẫu (50% overlap).
3. Thiết kế mạng học bán giám sát 1D-CNN Autoencoder bằng TensorFlow/Keras: Chỉ huấn luyện trên mẫu bơi bình thường, nén qua nút cổ chai (16, 1), tái tạo lại ma trận (128, 6).
4. Thực hiện lượng tử hóa toàn phần INT8 (TFLite Post-Training Quantization với Representative Dataset), tự động xuất ra mảng C byte array (model_data.h) để nạp thẳng lên vi điều khiển ARM Cortex-M4F (nRF52840).
5. Xây dựng module giả lập suy luận biên tích hợp logic kiểm định 3 cửa sổ liên tiếp (Threshold = mean + 3.5*std).
6. Viết dịch vụ Trạm bờ hồ Gateway bằng Python: Kết nối Mosquitto MQTT Broker, quản lý bản đồ nhiệt làn bơi và kích hoạt cảnh báo 3 cấp (Còi bờ, Smartwatch, SMS SIM7600).

Mã nguồn phải đạt chuẩn Clean Code, gán Type Hinting đầy đủ, cấu trúc module hóa phân tầng, có docstrings và sẵn sàng chạy thử nghiệm ngay lập tức.
```

---

### 2. CẤU TRÚC DỰ ÁN TRÊN ANTIGRAVITY IDE

```
aiot_swimming_tinyml/
│
├── data/
│   ├── raw_ethz/              # Dữ liệu IMU gốc từ ETH Zurich (.csv/.mat)
│   ├── custom_swimming/       # Dữ liệu thí nghiệm an toàn 4 lớp thu thập thực tế
│   └── processed/             # Mảng numpy chuẩn hóa đã tạo cửa sổ (.npz)
│
├── models/
│   ├── autoencoder_fp32.h5    # Mô hình huấn luyện gốc Keras
│   ├── autoencoder_int8.tflite# Mô hình sau lượng tử hóa INT8
│   └── model_data.h           # Mảng C phục vụ nhúng trực tiếp vào firmware C++
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Tham số hệ thống, tần số, ngưỡng FSM
│   ├── dsp_pipeline.py        # Tiền xử lý: Lọc Butterworth, Z-Score, Sliding Window
│   ├── model_autoencoder.py   # Kiến trúc mạng 1D-CNN Autoencoder Keras
│   ├── train.py               # Script huấn luyện bán giám sát & tính ngưỡng động
│   ├── quantize_tflite.py     # Lượng tử hóa INT8 & xuất C-Array
│   ├── edge_simulator.py      # Giả lập FSM suy luận biên 3 cấp tại vi điều khiển
│   └── gateway_service.py     # Dịch vụ MQTT Gateway RPi CM4 & Kênh cảnh báo
│
├── requirements.txt           # Thư viện phụ thuộc
└── README.md                  # Hướng dẫn chạy
```

---

### 3. MÃ NGUỒN TRIỂN KHAI CHI TIẾT (PYTHON)

#### 3.1. Cấu Hình Tham Số Hệ Thống (`src/config.py`)

```python
"""
src/config.py
Cấu hình các hằng số hệ thống, thông số phần cứng, mô hình và trạm Gateway.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class SystemConfig:
    # Thông số lấy mẫu cảm biến IMU
    SAMPLING_RATE_HZ: int = 50
    WINDOW_SIZE: int = 128             # 128 mẫu = 2.56 giây
    WINDOW_STRIDE: int = 64            # 64 mẫu = 50% overlap
    NUM_CHANNELS: int = 6              # Ax, Ay, Az, Gx, Gy, Gz
    
    # Bộ lọc số DSP
    BUTTER_ORDER: int = 2
    BUTTER_CUTOFF_HZ: float = 10.0
    
    # Tham số học máy Autoencoder
    LATENT_DIM: int = 16
    BATCH_SIZE: int = 64
    EPOCHS: int = 60
    LEARNING_RATE: float = 1e-3
    ANOMALY_THRESHOLD_K: float = 3.5   # tau = mean + K * std
    VOTING_CONSECUTIVE_WINDOWS: int = 3
    
    # Ngưỡng vật lý độ sâu & thời gian (MS5837-02BA)
    DEPTH_SURFACE_THRESHOLD_M: float = 0.15 # Dưới 15cm xem như ngoi mặt nước
    DEPTH_DANGER_THRESHOLD_M: float = 1.20  # Chìm sâu nguy hiểm
    MAX_SUBMERGED_TIME_SEC: float = 30.0   # Báo động đỏ nếu chìm quá 30s
    
    # Cấu hình MQTT Broker cục bộ
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    TOPIC_TELEMETRY: str = "swim/telemetry"
    TOPIC_ALERT_LOCAL: str = "swim/alert/local"
    TOPIC_ALERT_HAPTIC: str = "swim/alert/haptic"
    TOPIC_ALERT_UI: str = "swim/alert/ui"
    TOPIC_ALERT_SMS: str = "swim/alert/sms"

CONFIG = SystemConfig()
```

#### 3.2. Chuỗi Tiền Xử Lý Tín Hiệu DSP (`src/dsp_pipeline.py`)

```python
"""
src/dsp_pipeline.py
Triển khai bộ lọc Butterworth bậc 2, chuẩn hóa Z-Score và phân tách cửa sổ trượt.
"""
import numpy as np
from scipy import signal
from typing import Tuple
from src.config import CONFIG

class SwimmingDSPPipeline:
    def __init__(self, config=CONFIG):
        self.cfg = config
        # Khởi tạo ma trận hệ số bộ lọc số Butterworth bậc 2
        nyquist = 0.5 * self.cfg.SAMPLING_RATE_HZ
        normal_cutoff = self.cfg.BUTTER_CUTOFF_HZ / nyquist
        self.b, self.a = signal.butter(
            self.cfg.BUTTER_ORDER, 
            normal_cutoff, 
            btype='low', 
            analog=False
        )

    def apply_butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Lọc thông thấp Butterworth 2nd order trên 6 trục IMU.
        data shape: (N, 6)
        """
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            # Lọc hai chiều (forward-backward) để triệt tiêu độ trễ pha
            filtered[:, ch] = signal.filtfilt(self.b, self.a, data[:, ch])
        return filtered

    def z_score_normalize(self, data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chuẩn hóa Z-Score độc lập cho từng kênh cảm biến.
        """
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        normalized = (data - means) / (stds + eps)
        return normalized, means, stds

    def create_sliding_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Phân tách dữ liệu liên tục thành các cửa sổ kích thước (128, 6) với bước nhảy 64.
        """
        num_samples = data.shape[0]
        windows = []
        for start_idx in range(0, num_samples - self.cfg.WINDOW_SIZE + 1, self.cfg.WINDOW_STRIDE):
            end_idx = start_idx + self.cfg.WINDOW_SIZE
            windows.append(data[start_idx:end_idx, :])
        return np.array(windows, dtype=np.float32)

    def process_raw_stream(self, raw_imu: np.ndarray) -> np.ndarray:
        """
        Pipeline xử lý toàn vẹn từ tín hiệu thô đến tensor đầu vào cho AI.
        """
        filtered = self.apply_butterworth_filter(raw_imu)
        normalized, _, _ = self.z_score_normalize(filtered)
        windows = self.create_sliding_windows(normalized)
        return windows
```

#### 3.3. Kiến Trúc Mạng 1D-CNN Autoencoder (`src/model_autoencoder.py`)

```python
"""
src/model_autoencoder.py
Định nghĩa kiến trúc mạng 1D-CNN Autoencoder bán giám sát tối ưu cho vi điều khiển.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import CONFIG

def build_1d_cnn_autoencoder(config=CONFIG) -> Model:
    """
    Xây dựng mạng 1D-CNN Autoencoder đối xứng:
    Input: (128, 6)
    Encoder -> MaxPool giảm chiều -> Bottleneck (16, 1) -> Decoder UpSampling -> Output (128, 6)
    """
    input_shape = (config.WINDOW_SIZE, config.NUM_CHANNELS)
    inputs = layers.Input(shape=input_shape, name="imu_input")

    # ==================== ENCODER ====================
    # Tầng 1: 32 bộ lọc, trích xuất đặc trưng tần số thấp
    x = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)  # -> (64, 32)

    # Tầng 2: 16 bộ lọc
    x = layers.Conv1D(filters=16, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)  # -> (32, 16)

    # Tầng 3: 8 bộ lọc
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)   # -> (16, 8)

    # ==================== BOTTLENECK ====================
    # Nén về không gian vector ẩn cực đại
    bottleneck = layers.Conv1D(filters=1, kernel_size=3, padding="same", activation="relu", name="bottleneck")(x) # -> (16, 1)

    # ==================== DECODER ====================
    # Giải nén tầng 1
    x = layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding="same", activation="relu")(bottleneck) # -> (32, 8)

    # Giải nén tầng 2
    x = layers.Conv1DTranspose(filters=16, kernel_size=5, strides=2, padding="same", activation="relu")(x)         # -> (64, 16)

    # Giải nén tầng 3
    x = layers.Conv1DTranspose(filters=32, kernel_size=5, strides=2, padding="same", activation="relu")(x)         # -> (128, 32)

    # Tầng tái tạo đầu ra: Khôi phục lại chính xác 6 kênh
    outputs = layers.Conv1D(filters=config.NUM_CHANNELS, kernel_size=5, padding="same", activation="linear", name="reconstruction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TinyML_Swimming_1D_CNN_AE")
    return model

if __name__ == "__main__":
    model = build_1d_cnn_autoencoder()
    model.summary()
```

#### 3.4. Huấn Luyện Bán Giám Sát & Tính Ngưỡng Dị Thường (`src/train.py`)

```python
"""
src/train.py
Quy trình huấn luyện mạng CHỈ trên dữ liệu bơi lội bình thường và tính ngưỡng tau.
"""
import os
import json
import numpy as np
import tensorflow as tf
from src.config import CONFIG
from src.model_autoencoder import build_1d_cnn_autoencoder

def train_autoencoder(normal_data_path: str, model_save_path: str, stats_save_path: str):
    """
    Huấn luyện bán giám sát.
    normal_data_path: file .npy chứa mảng cửa sổ bơi chuẩn (N, 128, 6)
    """
    print(f"[*] Đang nạp tập dữ liệu bơi bình thường từ: {normal_data_path}")
    X_train = np.load(normal_data_path).astype(np.float32)
    print(f"[*] Kích thước tập huấn luyện: {X_train.shape}")

    # Phân chia tập Validation
    val_split = int(0.85 * len(X_train))
    train_data = X_train[:val_split]
    val_data = X_train[val_split:]

    model = build_1d_cnn_autoencoder()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE),
        loss="mse"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]

    print("[*] Bắt đầu quá trình huấn luyện Autoencoder bán giám sát...")
    history = model.fit(
        train_data, train_data,
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE,
        validation_data=(val_data, val_data),
        callbacks=callbacks,
        verbose=1
    )

    # Lưu mô hình Keras FP32
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[+] Đã lưu mô hình Keras FP32 tại: {model_save_path}")

    # Tính toán phân phối sai số MSE trên tập kiểm chuẩn
    val_preds = model.predict(val_data)
    # MSE trên từng cửa sổ: shape (M,)
    val_mse = np.mean(np.square(val_data - val_preds), axis=(1, 2))

    mean_mse = float(np.mean(val_mse))
    std_mse = float(np.std(val_mse))
    threshold_tau = mean_mse + (CONFIG.ANOMALY_THRESHOLD_K * std_mse)

    stats = {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "k_factor": CONFIG.ANOMALY_THRESHOLD_K,
        "threshold_tau": threshold_tau
    }

    with open(stats_save_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"[+] Thống kê sai số tái tạo bình thường:")
    print(f"    - Mean MSE: {mean_mse:.6f}")
    print(f"    - Std MSE:  {std_mse:.6f}")
    print(f"    - Ngưỡng dị thường quyết định tau: {threshold_tau:.6f}")
```

#### 3.5. Lượng Tử Hóa INT8 & Xuất C-Array Nhúng (`src/quantize_tflite.py`)

```python
"""
src/quantize_tflite.py
Thực hiện Full Integer INT8 Quantization (TFLite) và xuất mảng C byte array (model_data.h).
"""
import os
import subprocess
import numpy as np
import tensorflow as tf
from src.config import CONFIG

def representative_data_gen(sample_data: np.ndarray):
    """
    Tạo dữ liệu đại diện phục vụ cân chỉnh dải giá trị động INT8.
    """
    for i in range(min(300, len(sample_data))):
        yield [sample_data[i:i+1]]

def quantize_to_tflite_int8(saved_model_path: str, sample_data_path: str, output_tflite_path: str, output_header_path: str):
    print("[*] Đang khởi tạo bộ chuyển đổi TFLite Converter...")
    model = tf.keras.models.load_model(saved_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Kích hoạt tối ưu hóa lượng tử hóa
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Đọc dữ liệu đại diện
    calibration_data = np.load(sample_data_path).astype(np.float32)
    converter.representative_dataset = lambda: representative_data_gen(calibration_data)
    
    # Ép kiểu toàn bộ tensor đầu vào và đầu ra về INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    
    file_size_kb = len(tflite_quant_model) / 1024.0
    print(f"[+] Lượng tử hóa INT8 thành công!")
    print(f"    - Kích thước file TFLite: {file_size_kb:.2f} KB (Mục tiêu: <40 KB)")

    # Chuyển đổi file .tflite sang file header C (.h) sử dụng xxd
    try:
        print("[*] Đang xuất mảng C qua xxd...")
        with open(output_header_path, "w") as header_file:
            header_content = subprocess.check_output(["xxd", "-i", output_tflite_path]).decode("utf-8")
            header_file.write("// Auto-generated TinyML Model for ARM Cortex-M4F\n")
            header_file.write("#ifndef SWIMMING_MODEL_DATA_H\n#define SWIMMING_MODEL_DATA_H\n\n")
            header_file.write(header_content)
            header_file.write("\n#endif // SWIMMING_MODEL_DATA_H\n")
        print(f"[+] Đã tạo file header nhúng: {output_header_path}")
    except Exception as e:
        print(f"[!] Chú ý: Không tìm thấy tiện ích xxd ({e}), tạo mảng byte bằng Python thuần:")
        bytes_array = ", ".join([f"0x{b:02x}" for b in tflite_quant_model])
        with open(output_header_path, "w") as f:
            f.write(f"const unsigned char g_swimming_model_data[] = {{{bytes_array}}};\n")
            f.write(f"const unsigned int g_swimming_model_data_len = {len(tflite_quant_model)};\n")
        print(f"[+] Đã tạo file header C thay thế tại: {output_header_path}")
```

#### 3.6. Giả Lập Máy Trạng Thái Cảnh Báo 3 Cấp Tại Biên (`src/edge_simulator.py`)

```python
"""
src/edge_simulator.py
Mô phỏng bộ suy luận và máy trạng thái hữu hạn (FSM) chạy trên vi điều khiển nRF52840.
"""
import json
import numpy as np
import tensorflow as tf
from collections import deque
from enum import Enum
from src.config import CONFIG

class AlertState(Enum):
    NORMAL = 0
    LEVEL_1_YELLOW = 1  # Cảnh báo rung nhẹ cổ tay (0 - 5s)
    LEVEL_2_ORANGE = 2  # Nguy cơ: Bắn gói tin LoRa trạm bờ (5 - 15s)
    LEVEL_3_RED = 3     # Đỏ khẩn cấp: Sóng âm + LoRa + Còi hồ (>15s)

class EdgeWearableNodeSimulator:
    def __init__(self, tflite_model_path: str, stats_json_path: str):
        # Nạp mô hình TFLite
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Nạp thông số lượng tử hóa INT8
        self.in_scale, self.in_zero_point = self.input_details[0]['quantization']
        self.out_scale, self.out_zero_point = self.output_details[0]['quantization']

        # Nạp ngưỡng dị thường
        with open(stats_json_path, "r") as f:
            stats = json.load(f)
            self.threshold_tau = stats["threshold_tau"]

        # Hàng đợi biểu quyết 3 cửa sổ liên tiếp
        self.voting_queue = deque(maxlen=CONFIG.VOTING_CONSECUTIVE_WINDOWS)
        self.state = AlertState.NORMAL
        self.state_timer_sec = 0.0

    def infer_window_mse(self, window_fp32: np.ndarray) -> float:
        """
        Thực hiện suy luận INT8 giả lập đúng kiến trúc vi điều khiển.
        """
        # Quantize Float32 -> INT8
        window_int8 = np.clip(np.round(window_fp32 / self.in_scale) + self.in_zero_point, -128, 127).astype(np.int8)
        window_int8 = np.expand_dims(window_int8, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], window_int8)
        self.interpreter.invoke()
        output_int8 = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize INT8 -> Float32 để tính sai số MSE
        output_fp32 = (output_int8.astype(np.float32) - self.out_zero_point) * self.out_scale
        mse = float(np.mean(np.square(window_fp32 - output_fp32[0])))
        return mse

    def update_tick(self, window_fp32: np.ndarray, current_depth_m: float, delta_time_sec: float = 1.28) -> Tuple[AlertState, dict]:
        """
        Cập nhật chu kỳ logic FSM.
        """
        mse = self.infer_window_mse(window_fp32)
        is_window_anomalous = (mse > self.threshold_tau)
        self.voting_queue.append(is_window_anomalous)

        # Kiểm định đa cửa sổ liên tiếp
        is_idr_confirmed = len(self.voting_queue) == CONFIG.VOTING_CONSECUTIVE_WINDOWS and all(self.voting_queue)
        
        # Logic chuyển trạng thái máy FSM
        if is_idr_confirmed or (current_depth_m > CONFIG.DEPTH_DANGER_THRESHOLD_M):
            self.state_timer_sec += delta_time_sec
        else:
            self.state_timer_sec = max(0.0, self.state_timer_sec - delta_time_sec)
            if self.state_timer_sec == 0:
                self.state = AlertState.NORMAL

        if self.state_timer_sec > 15.0 or (current_depth_m > CONFIG.DEPTH_DANGER_THRESHOLD_M and self.state_timer_sec > CONFIG.MAX_SUBMERGED_TIME_SEC):
            self.state = AlertState.LEVEL_3_RED
        elif self.state_timer_sec > 5.0:
            self.state = AlertState.LEVEL_2_ORANGE
        elif self.state_timer_sec > 0.0:
            self.state = AlertState.LEVEL_1_YELLOW

        telemetry = {
            "state": self.state.name,
            "mse": mse,
            "tau": self.threshold_tau,
            "depth_m": current_depth_m,
            "timer": self.state_timer_sec,
            "idr_detected": is_idr_confirmed
        }
        return self.state, telemetry
```

#### 3.7. Dịch Vụ Điều Phối Trạm Bờ Hồ Gateway MQTT (`src/gateway_service.py`)

```python
"""
src/gateway_service.py
Dịch vụ trạm bờ hồ Raspberry Pi CM4: Điều phối bản tin MQTT Broker cục bộ,
phát cảnh báo đa kênh (Còi, Haptic Smartwatch, Web Dashboard, SMS SIM7600).
"""
import json
import time
import paho.mqtt.client as mqtt
from src.config import CONFIG

class PoolsideGatewayService:
    def __init__(self):
        self.client = mqtt.Client(client_id="EdgeGateway_CM4", clean_session=True)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print(f"[+] Gateway kết nối MQTT Broker thành công với mã kết quả: {rc}")
        # Đăng ký nhận toàn bộ telemetry và cảnh báo từ các node đeo
        self.client.subscribe(f"{CONFIG.TOPIC_TELEMETRY}/#", qos=1)

    def trigger_poolside_alarm(self, node_id: str, lane_id: int):
        """Kích hoạt còi hú & đèn chớp bờ hồ"""
        payload = json.dumps({"action": "SIREN_ON", "node": node_id, "lane": lane_id, "ts": time.time()})
        self.client.publish(CONFIG.TOPIC_ALERT_LOCAL, payload, qos=2)
        print(f"[!] [LOCAL SIREN] Bật còi hú công suất lớn tại hồ bơi! (Làn {lane_id})")

    def notify_lifeguard_smartwatch(self, node_id: str, lane_id: int, level: str):
        """Gửi xung rung tới Smartwatch cứu hộ"""
        payload = json.dumps({"vibration": "HIGH", "level": level, "lane": lane_id, "node": node_id})
        self.client.publish(CONFIG.TOPIC_ALERT_HAPTIC, payload, qos=2)
        print(f"[!] [SMARTWATCH] Gửi rung khẩn cấp tới cứu hộ: Mức {level} - Làn {lane_id}")

    def send_fail_safe_sms(self, node_id: str, lane_id: int):
        """Giả lập điều khiển module SIM7600 bắn tin nhắn SMS khẩn cấp"""
        print(f"[!] [SMS 4G FAIL-SAFE] Gửi tin nhắn khẩn cấp tới Đội trưởng Y tế: "
              f"PHÁT HIỆN ĐUỐI NƯỚC CẤP 3 TẠI LÀN {lane_id} - THIẾT BỊ {node_id}!")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            node_id = payload.get("node_id", "NODE_01")
            lane_id = payload.get("lane_id", 3)
            state = payload.get("state", "NORMAL")

            if state == "LEVEL_1_YELLOW":
                print(f"[i] Cảnh báo Cấp 1 (Vàng) - Vòng {node_id} rung nhẹ.")
            elif state == "LEVEL_2_ORANGE":
                print(f"[!] Cảnh báo Cấp 2 (Cam) - Bật đèn cam trạm bờ làn {lane_id}.")
                self.notify_lifeguard_smartwatch(node_id, lane_id, "CAM")
            elif state == "LEVEL_3_RED":
                print(f"[ALERT RED] BÁO ĐỘNG CẤP 3 - ĐUỐI NƯỚC XÁC NHẬN TẠI LÀN {lane_id}!")
                self.trigger_poolside_alarm(node_id, lane_id)
                self.notify_lifeguard_smartwatch(node_id, lane_id, "ĐỎ")
                self.send_fail_safe_sms(node_id, lane_id)

            # Cập nhật tọa độ nhiệt lên Web Dashboard
            self.client.publish(CONFIG.TOPIC_ALERT_UI, msg.payload, qos=0)

        except Exception as e:
            print(f"[!] Lỗi giải mã bản tin MQTT: {e}")

    def run(self):
        print(f"[*] Gateway đang lắng nghe trên cổng {CONFIG.MQTT_BROKER_PORT}...")
        self.client.connect(CONFIG.MQTT_BROKER_HOST, CONFIG.MQTT_BROKER_PORT, keepalive=60)
        self.client.loop_forever()

if __name__ == "__main__":
    gateway = PoolsideGatewayService()
    gateway.run()
```

---

### 4. BỘ YÊU CẦU THƯ VIỆN (`requirements.txt`)

```txt
numpy>=1.22.0
scipy>=1.8.0
tensorflow>=2.10.0
paho-mqtt>=1.6.1
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

---

### 5. QUY TRÌNH THỰC THI TOÀN TRÌNH TRÊN ANTIGRAVITY IDE

1. **Khởi tạo Môi trường:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Hoặc venv\Scripts\activate trên Windows
   pip install -r requirements.txt
   ```
2. **Huấn luyện Mô hình & Lượng tử hóa:**
   - Đặt dữ liệu bơi bình thường chuẩn hóa vào thư mục `data/processed/normal_swimming.npy`.
   - Chạy tập lệnh huấn luyện:
     ```bash
     python -c "from src.train import train_autoencoder; train_autoencoder('data/processed/normal_swimming.npy', 'models/autoencoder_fp32.h5', 'models/stats.json')"
     ```
   - Chạy lượng tử hóa INT8 và trích xuất header C:
     ```bash
     python -c "from src.quantize_tflite import quantize_to_tflite_int8; quantize_to_tflite_int8('models/autoencoder_fp32.h5', 'data/processed/normal_swimming.npy', 'models/autoencoder_int8.tflite', 'models/model_data.h')"
     ```
3. **Chạy Thử Nghiệm Giả Lập Biên & Gateway:**
   - Mở Terminal 1 khởi chạy Local MQTT Broker: `mosquitto -v`
   - Mở Terminal 2 chạy Gateway Service: `python -m src.gateway_service`
   - Mở Terminal 3 kích hoạt bộ giả lập suy luận nút biên: `python -m src.edge_simulator`