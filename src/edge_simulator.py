"""
src/edge_simulator.py
Mô phỏng bộ suy luận và máy trạng thái hữu hạn (FSM) chạy trên vi điều khiển Nordic nRF52840.
Tích hợp:
1. Suy luận TFLite INT8 chuẩn hóa Affine (Zero-dynamic memory logic)
2. Bộ biểu quyết đa cửa sổ trượt liên tiếp (Temporal Voting: 3 cửa sổ liên tiếp)
3. Máy trạng thái phân cấp an toàn 3 cấp (Vàng, Cam, Đỏ) tích hợp cảm biến áp suất độ sâu MS5837
"""
import os
import sys
import json
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Đảm bảo KERAS_HOME hợp lệ tránh lỗi quyền sandbox trên Windows
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import tensorflow as tf
from src.config import CONFIG, SystemConfig

class AlertState(Enum):
    NORMAL = 0          # Bình thường (Bơi ổn định, an toàn)
    LEVEL_1_YELLOW = 1  # Cấp 1 (Vàng): Cảnh báo rung haptic nhẹ cổ tay (0 - 5 giây)
    LEVEL_2_ORANGE = 2  # Cấp 2 (Cam): Nguy cơ, bắn gói tin LoRa 433MHz lên trạm bờ (5 - 15 giây)
    LEVEL_3_RED = 3     # Cấp 3 (Đỏ): Đuối nước khẩn cấp, kích hoạt còi sóng âm thủy âm 3-5kHz + còi bờ + SMS (> 15 giây)

class EdgeWearableNodeSimulator:
    def __init__(self, tflite_model_path: str, stats_json_path: str, node_id: str = "SWIM_NODE_01", lane_id: int = 2):
        self.node_id = node_id
        self.lane_id = lane_id

        # 1. Nạp mô hình TFLite INT8
        print(f"[*] Khởi tạo Edge Simulator từ mô hình: {tflite_model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 2. Lấy thông số lượng tử hóa Affine
        in_quant = self.input_details[0].get("quantization_parameters") or {}
        out_quant = self.output_details[0].get("quantization_parameters") or {}
        
        self.in_scale = float(self.input_details[0]["quantization"][0])
        self.in_zero_point = int(self.input_details[0]["quantization"][1])
        self.out_scale = float(self.output_details[0]["quantization"][0])
        self.out_zero_point = int(self.output_details[0]["quantization"][1])

        print(f"    - Input INT8 Quantization:  Scale={self.in_scale:.6f}, ZeroPoint={self.in_zero_point}")
        print(f"    - Output INT8 Quantization: Scale={self.out_scale:.6f}, ZeroPoint={self.out_zero_point}")

        # 3. Nạp ngưỡng bất thường thực nghiệm tau
        with open(stats_json_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            self.threshold_tau = float(stats["threshold_tau"])
        print(f"    - Ngưỡng phát hiện bất thường động tau: {self.threshold_tau:.6f}")

        # 4. Hàng đợi biểu quyết đa cửa sổ liên tiếp (3 cửa sổ liên tiếp)
        self.voting_queue = deque(maxlen=CONFIG.VOTING_CONSECUTIVE_WINDOWS)
        self.state = AlertState.NORMAL
        self.state_timer_sec = 0.0
        self.total_windows_processed = 0

    def infer_window_mse(self, window_fp32: np.ndarray) -> Tuple[float, float]:
        """
        Thực hiện suy luận INT8 giả lập đúng kiến trúc vi điều khiển ARM Cortex-M4F:
        1. Lượng tử hóa Affine Float32 -> INT8: q = round(val / scale) + zero_point
        2. TFLite Interpreter Invoke()
        3. Dequantize INT8 -> Float32: val = (q - zero_point) * scale
        4. Tính MSE toàn phương trung bình.
        Trả về (mse, latency_ms)
        """
        # Lượng tử hóa đầu vào
        t_start = time.perf_counter()
        window_int8 = np.clip(
            np.round(window_fp32 / self.in_scale) + self.in_zero_point, 
            -128, 127
        ).astype(np.int8)
        
        window_int8 = np.expand_dims(window_int8, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], window_int8)
        self.interpreter.invoke()
        
        output_int8 = self.interpreter.get_tensor(self.output_details[0]["index"])
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Giải lượng tử hóa đầu ra
        output_fp32 = (output_int8.astype(np.float32) - self.out_zero_point) * self.out_scale
        
        # Sai số tái tạo MSE trên cửa sổ (128, 6)
        mse = float(np.mean(np.square(window_fp32 - output_fp32[0])))
        return mse, latency_ms

    def update_tick(
        self, 
        window_fp32: np.ndarray, 
        current_depth_m: float, 
        is_surface_wrist: bool = False,
        delta_time_sec: float = 1.28
    ) -> Tuple[AlertState, Dict[str, Any]]:
        """
        Cập nhật 1 chu kỳ logic FSM trên vi điều khiển:
        - window_fp32: Cửa sổ IMU 128 mẫu đã chuẩn hóa
        - current_depth_m: Độ sâu đo từ cảm biến áp suất MS5837
        - is_surface_wrist: Tiếp điểm nước phát hiện tay nhô lên mặt nước
        - delta_time_sec: Thời gian trôi qua mỗi bước trượt (64 mẫu / 50Hz = 1.28s)
        """
        self.total_windows_processed += 1
        mse, latency_ms = self.infer_window_mse(window_fp32)
        
        # 1. So khớp ngưỡng dị thường
        is_window_anomalous = (mse > self.threshold_tau)
        self.voting_queue.append(is_window_anomalous)

        # 2. Biểu quyết đa cửa sổ (Temporal Voting: đủ 3/3 cửa sổ)
        is_idr_confirmed = (
            len(self.voting_queue) == CONFIG.VOTING_CONSECUTIVE_WINDOWS and 
            all(self.voting_queue)
        )

        # 3. Logic tích hợp cảm biến áp suất MS5837 và thời gian
        is_deep_danger = (current_depth_m > CONFIG.DEPTH_DANGER_THRESHOLD_M)

        if is_idr_confirmed or is_deep_danger:
            self.state_timer_sec += delta_time_sec
        else:
            # Nếu tín hiệu trở về bình thường, giảm dần bộ đếm thời gian
            self.state_timer_sec = max(0.0, self.state_timer_sec - delta_time_sec)
            if self.state_timer_sec == 0.0:
                self.state = AlertState.NORMAL

        # 4. Chuyển đổi trạng thái máy FSM (Phân cấp an toàn 3 cấp)
        if self.state_timer_sec > 15.0 or (is_deep_danger and self.state_timer_sec > CONFIG.MAX_SUBMERGED_TIME_SEC):
            self.state = AlertState.LEVEL_3_RED
        elif self.state_timer_sec > 5.0:
            self.state = AlertState.LEVEL_2_ORANGE
        elif self.state_timer_sec > 0.0:
            self.state = AlertState.LEVEL_1_YELLOW
        else:
            self.state = AlertState.NORMAL

        # Gói tin telemetry gửi về trạm bờ nếu phát LoRa bộc phát hoặc sóng âm
        telemetry = {
            "node_id": self.node_id,
            "lane_id": self.lane_id,
            "state": self.state.name,
            "alert_level": self.state.value,
            "mse": round(mse, 6),
            "tau": round(self.threshold_tau, 6),
            "depth_m": round(current_depth_m, 2),
            "is_surface_wrist": is_surface_wrist,
            "state_timer_sec": round(self.state_timer_sec, 2),
            "idr_detected": is_idr_confirmed,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        }
        return self.state, telemetry


def run_benchmark_simulation(
    tflite_path: str,
    stats_path: str,
    normal_data_path: str,
    anomaly_data_path: str
) -> Dict[str, Any]:
    """
    Chạy chuỗi kiểm thử giả lập hoàn chỉnh:
    10 cửa sổ bơi bình thường -> 15 cửa sổ đuối nước (IDR) -> 5 cửa sổ hồi phục.
    Quan sát sự chuyển biến trạng thái từ NORMAL -> VÀNG -> CAM -> ĐỎ -> Hồi phục.
    """
    sim = EdgeWearableNodeSimulator(tflite_path, stats_path)
    normal_windows = np.load(normal_data_path).astype(np.float32)
    anomaly_windows = np.load(anomaly_data_path).astype(np.float32)

    results = []
    print("\n" + "="*80)
    print(" BẮT ĐẦU MÔ PHỎNG GIẢ LẬP SUY LUẬN BIÊN & FSM CẢNH BÁO 3 CẤP (ARM CORTEX-M4F)")
    print("="*80)

    # 1. Pha 1: Bơi lội bình thường (cửa sổ 0 -> 9, độ sâu 0.3m, tay nhô mặt nước)
    print("\n--- PHA 1: BƠI LỘI BÌNH THƯỜNG (10 CỬA SỔ) ---")
    for i in range(min(10, len(normal_windows))):
        state, telem = sim.update_tick(normal_windows[i], current_depth_m=0.30, is_surface_wrist=True)
        results.append(telem)
        print(f"[T={i*1.28:5.1f}s] Window {i+1:02d} | MSE={telem['mse']:.5f} (tau={telem['tau']:.5f}) | Depth={telem['depth_m']}m | State={telem['state']:14s} | Latency={telem['latency_ms']:.2f}ms")

    # 2. Pha 2: Xuất hiện phản xạ đuối nước IDR (cửa sổ 10 -> 24, độ sâu chìm dần từ 0.5m -> 1.5m)
    print("\n--- PHA 2: PHẢN XẠ ĐUỐI NƯỚC TỰ NHIÊN (IDR) & CHÌM DẦN (15 CỬA SỔ) ---")
    for i in range(min(15, len(anomaly_windows))):
        depth = min(1.60, 0.40 + (i * 0.08))
        surface = False if depth > 0.6 else True
        state, telem = sim.update_tick(anomaly_windows[i], current_depth_m=depth, is_surface_wrist=surface)
        results.append(telem)
        print(f"[T={(10+i)*1.28:5.1f}s] Window {10+i+1:02d} | MSE={telem['mse']:.5f} (tau={telem['tau']:.5f}) | Depth={telem['depth_m']}m | State={telem['state']:14s} | Timer={telem['state_timer_sec']:4.1f}s")

    # 3. Pha 3: Nạn nhân được cứu hộ kéo lên mặt nước (5 cửa sổ bình thường, độ sâu 0.1m)
    print("\n--- PHA 3: CỨU HỘ TIẾP CẬN VÀ HỒI PHỤC (5 CỬA SỔ) ---")
    for i in range(5):
        idx = (i + 15) % len(normal_windows)
        state, telem = sim.update_tick(normal_windows[idx], current_depth_m=0.10, is_surface_wrist=True)
        results.append(telem)
        print(f"[T={(25+i)*1.28:5.1f}s] Window {25+i+1:02d} | MSE={telem['mse']:.5f} (tau={telem['tau']:.5f}) | Depth={telem['depth_m']}m | State={telem['state']:14s} | Timer={telem['state_timer_sec']:4.1f}s")

    print("="*80)
    print(" KẾT THÚC KIỂM THỬ GIẢ LẬP FSM BIÊN THÀNH CÔNG!")
    print("="*80 + "\n")
    return {"timeline": results}

if __name__ == "__main__":
    tflite = str(CONFIG.MODELS_DIR / "autoencoder_int8.tflite")
    stats = str(CONFIG.MODELS_DIR / "stats.json")
    normal_data = str(CONFIG.DATA_PROCESSED_DIR / "normal_swimming.npy")
    anomaly_data = str(CONFIG.DATA_PROCESSED_DIR / "anomaly_swimming.npy")
    run_benchmark_simulation(tflite, stats, normal_data, anomaly_data)
