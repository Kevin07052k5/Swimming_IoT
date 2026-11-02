"""
src/dsp_pipeline.py
Triển khai chuỗi tiền xử lý tín hiệu động học (DSP Pipeline):
1. Lọc số thông thấp Butterworth bậc 2 (fc=10Hz, fs=50Hz)
2. Chuẩn hóa phân phối Z-Score trên 6 trục IMU
3. Phân tách cửa sổ trượt (Sliding Window: W=128, Overlap=50%)
"""
from typing import Tuple, Optional
import numpy as np
from scipy import signal
from src.config import CONFIG, SystemConfig

class SwimmingDSPPipeline:
    """
    Chuỗi xử lý tín hiệu số offline và batch cho dữ liệu bơi lội 6 trục IMU.
    """
    def __init__(self, config: SystemConfig = CONFIG):
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
        Sử dụng filtfilt hai chiều (forward-backward) để triệt tiêu độ trễ pha.
        """
        if data.shape[0] <= 3 * max(len(self.a), len(self.b)):
            # Dữ liệu quá ngắn cho filtfilt, dùng lfilter
            filtered = signal.lfilter(self.b, self.a, data, axis=0)
            return filtered.astype(np.float32)

        filtered = np.zeros_like(data, dtype=np.float32)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.filtfilt(self.b, self.a, data[:, ch])
        return filtered

    def z_score_normalize(
        self, 
        data: np.ndarray, 
        means: Optional[np.ndarray] = None, 
        stds: Optional[np.ndarray] = None, 
        eps: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chuẩn hóa Z-Score độc lập cho từng kênh cảm biến: z = (x - mean) / (std + eps).
        Nếu means/stds không truyền vào, sẽ tính trực tiếp trên tập data hiện tại.
        """
        if means is None:
            means = np.mean(data, axis=0)
        if stds is None:
            stds = np.std(data, axis=0)
        
        normalized = (data - means) / (stds + eps)
        return normalized.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)

    def create_sliding_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Phân tách dữ liệu liên tục thành các cửa sổ kích thước (128, 6) với bước nhảy 64.
        """
        num_samples = data.shape[0]
        if num_samples < self.cfg.WINDOW_SIZE:
            return np.empty((0, self.cfg.WINDOW_SIZE, self.cfg.NUM_CHANNELS), dtype=np.float32)

        windows = []
        for start_idx in range(0, num_samples - self.cfg.WINDOW_SIZE + 1, self.cfg.WINDOW_STRIDE):
            end_idx = start_idx + self.cfg.WINDOW_SIZE
            windows.append(data[start_idx:end_idx, :])
        return np.array(windows, dtype=np.float32)

    def process_raw_stream(
        self, 
        raw_imu: np.ndarray, 
        means: Optional[np.ndarray] = None, 
        stds: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline xử lý toàn vẹn từ tín hiệu thô đến tensor đầu vào cho AI:
        raw_imu (N, 6) -> filter -> z-score -> windows (M, 128, 6)
        """
        filtered = self.apply_butterworth_filter(raw_imu)
        normalized, computed_means, computed_stds = self.z_score_normalize(filtered, means=means, stds=stds)
        windows = self.create_sliding_windows(normalized)
        return windows, computed_means, computed_stds


class RealTimeStreamingDSP:
    """
    Mô phỏng bộ lọc trực tuyến IIR từng mẫu một (Direct Form II Transposed),
    tương đương thuật toán DSP thực thi trên ARM Cortex-M4F bằng C++.
    """
    def __init__(self, config: SystemConfig = CONFIG):
        self.cfg = config
        nyquist = 0.5 * self.cfg.SAMPLING_RATE_HZ
        normal_cutoff = self.cfg.BUTTER_CUTOFF_HZ / nyquist
        self.b, self.a = signal.butter(
            self.cfg.BUTTER_ORDER, 
            normal_cutoff, 
            btype='low', 
            analog=False
        )
        # Khởi tạo trạng thái trễ bộ lọc IIR cho 6 kênh: state [channels, order]
        self.zi = np.zeros((self.cfg.NUM_CHANNELS, self.cfg.BUTTER_ORDER), dtype=np.float32)
        # Bộ đệm cửa sổ trượt (128, 6)
        self.window_buffer = np.zeros((self.cfg.WINDOW_SIZE, self.cfg.NUM_CHANNELS), dtype=np.float32)
        self.sample_count = 0

    def process_sample(self, raw_sample: np.ndarray, means: np.ndarray, stds: np.ndarray) -> Optional[np.ndarray]:
        """
        Nạp 1 mẫu IMU 6 trục (50Hz), lọc số và chuẩn hóa.
        Trả về cửa sổ đầy đủ (128, 6) khi đạt đủ số mẫu, nếu chưa trả về None.
        """
        filtered_sample = np.zeros(self.cfg.NUM_CHANNELS, dtype=np.float32)
        for c in range(self.cfg.NUM_CHANNELS):
            y, self.zi[c] = signal.lfilter(self.b, self.a, [raw_sample[c]], zi=self.zi[c])
            filtered_sample[c] = y[0]

        # Chuẩn hóa Z-Score
        normalized_sample = (filtered_sample - means) / (stds + 1e-8)
        
        self.window_buffer[self.sample_count] = normalized_sample
        self.sample_count += 1

        if self.sample_count >= self.cfg.WINDOW_SIZE:
            full_window = self.window_buffer.copy()
            # Dịch bước trượt 50% (giữ lại 64 mẫu cuối)
            self.window_buffer[:self.cfg.WINDOW_SIZE - self.cfg.WINDOW_STRIDE] = (
                self.window_buffer[self.cfg.WINDOW_STRIDE:]
            )
            self.sample_count = self.cfg.WINDOW_SIZE - self.cfg.WINDOW_STRIDE
            return full_window

        return None

