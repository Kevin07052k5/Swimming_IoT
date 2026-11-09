"""
scripts/generate_demo_gif.py
Tạo file video ảnh động (Animated GIF Demo) mô phỏng trực quan toàn bộ quá trình:
1. Thu nhận tín hiệu IMU 6 trục và tính sai số tái tạo MSE
2. So sánh với ngưỡng động tau và kích hoạt biểu quyết FSM
3. Chuyển cấp báo động: NORMAL (Xanh) -> LEVEL 1 (Vàng) -> LEVEL 2 (Cam) -> LEVEL 3 (Đỏ)
4. Chấm điểm ma trận nhầm lẫn (Confusion Matrix) và F1-Score trực tiếp theo thời gian thực
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))

from src.config import CONFIG
from src.edge_simulator import EdgeWearableNodeSimulator, AlertState

def generate_demo_animation(
    output_gif_path: str = str(BASE_DIR / "demo_simulation.gif"),
    tflite_path: str = str(CONFIG.MODELS_DIR / "autoencoder_int8.tflite"),
    stats_path: str = str(CONFIG.MODELS_DIR / "stats.json"),
    normal_data_path: str = str(CONFIG.DATA_PROCESSED_DIR / "normal_swimming.npy"),
    anomaly_data_path: str = str(CONFIG.DATA_PROCESSED_DIR / "anomaly_swimming.npy"),
    num_frames: int = 36
):
    print("\n[*] Đang khởi tạo bộ tạo GIF video demo...")
    sim = EdgeWearableNodeSimulator(tflite_path, stats_path)
    normal_windows = np.load(normal_data_path).astype(np.float32)
    anomaly_windows = np.load(anomaly_data_path).astype(np.float32)
    tau = sim.threshold_tau

    # Tạo kịch bản diễn biến:
    # Frame 0-10: Bơi bình thường (Normal)
    # Frame 11-25: Xuất hiện phản xạ IDR và chìm dần (Yellow -> Orange -> Red)
    # Frame 26-35: Cứu hộ tiếp cận và đưa lên bờ (Hồi phục)
    frames_data = []
    
    # 1. Pha bơi thường
    for i in range(11):
        w = normal_windows[i % len(normal_windows)]
        state, telem = sim.update_tick(w, current_depth_m=0.25, is_surface_wrist=True)
        frames_data.append({
            "stage": "BƠI BÌNH THƯỜNG",
            "window": w,
            "telem": telem,
            "ground_truth": 0
        })

    # 2. Pha đuối nước IDR
    for i in range(15):
        w = anomaly_windows[i % len(anomaly_windows)]
        depth = min(1.55, 0.40 + i * 0.08)
        state, telem = sim.update_tick(w, current_depth_m=depth, is_surface_wrist=(depth < 0.6))
        frames_data.append({
            "stage": "PHẢN XẠ ĐUỐI NƯỚC (IDR)",
            "window": w,
            "telem": telem,
            "ground_truth": 1
        })

    # 3. Pha cứu hộ
    for i in range(10):
        w = normal_windows[(i + 20) % len(normal_windows)]
        state, telem = sim.update_tick(w, current_depth_m=0.10, is_surface_wrist=True)
        frames_data.append({
            "stage": "CỨU HỘ & HỒI PHỤC",
            "window": w,
            "telem": telem,
            "ground_truth": 0
        })

    print(f"[*] Đang render {len(frames_data)} frames với độ phân giải cao...")
    images = []

    # Lưu vết lịch sử MSE để vẽ biểu đồ đường
    mse_history = []
    time_history = []

    # Thống kê ma trận nhầm lẫn tích lũy
    tp, fp, tn, fn = 0, 0, 0, 0

    # Thiết lập giao diện màu tối phong cách Cyber/Sci-Fi chuyên nghiệp
    plt.style.use("dark_background")

    for f_idx, item in enumerate(frames_data):
        telem = item["telem"]
        w = item["window"]
        gt = item["ground_truth"]
        pred = 1 if telem["mse"] > tau else 0

        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 1 and pred == 0:
            fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 1.0

        mse_history.append(telem["mse"])
        time_history.append(f_idx * 1.28)

        # Khởi tạo Figure 1200x750 px (12 x 7.5 inch, 100 dpi)
        fig = plt.figure(figsize=(12, 7.5), dpi=100)
        fig.patch.set_facecolor("#0b0f19")

        # Tiêu đề Header
        title_color = "#38bdf8"
        state_str = telem["state"]
        
        if state_str == "NORMAL":
            badge_bg = "#065f46"
            badge_text = "NORMAL - BƠI AN TOÀN"
            status_color = "#34d399"
        elif state_str == "LEVEL_1_YELLOW":
            badge_bg = "#854d0e"
            badge_text = "CẤP 1 - VÀNG (RUNG CỔ TAY)"
            status_color = "#facc15"
        elif state_str == "LEVEL_2_ORANGE":
            badge_bg = "#9a3412"
            badge_text = "CẤP 2 - CAM (BẮN LORA TRẠM BỜ)"
            status_color = "#fb923c"
        else:
            badge_bg = "#991b1b"
            badge_text = "CẤP 3 - ĐỎ (HÚ CÒI + SÓNG ÂM + SMS)"
            status_color = "#f87171"

        fig.text(0.05, 0.94, "AIoT WEARABLE WATER ANOMALY DETECTION (TINYML)", fontsize=15, fontweight="bold", color="#f8fafc")
        fig.text(0.05, 0.91, f"Mô phỏng nRF52840 Cortex-M4F @ 64MHz | INT8 Quantized | T = {f_idx * 1.28:.1f}s", fontsize=10, color="#94a3b8")

        # Badge trạng thái ở góc trên bên phải
        fig.text(0.70, 0.925, f"[{badge_text}]", fontsize=11, fontweight="bold", color=status_color,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_bg, edgecolor=status_color, lw=1.5))

        # Grid chia 4 khu vực
        # 1. IMU Kinematic Signals (Top Left)
        ax_imu = fig.add_axes([0.06, 0.52, 0.48, 0.34])
        ax_imu.set_facecolor("#111827")
        t_samples = np.arange(128)
        ax_imu.plot(t_samples, w[:, 0], color="#38bdf8", lw=1.5, label="Ax")
        ax_imu.plot(t_samples, w[:, 1], color="#818cf8", lw=1.5, label="Ay")
        ax_imu.plot(t_samples, w[:, 2], color="#c084fc", lw=1.5, label="Az")
        ax_imu.set_title("Tín Hiệu IMU 6 Trục Sau Lọc DSP (Window = 128 mẫu @ 50Hz)", fontsize=10, color="#cbd5e1", pad=6)
        ax_imu.set_ylabel("Gia Tốc Chuẩn Hóa", fontsize=8, color="#94a3b8")
        ax_imu.grid(True, color="#1f2937", linestyle="--", alpha=0.7)
        ax_imu.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.3)
        ax_imu.set_ylim([-3.5, 3.5])

        # 2. Reconstruction Loss MSE vs Tau (Top Right)
        ax_loss = fig.add_axes([0.60, 0.52, 0.35, 0.34])
        ax_loss.set_facecolor("#111827")
        ax_loss.plot(time_history, mse_history, color="#f43f5e", lw=2, marker="o", markersize=4, label="MSE Loss")
        ax_loss.axhline(tau, color="#fbbf24", lw=1.8, linestyle="--", label=f"Tau = {tau:.3f}")
        ax_loss.set_title("Sai Số Tái Tạo MSE vs Ngưỡng Động tau", fontsize=10, color="#cbd5e1", pad=6)
        ax_loss.set_ylabel("MSE Loss", fontsize=8, color="#94a3b8")
        ax_loss.set_xlabel("Thời Gian (giây)", fontsize=8, color="#94a3b8")
        ax_loss.grid(True, color="#1f2937", linestyle="--", alpha=0.7)
        ax_loss.legend(loc="upper left", fontsize=8, framealpha=0.3)
        ax_loss.set_ylim([0, 7.0])

        # 3. Thông số phần cứng & Cảm biến áp suất (Bottom Left)
        ax_info = fig.add_axes([0.06, 0.10, 0.48, 0.34])
        ax_info.set_facecolor("#111827")
        ax_info.axis("off")
        
        info_lines = [
            ("Giai Đoạn Diễn Biến:", item["stage"], status_color),
            ("Độ Sâu Cảm Biến MS5837:", f"{telem['depth_m']} mét ({'NGUY HIỂM > 1.2m' if telem['depth_m'] > 1.2 else 'AN TOÀN'})", "#38bdf8"),
            ("Tiếp Điểm Nước (Water Pin):", "Tay Ngoài Nước (Phát LoRa OK)" if telem['is_surface_wrist'] else "Tay Chìm Trong Nước", "#e2e8f0"),
            ("Bộ Đếm Thời Gian FSM:", f"{telem['state_timer_sec']:.1f} giây / 15.0s", status_color),
            ("Biểu Quyết Đa Cửa Sổ (3/3):", "XÁC NHẬN IDR!" if telem['idr_detected'] else "Bình thường", status_color),
            ("Kích Thước Flash ROM:", "29.11 KB (Mục tiêu: < 40 KB)", "#34d399"),
            ("Độ Trễ Suy Luận ARM M4F:", f"{telem['latency_ms']:.2f} ms (Chuẩn: < 15 ms)", "#34d399")
        ]

        y_pos = 0.90
        ax_info.text(0.04, 0.98, "GIÁM SÁT VI ĐIỀU KHIỂN & CẢM BIẾN", fontsize=10, fontweight="bold", color="#38bdf8")
        for label, val, c in info_lines:
            ax_info.text(0.04, y_pos, label, fontsize=9, color="#94a3b8")
            ax_info.text(0.55, y_pos, val, fontsize=9, fontweight="bold", color=c)
            y_pos -= 0.13

        # Vẽ viền ngoài card
        rect_info = patches.Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                      fill=False, edgecolor="#334155", lw=1.2)
        ax_info.add_patch(rect_info)

        # 4. Bảng Chấm Điểm F1-Score & Confusion Matrix (Bottom Right)
        ax_metrics = fig.add_axes([0.60, 0.10, 0.35, 0.34])
        ax_metrics.set_facecolor("#111827")
        ax_metrics.axis("off")

        ax_metrics.text(0.05, 0.98, "CHẤM ĐIỂM F1-SCORE & CONFUSION MATRIX", fontsize=10, fontweight="bold", color="#34d399")

        # Confusion Matrix mini
        ax_metrics.text(0.05, 0.85, f"TP (True Pos):   {tp:2d}", fontsize=9, fontweight="bold", color="#34d399")
        ax_metrics.text(0.55, 0.85, f"FP (False Pos):  {fp:2d}", fontsize=9, fontweight="bold", color="#f87171")
        ax_metrics.text(0.05, 0.72, f"FN (False Neg):  {fn:2d}", fontsize=9, fontweight="bold", color="#f87171")
        ax_metrics.text(0.55, 0.72, f"TN (True Neg):   {tn:2d}", fontsize=9, fontweight="bold", color="#34d399")

        ax_metrics.axhline(0.62, color="#334155", lw=1)

        # F1 Scores
        f1_color = "#34d399" if f1 >= 0.90 else "#fbbf24"
        ax_metrics.text(0.05, 0.48, "PRECISION:", fontsize=9, color="#94a3b8")
        ax_metrics.text(0.60, 0.48, f"{prec*100:6.1f} %", fontsize=10, fontweight="bold", color="#38bdf8")

        ax_metrics.text(0.05, 0.34, "RECALL:", fontsize=9, color="#94a3b8")
        ax_metrics.text(0.60, 0.34, f"{rec*100:6.1f} %", fontsize=10, fontweight="bold", color="#38bdf8")

        ax_metrics.text(0.05, 0.18, "F1-SCORE:", fontsize=11, fontweight="bold", color="#f8fafc")
        ax_metrics.text(0.55, 0.16, f"{f1*100:6.2f} %", fontsize=14, fontweight="bold", color=f1_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#064e3b", edgecolor=f1_color, lw=1.2))

        rect_metrics = patches.Rectangle((0, 0), 1, 1, transform=ax_metrics.transAxes,
                                         fill=False, edgecolor="#334155", lw=1.2)
        ax_metrics.add_patch(rect_metrics)

        # Lưu frame ra buffer ảnh của PIL
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba)
        images.append(img.convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)

    # Lưu thành GIF hoạt họa
    os.makedirs(os.path.dirname(os.path.abspath(output_gif_path)), exist_ok=True)
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=320,  # 320ms mỗi frame = mượt mà và dễ theo dõi
        loop=0
    )
    
    gif_size_kb = os.path.getsize(output_gif_path) / 1024.0
    print(f"\n[+] ĐÃ TẠO VIDEO DEMO HOẠT HỌA THÀNH CÔNG!")
    print(f"    - Đường dẫn: {output_gif_path}")
    print(f"    - Kích thước: {gif_size_kb:.2f} KB ({len(images)} frames)")
    print(f"    - F1-Score cuối cùng: {f1*100:.2f} %")

if __name__ == "__main__":
    generate_demo_animation()

