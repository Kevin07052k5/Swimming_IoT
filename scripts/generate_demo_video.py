"""
scripts/generate_demo_video.py
Xuất video mô phỏng độ phân giải cao dưới định dạng chuẩn MP4 (H.264/AAC) và MKV (Matroska).
Mô phỏng toàn bộ quy trình:
1. IMU 6 trục thu nhận và tính sai số tái tạo MSE qua TFLite INT8
2. So sánh ngưỡng động tau và biểu quyết FSM 3 cửa sổ liên tiếp
3. Phân cấp cảnh báo 3 cấp: NORMAL (Xanh) -> LEVEL 1 (Vàng) -> LEVEL 2 (Cam) -> LEVEL 3 (Đỏ)
4. Chấm điểm ma trận nhầm lẫn (Confusion Matrix) và F1-Score thời gian thực
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
import imageio

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

def generate_video_formats(
    output_mp4_path: str = str(BASE_DIR / "demo_simulation.mp4"),
    output_mkv_path: str = str(BASE_DIR / "demo_simulation.mkv"),
    tflite_path: str = str(CONFIG.MODELS_DIR / "autoencoder_int8.tflite"),
    stats_path: str = str(CONFIG.MODELS_DIR / "stats.json"),
    normal_data_path: str = str(CONFIG.DATA_PROCESSED_DIR / "normal_swimming.npy"),
    anomaly_data_path: str = str(CONFIG.DATA_PROCESSED_DIR / "anomaly_swimming.npy"),
    fps: int = 4
):
    print("\n" + "="*80)
    print(" KHỞI TẠO TIẾN TRÌNH RENDER VIDEO DEMO MP4 / MKV (H.264)")
    print("="*80)

    sim = EdgeWearableNodeSimulator(tflite_path, stats_path)
    normal_windows = np.load(normal_data_path).astype(np.float32)
    anomaly_windows = np.load(anomaly_data_path).astype(np.float32)
    tau = sim.threshold_tau

    # Xây dựng kịch bản 3 giai đoạn:
    # 1. Bơi bình thường: 11 cửa sổ
    # 2. Phản xạ đuối nước IDR: 15 cửa sổ (chìm dần)
    # 3. Cứu hộ tiếp cận: 10 cửa sổ (hồi phục)
    frames_data = []

    for i in range(11):
        w = normal_windows[i % len(normal_windows)]
        state, telem = sim.update_tick(w, current_depth_m=0.25, is_surface_wrist=True)
        frames_data.append({
            "stage": "BƠI BÌNH THƯỜNG (FREESTYLE / BREASTSTROKE)",
            "window": w,
            "telem": telem,
            "ground_truth": 0
        })

    for i in range(15):
        w = anomaly_windows[i % len(anomaly_windows)]
        depth = min(1.55, 0.40 + i * 0.08)
        state, telem = sim.update_tick(w, current_depth_m=depth, is_surface_wrist=(depth < 0.6))
        frames_data.append({
            "stage": "PHẢN XẠ ĐUỐI NƯỚC TỰ NHIÊN (IDR HOẢNG LOẠN)",
            "window": w,
            "telem": telem,
            "ground_truth": 1
        })

    for i in range(10):
        w = normal_windows[(i + 20) % len(normal_windows)]
        state, telem = sim.update_tick(w, current_depth_m=0.08, is_surface_wrist=True)
        frames_data.append({
            "stage": "CỨU HỘ TIẾP CẬN VÀ ĐƯA LÊN BỜ",
            "window": w,
            "telem": telem,
            "ground_truth": 0
        })

    print(f"[*] Tổng số khung cảnh mô phỏng: {len(frames_data)} cửa sổ")
    print(f"[*] Tốc độ khung hình (FPS): {fps} fps (~ {(len(frames_data) / fps):.1f} giây thời lượng video)")

    rendered_rgb_frames = []
    mse_history = []
    time_history = []
    tp, fp, tn, fn = 0, 0, 0, 0

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

        # Figure 1280x720 px chuẩn HD 720p (12.8 x 7.2 inch @ 100 dpi)
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        fig.patch.set_facecolor("#0b0f19")

        # Xác định trạng thái FSM và màu cảnh báo
        state_str = telem["state"]
        if state_str == "NORMAL":
            badge_bg = "#065f46"
            badge_text = "NORMAL - BƠI AN TOÀN"
            status_color = "#34d399"
        elif state_str == "LEVEL_1_YELLOW":
            badge_bg = "#854d0e"
            badge_text = "CẤP 1 - VÀNG (RUNG HAPTIC CỔ TAY)"
            status_color = "#facc15"
        elif state_str == "LEVEL_2_ORANGE":
            badge_bg = "#9a3412"
            badge_text = "CẤP 2 - CAM (BẮN LORA 433MHz TRẠM BỜ)"
            status_color = "#fb923c"
        else:
            badge_bg = "#991b1b"
            badge_text = "CẤP 3 - ĐỎ (HÚ CÒI + SÓNG ÂM 3-5kHz + SMS)"
            status_color = "#f87171"

        # Tiêu đề Header
        fig.text(0.04, 0.94, "AIoT WEARABLE WATER ANOMALY DETECTION (TINYML)", fontsize=15, fontweight="bold", color="#f8fafc")
        fig.text(0.04, 0.91, f"Mô phỏng ARM Cortex-M4F @ 64MHz | Model INT8 (29.11 KB) | T = {f_idx * 1.28:.1f}s / 46.1s", fontsize=10, color="#94a3b8")

        # Badge trạng thái ở góc trên bên phải
        fig.text(0.68, 0.925, f"[{badge_text}]", fontsize=11, fontweight="bold", color=status_color,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_bg, edgecolor=status_color, lw=1.5))

        # 1. Đồ thị sóng IMU 6 trục (Top Left)
        ax_imu = fig.add_axes([0.05, 0.52, 0.48, 0.35])
        ax_imu.set_facecolor("#111827")
        t_samples = np.arange(128)
        ax_imu.plot(t_samples, w[:, 0], color="#38bdf8", lw=1.6, label="Ax (Gia tốc X)")
        ax_imu.plot(t_samples, w[:, 1], color="#818cf8", lw=1.6, label="Ay (Gia tốc Y)")
        ax_imu.plot(t_samples, w[:, 2], color="#c084fc", lw=1.6, label="Az (Gia tốc Z)")
        ax_imu.set_title("Tín Hiệu IMU 6 Trục Sau Lọc DSP (Window = 128 mẫu @ 50Hz = 2.56s)", fontsize=10, color="#cbd5e1", pad=6)
        ax_imu.set_ylabel("Gia Tốc Chuẩn Hóa", fontsize=8, color="#94a3b8")
        ax_imu.grid(True, color="#1f2937", linestyle="--", alpha=0.7)
        ax_imu.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.3)
        ax_imu.set_ylim([-3.5, 3.5])

        # 2. Sai số tái tạo MSE vs Ngưỡng tau (Top Right)
        ax_loss = fig.add_axes([0.59, 0.52, 0.37, 0.35])
        ax_loss.set_facecolor("#111827")
        ax_loss.plot(time_history, mse_history, color="#f43f5e", lw=2.2, marker="o", markersize=4, label="MSE Loss Thực Tế")
        ax_loss.axhline(tau, color="#fbbf24", lw=2.0, linestyle="--", label=f"Ngưỡng tau = {tau:.3f}")
        ax_loss.set_title(f"Sai Số Tái Tạo Autoencoder MSE (Hiện tại: {telem['mse']:.3f})", fontsize=10, color="#cbd5e1", pad=6)
        ax_loss.set_ylabel("MSE Loss", fontsize=8, color="#94a3b8")
        ax_loss.set_xlabel("Thời Gian Mô Phỏng (giây)", fontsize=8, color="#94a3b8")
        ax_loss.grid(True, color="#1f2937", linestyle="--", alpha=0.7)
        ax_loss.legend(loc="upper left", fontsize=8, framealpha=0.3)
        ax_loss.set_ylim([0, 7.0])

        # 3. Thông số phần cứng & Cảm biến (Bottom Left)
        ax_info = fig.add_axes([0.05, 0.08, 0.48, 0.36])
        ax_info.set_facecolor("#111827")
        ax_info.axis("off")

        info_lines = [
            ("Giai Đoạn Mô Phỏng:", item["stage"][:38], status_color),
            ("Độ Sâu Cảm Biến MS5837:", f"{telem['depth_m']:.2f} m ({'NGUY HIỂM > 1.2m' if telem['depth_m'] > 1.2 else 'An Toàn'})", "#38bdf8"),
            ("Tiếp Điểm Nước (Water Pin):", "Tay Ngoài Nước (Phát LoRa OK)" if telem['is_surface_wrist'] else "Tay Chìm Trong Nước", "#e2e8f0"),
            ("Bộ Đếm Thời Gian FSM:", f"{telem['state_timer_sec']:.1f} s / 15.0 s", status_color),
            ("Biểu Quyết Đa Cửa Sổ (3/3):", "XÁC NHẬN IDR!" if telem['idr_detected'] else "Bình thường", status_color),
            ("Dung Lượng Mô Hình INT8:", "29.11 KB (Flash limit: 40 KB)", "#34d399"),
            ("Độ Trễ Suy Luận ARM M4F:", f"{telem['latency_ms']:.2f} ms (Yêu cầu: < 15 ms)", "#34d399")
        ]

        ax_info.text(0.04, 0.96, "THÔNG SỐ GIÁM SÁT VI ĐIỀU KHIỂN & CẢM BIẾN", fontsize=10, fontweight="bold", color="#38bdf8")
        y_pos = 0.86
        for label, val, c in info_lines:
            ax_info.text(0.04, y_pos, label, fontsize=8.5, color="#94a3b8")
            ax_info.text(0.52, y_pos, val, fontsize=8.5, fontweight="bold", color=c)
            y_pos -= 0.125

        rect_info = patches.Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                      fill=False, edgecolor="#334155", lw=1.2)
        ax_info.add_patch(rect_info)

        # 4. Bảng Chấm Điểm F1-Score & Confusion Matrix (Bottom Right)
        ax_metrics = fig.add_axes([0.59, 0.08, 0.37, 0.36])
        ax_metrics.set_facecolor("#111827")
        ax_metrics.axis("off")

        ax_metrics.text(0.05, 0.96, "CHẤM ĐIỂM F1-SCORE & MA TRẬN NHẦM LẪN", fontsize=10, fontweight="bold", color="#34d399")

        # Confusion Matrix
        ax_metrics.text(0.05, 0.83, f"TP (True Pos):   {tp:2d}", fontsize=9, fontweight="bold", color="#34d399")
        ax_metrics.text(0.55, 0.83, f"FP (False Pos):  {fp:2d}", fontsize=9, fontweight="bold", color="#f87171")
        ax_metrics.text(0.05, 0.70, f"FN (False Neg):  {fn:2d}", fontsize=9, fontweight="bold", color="#f87171")
        ax_metrics.text(0.55, 0.70, f"TN (True Neg):   {tn:2d}", fontsize=9, fontweight="bold", color="#34d399")

        ax_metrics.axhline(0.60, color="#334155", lw=1)

        f1_color = "#34d399" if f1 >= 0.90 else "#fbbf24"
        ax_metrics.text(0.05, 0.46, "ĐỘ CHUẨN XÁC (PRECISION):", fontsize=8.5, color="#94a3b8")
        ax_metrics.text(0.65, 0.46, f"{prec*100:6.1f} %", fontsize=9.5, fontweight="bold", color="#38bdf8")

        ax_metrics.text(0.05, 0.32, "ĐỘ NHẠY (RECALL):", fontsize=8.5, color="#94a3b8")
        ax_metrics.text(0.65, 0.32, f"{rec*100:6.1f} %", fontsize=9.5, fontweight="bold", color="#38bdf8")

        ax_metrics.text(0.05, 0.16, "F1-SCORE HIỆN TẠI:", fontsize=10.5, fontweight="bold", color="#f8fafc")
        ax_metrics.text(0.58, 0.13, f"{f1*100:6.2f} %", fontsize=14, fontweight="bold", color=f1_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#064e3b", edgecolor=f1_color, lw=1.2))

        rect_metrics = patches.Rectangle((0, 0), 1, 1, transform=ax_metrics.transAxes,
                                         fill=False, edgecolor="#334155", lw=1.2)
        ax_metrics.add_patch(rect_metrics)

        # Chuyển đổi thành mảng RGB
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]  # Bỏ kênh alpha để tương thích chuẩn RGB MP4
        rendered_rgb_frames.append(rgb)
        plt.close(fig)

    # 1. XUẤT ĐỊNH DẠNG MP4 (H.264, yuv420p cho mọi trình phát)
    print(f"\n[*] Đang mã hóa và xuất tệp MP4: {output_mp4_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
    with imageio.get_writer(
        output_mp4_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "slow"]
    ) as writer:
        for frame in rendered_rgb_frames:
            writer.append_data(frame)

    mp4_size_mb = os.path.getsize(output_mp4_path) / (1024.0 * 1024.0)
    print(f"[+] ĐÃ TẠO VIDEO MP4 THÀNH CÔNG: {output_mp4_path} ({mp4_size_mb:.2f} MB)")

    # 2. XUẤT ĐỊNH DẠNG MKV (Matroska Container)
    print(f"\n[*] Đang đóng gói tệp MKV: {output_mkv_path}...")
    with imageio.get_writer(
        output_mkv_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18"]
    ) as writer:
        for frame in rendered_rgb_frames:
            writer.append_data(frame)

    mkv_size_mb = os.path.getsize(output_mkv_path) / (1024.0 * 1024.0)
    print(f"[+] ĐÃ TẠO VIDEO MKV THÀNH CÔNG: {output_mkv_path} ({mkv_size_mb:.2f} MB)")

    return {
        "mp4_path": output_mp4_path,
        "mkv_path": output_mkv_path,
        "mp4_size_mb": mp4_size_mb,
        "mkv_size_mb": mkv_size_mb,
        "total_frames": len(rendered_rgb_frames),
        "f1_score": f1
    }

if __name__ == "__main__":
    generate_video_formats()

