"""
run_full_pipeline.py
Kịch bản thực thi toàn trình (End-to-End Execution Pipeline):
1. Chuẩn bị và tiền xử lý dữ liệu IMU 6 trục (DSP Butterworth + Z-score + Cửa sổ trượt)
2. Huấn luyện mạng 1D-CNN Autoencoder bán giám sát (Semi-supervised Learning)
3. Tính toán ngưỡng dị thường động tau = mean + 3.5 * std
4. Lượng tử hóa toàn phần INT8 (TFLite Micro) & kiểm tra kích thước (< 40 KB)
5. Xuất mảng byte C++ alignas(16) vào model_data.h
6. Chạy mô phỏng suy luận biên và máy trạng thái FSM 3 cấp
7. Kiểm thử điều phối cảnh báo Gateway trạm bờ (RPi CM4)
"""
import os
import sys
import time
from pathlib import Path

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo KERAS_HOME hợp lệ
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))

from src.config import CONFIG
from src.data_generator import prepare_and_process_datasets
from src.train import train_autoencoder
from src.quantize_tflite import quantize_to_tflite_int8
from src.edge_simulator import run_benchmark_simulation
from src.gateway_service import PoolsideGatewayService

def main():
    print("\n" + "#"*80)
    print(" KHỞI CHẠY HỆ THỐNG AIoT WEARABLE WATER ANOMALY DETECTION (TINYML)")
    print(" ĐỀ TÀI: XÂY DỰNG THIẾT KẾ MÔ HÌNH AIoT PHÁT HIỆN BẤT THƯỜNG KHI BƠI (TINYML)")
    print("#"*80 + "\n")

    start_total_time = time.time()

    # BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU DSP
    print("="*80)
    print(" [BƯỚC 1/5] CHUỖI TIỀN XỬ LÝ TÍN HIỆU ĐỘNG HỌC (DSP PIPELINE)")
    print("="*80)
    data_info = prepare_and_process_datasets(CONFIG)

    # BƯỚC 2: HUẤN LUYỆN BÁN GIÁM SÁT 1D-CNN AUTOENCODER
    print("\n" + "="*80)
    print(" [BƯỚC 2/5] HUẤN LUYỆN AUTOENCODER BÁN GIÁM SÁT (FP32)")
    print("="*80)
    model_save_path = str(CONFIG.MODELS_DIR / "autoencoder_fp32.keras")
    stats_save_path = str(CONFIG.MODELS_DIR / "stats.json")
    train_stats = train_autoencoder(
        normal_data_path=data_info["normal_file"],
        model_save_path=model_save_path,
        stats_save_path=stats_save_path,
        config=CONFIG,
        epochs=30,  # Huấn luyện hội tụ tối ưu
        batch_size=CONFIG.BATCH_SIZE
    )

    # BƯỚC 3: LƯỢNG TỬ HÓA INT8 & XUẤT C++ HEADER MODEL_DATA.H
    print("\n" + "="*80)
    print(" [BƯỚC 3/5] LƯỢNG TỬ HÓA TOÀN PHẦN INT8 & XUẤT C-ARRAY ARM CORTEX-M4F")
    print("="*80)
    output_tflite = str(CONFIG.MODELS_DIR / "autoencoder_int8.tflite")
    output_header = str(CONFIG.MODELS_DIR / "model_data.h")
    tflite_bytes = quantize_to_tflite_int8(
        saved_model_path=model_save_path,
        sample_data_path=data_info["normal_file"],
        output_tflite_path=output_tflite,
        output_header_path=output_header,
        stats_json_path=stats_save_path,
        config=CONFIG
    )

    # BƯỚC 4: GIẢ LẬP SUY LUẬN BIÊN & FSM 3 CẤP TẠI NRF52840
    print("\n" + "="*80)
    print(" [BƯỚC 4/5] GIẢ LẬP SUY LUẬN BIÊN & MÁY TRẠNG THÁI AN TOÀN 3 CẤP (FSM)")
    print("="*80)
    sim_results = run_benchmark_simulation(
        tflite_path=output_tflite,
        stats_path=stats_save_path,
        normal_data_path=data_info["normal_file"],
        anomaly_data_path=data_info["anomaly_file"]
    )

    # BƯỚC 5: ĐIỀU PHỐI CẢNH BÁO TRẠM BỜ GATEWAY RPI CM4
    print("\n" + "="*80)
    print(" [BƯỚC 5/5] TRẠM BỜ GATEWAY: ĐIỀU PHỐI CẢNH BÁO ĐA KÊNH & KHẨN CẤP")
    print("="*80)
    gateway = PoolsideGatewayService()
    # Lấy các mốc sự kiện quan trọng từ mô phỏng để bắn qua Gateway
    timeline = sim_results["timeline"]
    events_to_test = [
        ("SỰ KIỆN 1: BƠI BÌNH THƯỜNG", timeline[2]),
        ("SỰ KIỆN 2: CẢNH BÁO CẤP 1 (VÀNG)", next(e for e in timeline if e["state"] == "LEVEL_1_YELLOW")),
        ("SỰ KIỆN 3: CẢNH BÁO CẤP 2 (CAM)", next(e for e in timeline if e["state"] == "LEVEL_2_ORANGE")),
        ("SỰ KIỆN 4: BÁO ĐỘNG ĐỎ CẤP 3 (ĐỎ)", next(e for e in timeline if e["state"] == "LEVEL_3_RED"))
    ]

    for label, event in events_to_test:
        print(f"\n>>> {label} <<<")
        gateway.process_telemetry_event(event)

    total_time = time.time() - start_total_time
    print("\n" + "#"*80)
    print(f" [HOÀN TẤT TOÀN DIỆN] TOÀN BỘ QUY TRÌNH THỰC THI TRONG {total_time:.2f} GIÂY!")
    print(f" - Mô hình FP32:       {model_save_path}")
    print(f" - Mô hình INT8:       {output_tflite} ({len(tflite_bytes)/1024.0:.2f} KB)")
    print(f" - C Header:           {output_header} & firmware/include/model_data.h")
    print(f" - Ngưỡng dị thường tau: {train_stats['threshold_tau']:.6f}")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
