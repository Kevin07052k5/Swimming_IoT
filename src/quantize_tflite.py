"""
src/quantize_tflite.py
Thực hiện Full Integer INT8 Quantization (TFLite Post-Training Quantization)
và tự động xuất mảng C byte array (model_data.h) căn chỉnh alignas(16)
phục vụ nhúng trực tiếp vào firmware C++ trên ARM Cortex-M4F (Nordic nRF52840).
"""
import os
import sys
import json
from pathlib import Path
from typing import Generator, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import tensorflow as tf
from src.config import CONFIG, SystemConfig

def representative_data_gen(sample_data: np.ndarray, max_samples: int = 300) -> Generator[List[np.ndarray], None, None]:
    """
    Tạo dữ liệu đại diện phục vụ cân chỉnh dải giá trị động (Dynamic Range Calibration) cho INT8.
    """
    n = min(max_samples, len(sample_data))
    for i in range(n):
        # Shape đầu vào: (1, 128, 6) kiểu float32
        yield [sample_data[i:i + 1].astype(np.float32)]

def export_c_header(
    tflite_bytes: bytes, 
    output_header_path: str, 
    threshold_tau: Optional[float] = None,
    array_name: str = "g_swimming_model_data"
) -> None:
    """
    Xuất mảng byte C++ chuẩn công nghiệp với căn chỉnh alignas(16) cho ARM CMSIS-NN SIMD.
    """
    file_size = len(tflite_bytes)
    lines = []
    lines.append("// ===========================================================================")
    lines.append("// AUTO-GENERATED TINYML MODEL HEADER FOR ARM CORTEX-M4F (NORDIC nRF52840)")
    lines.append(f"// Dung lượng mô hình TFLite INT8: {file_size} bytes ({file_size / 1024.0:.2f} KB)")
    lines.append("// Thuộc tính: alignas(16), Zero-Heap Static ROM Allocation")
    lines.append("// ===========================================================================\n")
    lines.append("#ifndef SWIMMING_MODEL_DATA_H_")
    lines.append("#define SWIMMING_MODEL_DATA_H_\n")
    lines.append("#include <stdint.h>\n")

    if threshold_tau is not None:
        lines.append(f"// Ngưỡng bất thường thực nghiệm tau = mean + 3.5 * std")
        lines.append(f"#define DYNAMIC_ANOMALY_THRESHOLD ({threshold_tau:.6f}f)\n")

    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif\n")
    lines.append(f"alignas(16) const unsigned char {array_name}[] = {{")

    # Định dạng mảng byte C: 12 byte mỗi dòng (chuẩn hex)
    hex_values = [f"0x{b:02x}" for b in tflite_bytes]
    for i in range(0, len(hex_values), 12):
        chunk = hex_values[i:i + 12]
        lines.append("    " + ", ".join(chunk) + ("," if i + 12 < len(hex_values) else ""))

    lines.append("};\n")
    lines.append(f"const unsigned int {array_name}_len = {file_size};\n")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif\n")
    lines.append("#endif // SWIMMING_MODEL_DATA_H_\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] Đã tạo file header C nhúng tại: {output_header_path}")

def quantize_to_tflite_int8(
    saved_model_path: str,
    sample_data_path: str,
    output_tflite_path: str,
    output_header_path: str,
    stats_json_path: Optional[str] = None,
    config: SystemConfig = CONFIG
) -> bytes:
    """
    Thực hiện chuyển đổi mô hình Keras sang TFLite INT8 toàn phần và xuất header C.
    """
    print("[*] Đang khởi tạo bộ chuyển đổi TFLite Converter...")
    model = tf.keras.models.load_model(saved_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 1. Kích hoạt tối ưu hóa lượng tử hóa
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. Đọc dữ liệu đại diện để hiệu chỉnh dải giá trị
    calibration_data = np.load(sample_data_path).astype(np.float32)
    converter.representative_dataset = lambda: representative_data_gen(calibration_data, max_samples=300)
    
    # 3. Ép kiểu toàn bộ tensor đầu vào và đầu ra về INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("[*] Đang thực thi chuyển đổi INT8 Post-Training Quantization...")
    tflite_quant_model = converter.convert()
    
    # Lưu file .tflite
    os.makedirs(os.path.dirname(os.path.abspath(output_tflite_path)), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    
    file_size_kb = len(tflite_quant_model) / 1024.0
    print(f"[+] Lượng tử hóa INT8 thành công!")
    print(f"    - Kích thước file TFLite: {file_size_kb:.2f} KB (Mục tiêu: < {config.MAX_FLASH_SIZE_KB} KB)")
    
    if file_size_kb > config.MAX_FLASH_SIZE_KB:
        print(f"[!] CẢNH BÁO: Kích thước ({file_size_kb:.2f} KB) vượt quá ngưỡng {config.MAX_FLASH_SIZE_KB} KB!")
    else:
        print(f"[+] Kích thước hoàn toàn thỏa mãn ngân sách Flash ROM nRF52840!")

    # Đọc threshold_tau từ stats_json_path nếu có
    threshold_tau = None
    if stats_json_path and os.path.exists(stats_json_path):
        with open(stats_json_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            threshold_tau = stats.get("threshold_tau")

    # Xuất header file C
    export_c_header(tflite_quant_model, output_header_path, threshold_tau=threshold_tau)

    # Đồng bộ lưu một bản vào firmware/include/model_data.h nếu đường dẫn khác nhau
    firmware_header = config.FIRMWARE_INC_DIR / "model_data.h"
    if str(firmware_header.resolve()) != str(Path(output_header_path).resolve()):
        export_c_header(tflite_quant_model, str(firmware_header), threshold_tau=threshold_tau)

    return tflite_quant_model

if __name__ == "__main__":
    saved_model = str(CONFIG.MODELS_DIR / "autoencoder_fp32.keras")
    sample_data = str(CONFIG.DATA_PROCESSED_DIR / "normal_swimming.npy")
    out_tflite = str(CONFIG.MODELS_DIR / "autoencoder_int8.tflite")
    out_header = str(CONFIG.MODELS_DIR / "model_data.h")
    stats_file = str(CONFIG.MODELS_DIR / "stats.json")
    quantize_to_tflite_int8(saved_model, sample_data, out_tflite, out_header, stats_json_path=stats_file)
