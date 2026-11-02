"""
scripts/export_model_header.py
Tự động xuất tệp nhị phân TFLite INT8 sang định dạng C/C++ Header tương thích ARM CMSIS-NN.
Đảm bảo thuộc tính alignas(16) và cơ chế Zero-dynamic-memory cho Nordic nRF52840.
"""
import os
import sys
from pathlib import Path

def convert_tflite_to_header(tflite_path: str, output_header_path: str, array_name: str = "g_swimming_model_data"):
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Không tìm thấy file TFLite: {tflite_path}")

    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()

    file_size = len(tflite_bytes)
    print(f"[*] Đang xử lý chuyển đổi: {tflite_path} ({file_size} bytes = {file_size / 1024.0:.2f} KB)")

    lines = []
    lines.append("// ===========================================================================")
    lines.append("// AUTO-GENERATED TINYML MODEL FOR NORDIC nRF52840 (ARM CORTEX-M4F)")
    lines.append(f"// Dung lượng mô hình: {file_size} bytes ({file_size / 1024.0:.2f} KB)")
    lines.append("// ===========================================================================\n")
    lines.append("#ifndef MODEL_DATA_H_")
    lines.append("#define MODEL_DATA_H_\n")
    lines.append("#include <stdint.h>\n")
    lines.append("// Đảm bảo căn chỉnh 16-byte cho tập lệnh CMSIS-NN / FPU của Cortex-M4F")
    lines.append('#ifdef __cplusplus')
    lines.append('extern "C" {')
    lines.append('#endif\n')
    lines.append(f"alignas(16) const unsigned char {array_name}[] = {{")

    # Định dạng mảng byte C: 12 byte mỗi dòng (chuẩn hex)
    hex_values = [f"0x{b:02x}" for b in tflite_bytes]
    for i in range(0, len(hex_values), 12):
        chunk = hex_values[i:i + 12]
        lines.append("    " + ", ".join(chunk) + ("," if i + 12 < len(hex_values) else ""))

    lines.append("};\n")
    lines.append(f"const unsigned int {array_name}_len = {file_size};\n")
    lines.append('#ifdef __cplusplus')
    lines.append('}')
    lines.append('#endif\n')
    lines.append("#endif // MODEL_DATA_H_\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[+] Đã sinh tệp header thành công tại: {output_header_path}")

if __name__ == "__main__":
    tflite_file = "models/autoencoder_int8.tflite"
    header_file = "firmware/include/model_data.h"
    convert_tflite_to_header(tflite_file, header_file)

