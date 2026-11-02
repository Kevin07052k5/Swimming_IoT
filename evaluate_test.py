import numpy as np
import tensorflow as tf
from build_pipeline import extract_windows

# Tải tệp TFLite INT8
interpreter = tf.lite.Interpreter(model_path="build/model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_scale, in_zero = input_details["quantization"]
out_scale, out_zero = output_details["quantization"]

def run_inference(sample_window):
    # Lượng tử hóa đầu vào từ float32 -> int8
    quantized_in = np.round(sample_window / in_scale + in_zero).astype(np.int8)
    quantized_in = np.expand_dims(quantized_in, axis=0)

    interpreter.set_tensor(input_details["index"], quantized_in)
    interpreter.invoke()

    # Giải lượng tử hóa đầu ra int8 -> float32
    quantized_out = interpreter.get_tensor(output_details["index"])[0]
    dequantized_out = (quantized_out.astype(np.float32) - out_zero) * out_scale
    
    # Tính sai số toàn phương trung bình (MSE)
    mse = np.mean(np.square(sample_window - dequantized_out))
    return mse

# Đánh giá trên dữ liệu bơi thường và dữ liệu dị thường
normal_windows = extract_windows("datasets/test_normal.csv")
anomaly_windows = extract_windows("datasets/test_anomaly.csv")

print("--- KẾT QUẢ THỬ NGHIỆM MÔ HÌNH INT8 ---")
mse_normal_list = [run_inference(w) for w in normal_windows[:10]]
mse_anomaly_list = [run_inference(w) for w in anomaly_windows[:10]]

print(f"MSE trung bình khi BƠI BÌNH THƯỜNG : {np.mean(mse_normal_list):.5f}")
print(f"MSE trung bình khi ĐUỐI NƯỚC (IDR) : {np.mean(mse_anomaly_list):.5f}")

if np.mean(mse_anomaly_list) > np.mean(mse_normal_list) * 2.0:
    print("-> ĐÁNH GIÁ: Mô hình phân tách bất thường cực kỳ rõ nét. Đạt tiêu chuẩn triển khai nhúng.")