import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. BỘ LỌC SỐ DSP & CỬA SỔ TRƯỢT
def butter_lowpass_filter(data, cutoff=10.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def extract_windows(csv_path, window_size=128, step=64):
    df = pd.read_csv(csv_path)
    raw_values = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values
    filtered = butter_lowpass_filter(raw_values)
    mean = np.mean(filtered, axis=0)
    std = np.std(filtered, axis=0) + 1e-8
    normalized = (filtered - mean) / std

    windows = []
    for start in range(0, len(normalized) - window_size + 1, step):
        windows.append(normalized[start : start + window_size, :])
    return np.array(windows, dtype=np.float32)

# 2. XÂY DỰNG MÔ HÌNH 1D-CNN AUTOENCODER (ĐÚNG SPEC ĐỀ CƯƠNG)
def build_autoencoder():
    inputs = layers.Input(shape=(128, 6), name="input_imu")
    
    # Encoder
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (64, 32)
    x = layers.Conv1D(16, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (32, 16)
    x = layers.Conv1D(8, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (16, 8)
    
    # Bottleneck
    latent = layers.Conv1D(1, kernel_size=3, padding="same", activation="relu", name="latent_space")(x) # (16, 1)

    # Decoder
    x = layers.Conv1DTranspose(8, kernel_size=3, strides=2, padding="same", activation="relu")(latent)   # (32, 8)
    x = layers.Conv1DTranspose(16, kernel_size=5, strides=2, padding="same", activation="relu")(x)        # (64, 16)
    x = layers.Conv1DTranspose(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)        # (128, 32)
    outputs = layers.Conv1D(6, kernel_size=3, padding="same", name="output_reconstruct")(x)              # (128, 6)

    model = models.Model(inputs=inputs, outputs=outputs, name="1D_CNN_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model

print("[1/4] Đang nạp và lọc dữ liệu DSP...")
x_train = extract_windows("datasets/train_normal.csv")
x_test_normal = extract_windows("datasets/test_normal.csv")
x_test_anomaly = extract_windows("datasets/test_anomaly.csv")

print("[2/4] Bắt đầu huấn luyện mô hình Autoencoder...")
model = build_autoencoder()
model.fit(x_train, x_train, epochs=25, batch_size=32, validation_split=0.15, verbose=1)

# Tính ngưỡng sai số động tau = mu + 3*sigma
train_reconstructions = model.predict(x_train)
train_mse = np.mean(np.square(x_train - train_reconstructions), axis=(1, 2))
threshold = np.mean(train_mse) + 3.0 * np.std(train_mse)
print(f"-> Ngưỡng phát hiện bất thường động (tau): {threshold:.5f}")

# 3. LƯỢNG TỬ HÓA INT8 BẢO ĐẢM KÍCH THƯỚC < 40 KB
print("[3/4] Lượng tử hóa mô hình sang INT8 cho ARM Cortex-M4F...")
def representative_data_gen():
    for i in range(min(100, len(x_train))):
        yield [x_train[i : i + 1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
os.makedirs("build", exist_ok=True)
with open("build/model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

size_kb = len(tflite_quant_model) / 1024.0
print(f"-> Kích thước mô hình TFLite INT8: {size_kb:.2f} KB (Mục tiêu: <= 32.4 KB)")

# 4. XUẤT RA FILE C++ HEADER MODEL_DATA.H
print("[4/4] Đang chuyển đổi sang file model_data.h...")
os.makedirs("firmware/include", exist_ok=True)
with open("firmware/include/model_data.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H_\n#define MODEL_DATA_H_\n\n")
    f.write("#include <cstdint>\n\n")
    f.write(f"// Ngưỡng tính toán từ dữ liệu huấn luyện: {threshold:.5f}\n")
    f.write(f"const float DYNAMIC_ANOMALY_THRESHOLD = {threshold:.5f}f;\n\n")
    f.write("// Căn chỉnh 16-byte cho tập lệnh ARM SIMD\n")
    f.write("alignas(16) const unsigned char g_model[] = {\n    ")
    
    for i, byte in enumerate(tflite_quant_model):
        f.write(f"0x{byte:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n    ")
            
    f.write("\n};\n\n")
    f.write(f"const int g_model_len = {len(tflite_quant_model)};\n\n")
    f.write("#endif // MODEL_DATA_H_\n")

print("-> [HOÀN TẤT] File firmware/include/model_data.h đã được tạo thành công!")