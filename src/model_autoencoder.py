"""
src/model_autoencoder.py
Định nghĩa kiến trúc mạng 1D-CNN Autoencoder bán giám sát tối ưu cho vi điều khiển ARM Cortex-M4F.
Mô hình nén dữ liệu qua nút cổ chai (16, 1) và tái tạo lại chính xác ma trận động học (128, 6).
"""
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import CONFIG, SystemConfig

def build_1d_cnn_autoencoder(config: SystemConfig = CONFIG) -> Model:
    """
    Xây dựng mạng 1D-CNN Autoencoder đối xứng:
    Input: (128, 6)
    Encoder -> MaxPool giảm chiều -> Bottleneck (16, 1) -> Decoder UpSampling -> Output (128, 6)
    """
    input_shape = (config.WINDOW_SIZE, config.NUM_CHANNELS)
    inputs = layers.Input(shape=input_shape, name="imu_input")

    # ==================== ENCODER ====================
    # Tầng 1: 32 bộ lọc kernel=5 trích xuất đặc trưng tần số thấp
    x = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="enc_conv1")(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="enc_pool1")(x)  # -> (64, 32)

    # Tầng 2: 16 bộ lọc kernel=5
    x = layers.Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="enc_pool2")(x)  # -> (32, 16)

    # Tầng 3: 8 bộ lọc kernel=3
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same", activation="relu", name="enc_conv3")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="enc_pool3")(x)   # -> (16, 8)

    # ==================== BOTTLENECK ====================
    # Nén về không gian vector ẩn cực đại (16, 1)
    bottleneck = layers.Conv1D(filters=1, kernel_size=3, padding="same", activation="relu", name="bottleneck")(x) # -> (16, 1)

    # ==================== DECODER ====================
    # Giải nén tầng 1: khôi phục kích thước (32, 8)
    x = layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding="same", activation="relu", name="dec_deconv1")(bottleneck)

    # Giải nén tầng 2: khôi phục kích thước (64, 16)
    x = layers.Conv1DTranspose(filters=16, kernel_size=5, strides=2, padding="same", activation="relu", name="dec_deconv2")(x)

    # Giải nén tầng 3: khôi phục kích thước (128, 32)
    x = layers.Conv1DTranspose(filters=32, kernel_size=5, strides=2, padding="same", activation="relu", name="dec_deconv3")(x)

    # Tầng tái tạo đầu ra: Khôi phục lại chính xác 6 kênh
    outputs = layers.Conv1D(filters=config.NUM_CHANNELS, kernel_size=5, padding="same", activation="linear", name="reconstruction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TinyML_Swimming_1D_CNN_AE")
    return model

if __name__ == "__main__":
    model = build_1d_cnn_autoencoder()
    model.summary()

