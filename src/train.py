"""
src/train.py
Quy trình huấn luyện mạng 1D-CNN Autoencoder bán giám sát (Semi-supervised Anomaly Detection).
Mạng chỉ được huấn luyện trên mẫu bơi lội bình thường để học phân phối chuẩn động học.
Tính toán ngưỡng động tau = mean + K * std (K = 3.5).
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import tensorflow as tf
from src.config import CONFIG, SystemConfig
from src.model_autoencoder import build_1d_cnn_autoencoder

def train_autoencoder(
    normal_data_path: str,
    model_save_path: str,
    stats_save_path: str,
    config: SystemConfig = CONFIG,
    epochs: int = CONFIG.EPOCHS,
    batch_size: int = CONFIG.BATCH_SIZE
) -> Dict[str, Any]:
    """
    Huấn luyện bán giám sát:
    - normal_data_path: Đường dẫn tệp .npy chứa các cửa sổ (N, 128, 6)
    - model_save_path: Nơi lưu mô hình FP32 (.keras hoặc .h5)
    - stats_save_path: Nơi lưu chỉ số thống kê sai số MSE và ngưỡng tau
    """
    print(f"[*] Đang nạp tập dữ liệu bơi bình thường từ: {normal_data_path}")
    X = np.load(normal_data_path).astype(np.float32)
    print(f"[*] Kích thước tập dữ liệu đầu vào: {X.shape}")

    # Xáo trộn dữ liệu (shuffle) với seed cố định
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]

    # Phân chia tập Train / Validation (85% / 15%)
    val_split = int(0.85 * len(X))
    train_data = X[:val_split]
    val_data = X[val_split:]
    print(f"    - Tập huấn luyện (Train):      {train_data.shape}")
    print(f"    - Tập kiểm chuẩn (Validation): {val_data.shape}")

    # Khởi tạo mô hình Autoencoder
    model = build_1d_cnn_autoencoder(config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=4,
            verbose=1
        )
    ]

    print("[*] Bắt đầu quá trình huấn luyện Autoencoder bán giám sát...")
    history = model.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_data),
        callbacks=callbacks,
        verbose=1
    )

    # Lưu mô hình Keras FP32
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    model.save(model_save_path)
    print(f"[+] Đã lưu mô hình Keras FP32 tại: {model_save_path}")

    # =========================================================================
    # TÍNH TOÁN PHÂN PHỐI SAI SỐ MSE TRÊN TẬP KIỂM CHUẨN (VALIDATION)
    # =========================================================================
    print("[*] Đang tính toán phân phối sai số tái tạo (Reconstruction Error)...")
    val_preds = model.predict(val_data, verbose=0)
    # MSE trên từng cửa sổ: shape (M,)
    val_mse = np.mean(np.square(val_data - val_preds), axis=(1, 2))

    mean_mse = float(np.mean(val_mse))
    std_mse = float(np.std(val_mse))
    threshold_tau = mean_mse + (config.ANOMALY_THRESHOLD_K * std_mse)

    stats = {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "k_factor": config.ANOMALY_THRESHOLD_K,
        "threshold_tau": threshold_tau,
        "train_samples": int(len(train_data)),
        "val_samples": int(len(val_data)),
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1])
    }

    os.makedirs(os.path.dirname(os.path.abspath(stats_save_path)), exist_ok=True)
    with open(stats_save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"[+] Thống kê sai số tái tạo bình thường:")
    print(f"    - Mean MSE: {mean_mse:.6f}")
    print(f"    - Std MSE:  {std_mse:.6f}")
    print(f"    - Ngưỡng dị thường quyết định tau: {threshold_tau:.6f}")
    print(f"[+] Đã lưu thông số ngưỡng tại: {stats_save_path}")

    return stats

if __name__ == "__main__":
    normal_data = str(CONFIG.DATA_PROCESSED_DIR / "normal_swimming.npy")
    model_save = str(CONFIG.MODELS_DIR / "autoencoder_fp32.keras")
    stats_save = str(CONFIG.MODELS_DIR / "stats.json")
    train_autoencoder(normal_data, model_save, stats_save)
