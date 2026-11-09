"""
evaluate_ai.py
Kịch bản đánh giá toàn diện mô hình AI TinyML (1D-CNN Autoencoder INT8)
sử dụng scikit-learn (Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix)
và kiểm định máy trạng thái FSM 3 cửa sổ liên tiếp trên nRF52840.
"""
import os
import sys
from pathlib import Path

# Thiết lập thư mục gốc và KERAS_HOME trước khi import TensorFlow
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import CONFIG
from src.edge_simulator import EdgeWearableNodeSimulator
from src.evaluate_f1 import evaluate_model_f1


def run_sklearn_evaluation() -> None:
    # Chạy bộ đánh giá chuẩn từ src.evaluate_f1
    report = evaluate_model_f1()

    # Đánh giá chuyên sâu bổ sung bằng scikit-learn (ROC-AUC & Classification Report)
    normal_path = CONFIG.DATA_PROCESSED_DIR / "normal_swimming.npy"
    anomaly_path = CONFIG.DATA_PROCESSED_DIR / "anomaly_swimming.npy"
    tflite_path = CONFIG.MODELS_DIR / "autoencoder_int8.tflite"
    stats_path = CONFIG.MODELS_DIR / "stats.json"

    normal_windows = np.load(normal_path).astype(np.float32)[-200:]
    anomaly_windows = np.load(anomaly_path).astype(np.float32)

    sim = EdgeWearableNodeSimulator(str(tflite_path), str(stats_path))
    tau = sim.threshold_tau

    y_true = np.concatenate([
        np.zeros(len(normal_windows), dtype=int),
        np.ones(len(anomaly_windows), dtype=int),
    ])

    mse_scores = []
    for w in np.concatenate([normal_windows, anomaly_windows], axis=0):
        mse, _ = sim.infer_window_mse(w)
        mse_scores.append(mse)
    mse_scores = np.array(mse_scores, dtype=np.float32)

    y_pred = (mse_scores > tau).astype(int)

    roc_auc = roc_auc_score(y_true, mse_scores)
    cm = confusion_matrix(y_true, y_pred)

    # Tìm ngưỡng tối ưu F1-Score qua Precision-Recall Curve (scikit-learn)
    precisions, recalls, thresholds = precision_recall_curve(y_true, mse_scores)
    f1_Curve = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_Curve))
    best_tau = float(thresholds[best_idx])
    y_pred_opt = (mse_scores >= best_tau).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)

    print("=" * 80)
    print(" BÁO CÁO PHÂN LOẠI CHI TIẾT (SCIKIT-LEARN CLASSIFICATION REPORT)")
    print("=" * 80)
    print(f" [A] Tại ngưỡng thống kê mặc định (tau = {tau:.6f}):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Bơi bình thường (0)", "Đuối nước IDR (1)"],
            digits=4,
        )
    )
    print(f" [B] Tại ngưỡng tối ưu PR-Curve (optimal_tau = {best_tau:.6f}):")
    print(
        classification_report(
            y_true,
            y_pred_opt,
            target_names=["Bơi bình thường (0)", "Đuối nước IDR (1)"],
            digits=4,
        )
    )
    print(f" [*] Chỉ số diện tích dưới đường cong ROC (ROC-AUC): {roc_auc:.6f}")
    print(f" [*] Ma trận nhầm lẫn tại optimal_tau ({best_tau:.6f}):")
    print(f"     [[TN={cm_opt[0, 0]:4d}, FP={cm_opt[0, 1]:4d}]")
    print(f"      [FN={cm_opt[1, 0]:4d}, TP={cm_opt[1, 1]:4d}]]")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_sklearn_evaluation()
