from pathlib import Path
import time
import numpy as np

from src.extract_features import build_dataset, load_dataset
from src.knn import predict
from src.evaluate import (
    confusion_matrix,
    calculate_metrics_from_cm,
    format_confusion_matrix,
    format_metrics_table
)


# disini bebas mau ubah bagaimana, pastikan nama foldernya sesuai
# 500
TRAIN_DIR = "data/500/train"
VAL_DIR = "data/500/val"
TEST_DIR = "data/500/test"

# 750
# TRAIN_DIR = "data/750/train"
# VAL_DIR = "data/750/val"
# TEST_DIR = "data/750/test"

# 1000
# TRAIN_DIR = "data/1000/train"
# VAL_DIR = "data/1000/val"
# TEST_DIR = "data/1000/test"

FEATURES_DIR = "features"
RESULTS_DIR = "results"

TRAIN_CSV = f"{FEATURES_DIR}/train_features.csv"
VAL_CSV = f"{FEATURES_DIR}/val_features.csv"
TEST_CSV = f"{FEATURES_DIR}/test_features.csv"

K_CANDIDATES = [1, 3, 5, 7, 9]


def standardize(train_X, other_X):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return (train_X - mean) / std, (other_X - mean) / std


def save_text_file(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def evaluate_dataset(train_X, train_y, eval_X, eval_y, labels, k):
    pred = predict(train_X, train_y, eval_X, k=k)
    cm = confusion_matrix(eval_y, pred, labels)
    results, overall_acc, overall_err = calculate_metrics_from_cm(cm, labels)
    return pred, cm, results, overall_acc, overall_err


def main():
    Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    start_all = time.time()

    print("=== Ekstraksi fitur train ===")
    build_dataset(TRAIN_DIR, TRAIN_CSV)

    print("=== Ekstraksi fitur validation ===")
    build_dataset(VAL_DIR, VAL_CSV)

    print("=== Ekstraksi fitur test ===")
    build_dataset(TEST_DIR, TEST_CSV)

    print("=== Load dataset ===")
    X_train, y_train, _, _ = load_dataset(TRAIN_CSV)
    X_val, y_val, _, _ = load_dataset(VAL_CSV)
    X_test, y_test, _, _ = load_dataset(TEST_CSV)

    labels = sorted(list(set(y_train.tolist()) | set(y_val.tolist()) | set(y_test.tolist())))

    print("=== Standardisasi fitur ===")
    X_train_norm, X_val_norm = standardize(X_train, X_val)
    _, X_test_norm = standardize(X_train, X_test)

    print("=== Cari K terbaik dari validation ===")
    best_k = None
    best_val_acc = -1
    best_val_cm_text = ""
    best_val_metrics_text = ""

    for k in K_CANDIDATES:
        print(f"  - Evaluasi k={k}")
        _, cm_val, results_val, val_acc, val_err = evaluate_dataset(
            X_train_norm, y_train, X_val_norm, y_val, labels, k
        )

        print(f"    Validation Accuracy = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            best_val_cm_text = format_confusion_matrix(cm_val, labels)
            best_val_metrics_text = format_metrics_table(results_val, val_acc, val_err)

    print(f"\n[INFO] K terbaik dari validation: {best_k} dengan akurasi {best_val_acc:.4f}")

    print("\n=== Evaluasi akhir pada test ===")
    y_pred_test, cm_test, results_test, test_acc, test_err = evaluate_dataset(
        X_train_norm, y_train, X_test_norm, y_test, labels, best_k
    )

    cm_test_text = format_confusion_matrix(cm_test, labels)
    metrics_test_text = format_metrics_table(results_test, test_acc, test_err)

    print("\n--- Validation Result ---")
    print(best_val_cm_text)
    print(best_val_metrics_text)

    print("\n--- Test Result ---")
    print(cm_test_text)
    print(metrics_test_text)

    save_text_file(f"{RESULTS_DIR}/validation_confusion_matrix.txt", best_val_cm_text)
    save_text_file(f"{RESULTS_DIR}/validation_metrics.txt", best_val_metrics_text)
    save_text_file(f"{RESULTS_DIR}/test_confusion_matrix.txt", cm_test_text)
    save_text_file(f"{RESULTS_DIR}/test_metrics.txt", metrics_test_text)

    print(f"\n[OK] Hasil disimpan ke folder '{RESULTS_DIR}'")
    print(f"[TOTAL TIME] {time.time() - start_all:.2f} detik")


if __name__ == "__main__":
    main()