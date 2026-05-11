from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.extract_features import extract_features
from src.knn import predict


# =========================
# PILIH GAMBAR TEST
# =========================
SCENARIO = "500"  # ubah ke "750" atau "1000" jika perlu

# IMAGE_PATH = "data/500/test/jeruk_lemon/good_quality_304.jpg"
# IMAGE_PATH = "data/750/test/jeruk_manis/h (306).jpg"
IMAGE_PATH = "data/1000/test/jeruk_nipis/Lime_Whole_0611.jpg"

TRAIN_CSV = "features/train_features.csv"

OUTPUT_DIR = Path("results") / "process_visualization" / SCENARIO
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 3


# =========================
# LOAD DATA TRAIN
# =========================
def load_training_data(csv_path):
    df = pd.read_csv(csv_path)

    print("\n[INFO] Kolom CSV:")
    print(df.columns)

    if "label" not in df.columns:
        raise ValueError("CSV harus punya kolom 'label'.")

    # Ambil kolom fitur yang namanya f1, f2, f3, ...
    feature_cols = [c for c in df.columns if re.match(r"^f\d+$", str(c))]

    if not feature_cols:
        # fallback jika format CSV berbeda
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols

    if not feature_cols:
        raise ValueError("Tidak ada kolom fitur numerik yang ditemukan di CSV.")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].astype(str).values

    return X, y


# =========================
# STANDARDISASI
# =========================
def standardize(train_X, test_X):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0

    train_scaled = (train_X - mean) / std
    test_scaled = (test_X - mean) / std

    return train_scaled, test_scaled


# =========================
# PROSES GAMBAR UNTUK VISUALISASI
# =========================
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    resized = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    return {
        "original": img,
        "resized": resized,
        "gray": gray,
        "blur": blur,
        "edges": edges,
    }


# =========================
# SIMPAN VISUALISASI
# =========================
def save_visualization(process_data, prediction, image_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Proses Pengolahan Citra dan Klasifikasi KNN\nPrediksi: {prediction}",
        fontsize=15,
        fontweight="bold"
    )

    # Original
    axes[0, 0].imshow(cv2.cvtColor(process_data["original"], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Resize
    axes[0, 1].imshow(cv2.cvtColor(process_data["resized"], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Resize 200x200")
    axes[0, 1].axis("off")

    # Grayscale
    axes[0, 2].imshow(process_data["gray"], cmap="gray")
    axes[0, 2].set_title("Grayscale")
    axes[0, 2].axis("off")

    # Blur
    axes[1, 0].imshow(process_data["blur"], cmap="gray")
    axes[1, 0].set_title("Gaussian Blur")
    axes[1, 0].axis("off")

    # Edges
    axes[1, 1].imshow(process_data["edges"], cmap="gray")
    axes[1, 1].set_title("Edge Detection")
    axes[1, 1].axis("off")

    # Info text
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.85,
        f"Image:\n{Path(image_path).name}\n\n"
        f"Proses:\n"
        f"1. Original\n"
        f"2. Resize\n"
        f"3. Grayscale\n"
        f"4. Blur\n"
        f"5. Edge Detection\n\n"
        f"Hasil akhir:\n{prediction}",
        fontsize=12,
        va="top"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = OUTPUT_DIR / "hasil_visualisasi.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Visualisasi disimpan ke: {output_path}")


# =========================
# MAIN
# =========================
def main():
    print("=== Membaca gambar ===")
    process_data = process_image(IMAGE_PATH)

    print("=== Ekstraksi fitur dari gambar test ===")
    test_features = extract_features(IMAGE_PATH)
    test_features = np.array(test_features, dtype=np.float32)

    print("=== Load training data ===")
    train_X, train_y = load_training_data(TRAIN_CSV)

    if train_X.shape[1] != len(test_features):
        raise ValueError(
            f"Jumlah fitur train ({train_X.shape[1]}) tidak sama dengan fitur test ({len(test_features)}). "
            f"Pastikan src.extract_features.py sama untuk training dan visualisasi."
        )

    print("=== Standardisasi fitur ===")
    train_X_scaled, test_X_scaled = standardize(train_X, test_features.reshape(1, -1))

    print(f"=== Prediksi dengan KNN (K={K}) ===")
    prediction = predict(train_X_scaled, train_y, test_X_scaled, k=K)[0]

    print(f"[HASIL] Prediksi: {prediction}")

    print("=== Membuat visualisasi ===")
    save_visualization(process_data, prediction, IMAGE_PATH)

    print("[SELESAI]")


if __name__ == "__main__":
    main()