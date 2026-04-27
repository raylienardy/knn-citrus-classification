import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(image_path, size=(128, 128)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    bgr_mean = np.mean(img, axis=(0, 1))
    bgr_std = np.std(img, axis=(0, 1))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))
    gray_var = float(np.var(gray))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    features = np.concatenate([
        bgr_mean, bgr_std,
        hsv_mean, hsv_std,
        np.array([gray_mean, gray_std, gray_var, lap_var, edge_density], dtype=np.float32)
    ]).astype(np.float32)

    return features


def _iter_image_files(folder_path):
    folder_path = Path(folder_path)
    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def count_images(data_dir):
    data_dir = Path(data_dir)
    total = 0
    for class_folder in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        total += sum(1 for _ in _iter_image_files(class_folder))
    return total


def build_dataset(data_dir, output_csv):
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    class_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    total_images = count_images(data_dir)

    rows = []
    feature_len = None

    print(f"[INFO] Total gambar di {data_dir}: {total_images}")

    with tqdm(total=total_images, desc=f"Ekstraksi {data_dir.name}", unit="img") as pbar:
        for class_folder in class_folders:
            label = class_folder.name
            pbar.set_postfix_str(f"kelas={label}")

            for image_path in _iter_image_files(class_folder):
                try:
                    feats = extract_features(image_path)
                    if feature_len is None:
                        feature_len = len(feats)
                    rows.append([str(image_path), label] + feats.tolist())
                except Exception as e:
                    print(f"[SKIP] {image_path} -> {e}")
                pbar.update(1)

    if not rows:
        raise RuntimeError(f"Tidak ada gambar valid di folder: {data_dir}")

    header = ["path", "label"] + [f"f{i+1}" for i in range(feature_len)]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] Dataset tersimpan ke: {output_csv}")


def load_dataset(csv_path):
    csv_path = Path(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        paths = []
        labels = []
        features = []

        for row in reader:
            if not row:
                continue
            paths.append(row[0])
            labels.append(row[1])
            features.append([float(x) for x in row[2:]])

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    feature_names = header[2:]

    return X, y, paths, feature_names