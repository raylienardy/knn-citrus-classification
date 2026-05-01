import os
import re
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# PARSE CONFUSION MATRIX
# =========================
def parse_confusion_matrix(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    labels = []
    matrix = []

    for line in lines:
        line = line.strip()

        # skip baris kosong / header
        if not line:
            continue
        if "Confusion Matrix" in line:
            continue
        if "Aktual" in line:
            continue
        if "-" in line:
            continue

        parts = line.split()

        # pastikan baris ini benar-benar data (harus angka di belakang)
        if len(parts) < 2:
            continue

        try:
            values = list(map(int, parts[1:]))
        except:
            continue  # kalau gagal convert, skip

        label = parts[0]

        labels.append(label)
        matrix.append(values)

    return labels, np.array(matrix)

# =========================
# PLOT CONFUSION MATRIX
# =========================
def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =========================
# PARSE METRICS
# =========================
def parse_metrics(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    classes = re.findall(r"Kelas\s*:\s*(\w+)", text)
    precision = re.findall(r"Precision\s*:\s*([0-9.]+)", text)
    recall = re.findall(r"Recall\s*:\s*([0-9.]+)", text)
    f1 = re.findall(r"F1-Score\s*:\s*([0-9.]+)", text)

    precision = list(map(float, precision))
    recall = list(map(float, recall))
    f1 = list(map(float, f1))

    return classes, precision, recall, f1


# =========================
# PLOT METRICS BAR
# =========================
def plot_metrics(classes, precision, recall, f1, title, filename):
    x = np.arange(len(classes))
    width = 0.25

    plt.figure()

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-Score")

    plt.xticks(x, classes)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    # FILE PATH
    test_cm_file = os.path.join(RESULTS_DIR, "test_confusion_matrix.txt")
    test_metrics_file = os.path.join(RESULTS_DIR, "test_metrics.txt")

    val_cm_file = os.path.join(RESULTS_DIR, "validation_confusion_matrix.txt")
    val_metrics_file = os.path.join(RESULTS_DIR, "validation_metrics.txt")

    # =========================
    # CONFUSION MATRIX
    # =========================
    labels, cm = parse_confusion_matrix(test_cm_file)
    plot_confusion_matrix(
        cm, labels,
        "Confusion Matrix (Test)",
        os.path.join(OUTPUT_DIR, "cm_test.png")
    )

    labels, cm = parse_confusion_matrix(val_cm_file)
    plot_confusion_matrix(
        cm, labels,
        "Confusion Matrix (Validation)",
        os.path.join(OUTPUT_DIR, "cm_val.png")
    )

    # =========================
    # METRICS
    # =========================
    classes, p, r, f1 = parse_metrics(test_metrics_file)
    plot_metrics(
        classes, p, r, f1,
        "Metrics per Class (Test)",
        os.path.join(OUTPUT_DIR, "metrics_test.png")
    )

    classes, p, r, f1 = parse_metrics(val_metrics_file)
    plot_metrics(
        classes, p, r, f1,
        "Metrics per Class (Validation)",
        os.path.join(OUTPUT_DIR, "metrics_val.png")
    )

    print(f"[OK] Semua grafik disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()