import numpy as np


def confusion_matrix(y_true, y_pred, labels):
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            i = label_to_idx[t]
            j = label_to_idx[p]
            cm[i][j] += 1

    return cm


def calculate_metrics_from_cm(cm, labels):
    total = cm.sum()
    results = []

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fp - fn

        accuracy = (tp + tn) / total if total else 0
        error_rate = 1 - accuracy
        recall = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        results.append({
            "label": label,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Accuracy": accuracy,
            "Error Rate": error_rate,
            "Recall": recall,
            "Specificity": specificity,
            "Precision": precision,
            "F1-Score": f1,
        })

    overall_accuracy = np.trace(cm) / total if total else 0
    overall_error_rate = 1 - overall_accuracy

    return results, overall_accuracy, overall_error_rate


def format_confusion_matrix(cm, labels):
    lines = []
    lines.append("Confusion Matrix")
    lines.append("Aktual \\ Prediksi")
    header = [" " * 15] + [f"{label:>15}" for label in labels]
    lines.append("".join(header))

    for i, label in enumerate(labels):
        row = [f"{label:>15}"] + [f"{cm[i, j]:>15}" for j in range(len(labels))]
        lines.append("".join(row))

    return "\n".join(lines)


def format_metrics_table(results, overall_accuracy, overall_error_rate):
    lines = []
    lines.append("Hasil Evaluasi Per Kelas")
    lines.append("-" * 90)

    for r in results:
        lines.append(f"Kelas        : {r['label']}")
        lines.append(f"TP           : {r['TP']}")
        lines.append(f"FP           : {r['FP']}")
        lines.append(f"FN           : {r['FN']}")
        lines.append(f"TN           : {r['TN']}")
        lines.append(f"Accuracy     : {r['Accuracy']:.4f}")
        lines.append(f"Error Rate   : {r['Error Rate']:.4f}")
        lines.append(f"Recall       : {r['Recall']:.4f}")
        lines.append(f"Specificity  : {r['Specificity']:.4f}")
        lines.append(f"Precision    : {r['Precision']:.4f}")
        lines.append(f"F1-Score     : {r['F1-Score']:.4f}")
        lines.append("-" * 90)

    lines.append(f"Overall Accuracy   : {overall_accuracy:.4f}")
    lines.append(f"Overall Error Rate : {overall_error_rate:.4f}")

    return "\n".join(lines)