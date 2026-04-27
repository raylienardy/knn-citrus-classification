import numpy as np
from collections import Counter


def predict_one(train_X, train_y, x, k=3):
    distances = np.linalg.norm(train_X - x, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = train_y[nearest_idx]

    counts = Counter(nearest_labels)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]

    if len(candidates) > 1:
        for idx in nearest_idx:
            if train_y[idx] in candidates:
                return train_y[idx]

    return counts.most_common(1)[0][0]


def predict(train_X, train_y, test_X, k=3):
    preds = []
    for x in test_X:
        preds.append(predict_one(train_X, train_y, x, k=k))
    return np.array(preds)