import argparse
import json
from pathlib import Path

import numpy as np


def load_records(filename, feature_count):
    dtype = np.dtype([("features", "<f4", (feature_count,)), ("label", "u1")])
    byte_size = Path(filename).stat().st_size
    if byte_size == 0 or byte_size % dtype.itemsize:
        raise ValueError(f"Training data has an invalid byte length: {filename}")
    return np.memmap(filename, mode="r", dtype=dtype)


def calculate_metrics(confusion):
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    class_metrics = []
    for index in range(confusion.shape[0]):
        true_positive = int(confusion[index, index])
        actual = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / max(1, predicted)
        recall = true_positive / max(1, actual)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        class_metrics.append(
            {"support": actual, "precision": precision, "recall": recall, "f1": f1}
        )

    note_actual = total - int(confusion[0].sum())
    note_predicted = total - int(confusion[:, 0].sum())
    note_true_positive = int(confusion[1:, 1:].sum())
    note_precision = note_true_positive / max(1, note_predicted)
    note_recall = note_true_positive / max(1, note_actual)
    big_note_classes = (3, 4)
    big_note_actual = sum(int(confusion[index].sum()) for index in big_note_classes)
    big_note_predicted = sum(int(confusion[:, index].sum()) for index in big_note_classes)
    big_note_true_positive = sum(
        int(confusion[actual, predicted])
        for actual in big_note_classes
        for predicted in big_note_classes
    )
    big_note_precision = big_note_true_positive / max(1, big_note_predicted)
    big_note_recall = big_note_true_positive / max(1, big_note_actual)
    return {
        "accuracy": correct / max(1, total),
        "notePresence": {
            "precision": note_precision,
            "recall": note_recall,
            "f1": 2 * note_precision * note_recall / max(
                1e-12, note_precision + note_recall
            ),
        },
        "bigNotePlacement": {
            "precision": big_note_precision,
            "recall": big_note_recall,
            "f1": 2 * big_note_precision * big_note_recall / max(
                1e-12, big_note_precision + big_note_recall
            ),
            "support": big_note_actual,
        },
        "classMetrics": class_metrics,
        "confusionMatrix": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--feature-count", type=int, required=True)
    parser.add_argument("--class-count", type=int, required=True)
    parser.add_argument("--class-counts", required=True)
    args = parser.parse_args()

    training = load_records(args.train, args.feature_count)
    validation = load_records(args.validation, args.feature_count)
    if not len(training) or not len(validation):
        raise ValueError("Training and validation datasets must both contain examples.")

    class_counts = np.fromstring(args.class_counts, sep=",", dtype=np.float64)
    if len(class_counts) != args.class_count or np.any(class_counts < 0):
        raise ValueError("The class-count vector does not match the configured classes.")
    positive_counts = class_counts[1:]
    max_positive = max(1.0, float(positive_counts.max()))
    class_weights = np.ones(args.class_count, dtype=np.float32)
    class_weights[1:] = np.minimum(
        4.0, np.sqrt(max_positive / np.maximum(1.0, positive_counts))
    )

    rng = np.random.default_rng(0x544A4147)
    weights = np.zeros((args.class_count, args.feature_count + 1), dtype=np.float32)
    indices = np.arange(len(training), dtype=np.int64)
    batch_size = 512
    learning_rate = 0.04
    l2 = 0.0001
    for epoch in range(args.epochs):
        rng.shuffle(indices)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = training[batch_indices]
            features = np.asarray(batch["features"], dtype=np.float32)
            labels = np.asarray(batch["label"], dtype=np.int64)
            examples = np.empty((len(labels), args.feature_count + 1), dtype=np.float32)
            examples[:, :-1] = features
            examples[:, -1] = 1.0

            logits = examples @ weights.T
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            sample_weights = class_weights[labels]
            chosen = probabilities[np.arange(len(labels)), labels].copy()
            epoch_loss += float((-np.log(np.maximum(chosen, 1e-12)) * sample_weights).sum())
            errors = probabilities
            errors[np.arange(len(labels)), labels] -= 1.0
            errors *= sample_weights[:, None]
            gradient = errors.T @ examples
            gradient /= max(1.0, float(sample_weights.sum()))
            gradient[:, :-1] += l2 * weights[:, :-1]
            weights -= learning_rate * gradient
            seen += len(labels)
        print(
            f"学習エポック {epoch + 1}/{args.epochs}: "
            f"{seen:,} examples, weighted loss {epoch_loss / max(1, seen):.4f}",
            flush=True,
        )

    confusion = np.zeros((args.class_count, args.class_count), dtype=np.int64)
    for start in range(0, len(validation), batch_size):
        batch = validation[start : start + batch_size]
        features = np.asarray(batch["features"], dtype=np.float32)
        labels = np.asarray(batch["label"], dtype=np.int64)
        examples = np.empty((len(labels), args.feature_count + 1), dtype=np.float32)
        examples[:, :-1] = features
        examples[:, -1] = 1.0
        predictions = np.argmax(examples @ weights.T, axis=1)
        confusion += np.bincount(
            labels * args.class_count + predictions,
            minlength=args.class_count * args.class_count,
        ).reshape(args.class_count, args.class_count)

    model = {
        "formatVersion": 1,
        "classes": args.class_count,
        "features": [
            "rms_before",
            "rms_center",
            "rms_after",
            "rms_local_max",
            "rms_local_mean",
            "onset_before",
            "onset_center",
            "onset_after",
            "onset_local_max",
            "onset_local_mean",
            "measure_phase_sin",
            "measure_phase_cos",
            "beat_phase_sin",
            "beat_phase_cos",
            "bpm",
            "level",
            "course_easy",
            "course_normal",
            "course_hard",
            "course_oni",
            "course_edit",
        ],
        "weights": weights.tolist(),
        "trainingEpochs": args.epochs,
    }
    if len(model["features"]) != args.feature_count:
        raise ValueError("Feature-name count does not match the trainer configuration.")

    Path(args.model).write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    Path(args.metrics).write_text(
        json.dumps(calculate_metrics(confusion), separators=(",", ":")), encoding="utf-8"
    )
    print(
        f"検証: {len(validation):,} examples, "
        f"note F1 {calculate_metrics(confusion)['notePresence']['f1']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
