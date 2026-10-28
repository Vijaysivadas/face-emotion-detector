"""Measure the deployed DeepFace emotion model and save the Model insights figures.

Everything here runs the app's own pipeline code on FER-2013 (downloaded on
first use) and writes reports/model_report.json, which the app plots:
accuracy per split, confusion matrix, per-class metrics, calibration,
the preprocessing ablation, the crop-framing study and a detector benchmark.

Example: python scripts/build_model_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fer2013 import DEFAULT_DATA_DIR, load_split  # noqa: E402

UPSCALED = 192  # FER faces are upscaled 4x and padded so detectors see a photo-like face
PAD = 96


def canvas(face: np.ndarray) -> np.ndarray:
    big = cv2.resize(face, (UPSCALED, UPSCALED), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.copyMakeBorder(big, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE), cv2.COLOR_GRAY2RGB)


def deepface_letterbox(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """DeepFace's own analyze() preprocessing: tight crop, black letterbox, 224 → 48."""
    x, y, width, height = box
    crop = gray[y : y + height, x : x + width]
    factor = 224 / max(crop.shape)
    crop = cv2.resize(crop, (int(crop.shape[1] * factor), int(crop.shape[0] * factor)))
    rows, columns = 224 - crop.shape[0], 224 - crop.shape[1]
    crop = np.pad(crop, ((rows // 2, rows - rows // 2), (columns // 2, columns - columns // 2)))
    return cv2.resize(cv2.resize(crop, (224, 224)), (48, 48)).astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "model_report.json")
    parser.add_argument("--benchmark-samples", type=int, default=400)
    parser.add_argument("--retinaface-samples", type=int, default=60)
    args = parser.parse_args()

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    import deepface
    import tensorflow as tf

    from emotion_detector.config import CROP_SCALE, DISPLAY_ORDER, EMOTIONS
    from emotion_detector.detector import (
        YuNetDetector,
        detect_faces,
        emotion_classifier,
        face_to_model_input,
        load_model,
    )

    classifier = emotion_classifier()
    splits = {name: load_split(name, args.data_dir) for name in ("train", "publicTest", "privateTest")}
    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"deepface": deepface.__version__, "tensorflow": tf.__version__, "opencv": cv2.__version__},
        "labels": list(EMOTIONS),
        "crop_scale": CROP_SCALE,
    }
    report["dataset"] = {
        name: {emotion: int((labels == index).sum()) for index, emotion in enumerate(EMOTIONS)}
        for name, (_, labels) in splits.items()
    }

    def predict(images: np.ndarray, tta: bool) -> np.ndarray:
        return np.concatenate(
            [classifier.predict(images[start : start + 2048], tta=tta) for start in range(0, len(images), 2048)]
        )

    print("Classifying all splits…")
    report["accuracy_by_split"] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, (images, labels) in splits.items():
        scaled = images.astype(np.float32) / 255.0
        single, mirrored = predict(scaled, False), predict(scaled, True)
        probabilities[name] = mirrored
        report["accuracy_by_split"][name] = {
            "single": float((single.argmax(1) == labels).mean()),
            "tta": float((mirrored.argmax(1) == labels).mean()),
            "n": int(len(labels)),
        }

    images, labels = splits["privateTest"]
    probs = probabilities["privateTest"]
    predicted = probs.argmax(1)
    confidence = probs.max(1)
    correct = predicted == labels
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=range(7), zero_division=0)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    bins = np.linspace(0, 1, 11)
    reliability = []
    calibration_error = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            calibration_error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        reliability.append(
            {
                "bin_start": float(low),
                "bin_end": float(high),
                "count": int(mask.sum()),
                "confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    histogram_edges = np.linspace(0, 1, 21)
    report["test"] = {
        "split": "privateTest",
        "accuracy": float(correct.mean()),
        "top2_accuracy": float((top2 == labels[:, None]).any(axis=1).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "expected_calibration_error": float(calibration_error),
        "mean_confidence": float(confidence.mean()),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=range(7)).tolist(),
        "per_class": {
            emotion: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, emotion in enumerate(EMOTIONS)
        },
        "reliability": reliability,
        "confidence_histogram": {
            "edges": histogram_edges.tolist(),
            "correct": np.histogram(confidence[correct], histogram_edges)[0].tolist(),
            "incorrect": np.histogram(confidence[~correct], histogram_edges)[0].tolist(),
        },
    }

    raw = predict(images.astype(np.float32), False)  # the old live path fed 0–255 pixels
    report["ablation"] = [
        {"variant": "Raw 0–255 pixels (old live mode)", "accuracy": float((raw.argmax(1) == labels).mean())},
        {"variant": "DeepFace preprocessing ([0, 1])", "accuracy": report["accuracy_by_split"]["privateTest"]["single"]},
        {"variant": "+ mirror test-time augmentation", "accuracy": report["accuracy_by_split"]["privateTest"]["tta"]},
    ]

    print("Framing study…")
    yunet = YuNetDetector()
    canvases, detections, kept = [], [], []
    for index, face in enumerate(images):
        rgb = canvas(face)
        found = yunet.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 1280)
        if found:
            canvases.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            detections.append(max(found, key=lambda item: item.box[2] * item.box[3]))
            kept.append(index)
    kept_labels = labels[kept]

    def accuracy(inputs: list[np.ndarray], tta: bool) -> float:
        return float((classifier.predict(np.stack(inputs), tta=tta).argmax(1) == kept_labels).mean())

    letterbox = [deepface_letterbox(gray, item.box) for gray, item in zip(canvases, detections)]
    framing = []
    for scale in (0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4):
        inputs = [face_to_model_input(gray, item.box, item.eyes, True, scale) for gray, item in zip(canvases, detections)]
        framing.append({"scale": scale, "variant": "Single pass", "accuracy": accuracy(inputs, False)})
        framing.append({"scale": scale, "variant": "With mirror TTA", "accuracy": accuracy(inputs, True)})
    unaligned = [face_to_model_input(gray, item.box, item.eyes, False) for gray, item in zip(canvases, detections)]
    report["framing"] = {
        "detected": len(kept),
        "total": int(len(images)),
        "deepface_letterbox": {"single": accuracy(letterbox, False), "tta": accuracy(letterbox, True)},
        "app_unaligned_tta": accuracy(unaligned, True),
        "points": framing,
    }

    print("Detector benchmark…")
    names = {"yunet": "YuNet", "ssd": "SSD", "opencv": "OpenCV Haar", "mtcnn": "MTCNN", "retinaface": "RetinaFace"}
    benchmark = []
    for name, label in names.items():
        count = args.retinaface_samples if name == "retinaface" else args.benchmark_samples
        detect_faces(canvas(images[0]), name)  # load weights before timing
        found_count, hits, elapsed = 0, 0, 0.0
        inputs, targets = [], []
        for index in range(count):
            rgb = canvas(images[index])
            started = time.perf_counter()
            found = detect_faces(rgb, name)
            elapsed += time.perf_counter() - started
            if found:
                found_count += 1
                best = max(found, key=lambda item: item.box[2] * item.box[3])
                inputs.append(face_to_model_input(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), best.box, best.eyes))
                targets.append(labels[index])
        if inputs:
            hits = int((classifier.predict(np.stack(inputs)).argmax(1) == np.array(targets)).sum())
        benchmark.append(
            {
                "detector": label,
                "samples": count,
                "detection_rate": found_count / count,
                "accuracy_on_detected": hits / max(found_count, 1),
                "end_to_end_accuracy": hits / count,
                "ms_per_image": elapsed / count * 1000,
            }
        )
        print(f"  {label}: {found_count}/{count} found, {elapsed / count * 1000:.1f} ms")
    report["detectors"] = benchmark

    print("Inference speed…")
    batch = np.random.default_rng(0).random((8, 48, 48)).astype(np.float32)
    keras_model = load_model().model

    def timed(function, repeats: int = 50) -> float:
        function()
        started = time.perf_counter()
        for _ in range(repeats):
            function()
        return (time.perf_counter() - started) / repeats * 1000

    report["speed_ms"] = [
        {"setup": "DeepFace default (eager, 1 face)", "ms": timed(lambda: keras_model(batch[:1, ..., None], training=False))},
        {"setup": "Compiled call, 1 face", "ms": timed(lambda: classifier.predict(batch[:1], tta=False))},
        {"setup": "Compiled call, 1 face + mirror TTA", "ms": timed(lambda: classifier.predict(batch[:1], tta=True))},
        {"setup": "Compiled call, 8 faces + mirror TTA", "ms": timed(lambda: classifier.predict(batch, tta=True))},
    ]
    report["display_order"] = list(DISPLAY_ORDER)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PrivateTest accuracy {report['test']['accuracy']:.2%} (TTA). Saved {args.output}")


if __name__ == "__main__":
    main()
