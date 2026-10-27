r"""Evaluate the pre-trained model on FER-2013 image-folder test data.

Example: python evaluate.py --dataset C:\datasets\fer2013\test
The dataset is intentionally not included in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

from emotion_detector.config import EMOTIONS
from emotion_detector.detector import analyze, prepare_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepFace emotion inference on FER-2013 test images.")
    parser.add_argument("--dataset", type=Path, required=True, help="Folder structured as test/<emotion>/*.jpg")
    parser.add_argument("--output", type=Path, default=Path("evaluation_output"))
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum samples per emotion (0 = all).")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    actual: list[str] = []
    predicted: list[str] = []
    for label in EMOTIONS:
        files = sorted(path for path in (args.dataset / label).glob("*") if path.is_file())
        if args.limit:
            files = files[: args.limit]
        for index, file in enumerate(files, start=1):
            image = prepare_image(Image.open(file))
            result = analyze(image, detector="skip", align=False, whole_image_fallback=True)
            if result:
                actual.append(label)
                predicted.append(result[0].dominant_emotion)
            print(f"{label}: {index}/{len(files)}", end="\r")

    if not actual:
        raise SystemExit("No readable images found. Check --dataset points to test/<emotion> folders.")
    report = pd.DataFrame(classification_report(actual, predicted, labels=list(EMOTIONS), output_dict=True, zero_division=0)).T
    report.to_csv(args.output / "classification_report.csv")
    pd.DataFrame({"actual": actual, "predicted": predicted}).to_csv(args.output / "predictions.csv", index=False)
    matrix = confusion_matrix(actual, predicted, labels=list(EMOTIONS))
    figure, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=EMOTIONS).plot(ax=axis, xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(args.output / "confusion_matrix.png", dpi=180)
    print(f"\nAccuracy: {accuracy_score(actual, predicted):.2%}\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
