"""Re-train DeepFace's emotion CNN architecture from scratch to record learning curves.

DeepFace publishes pre-trained weights but no training logs. This script
clones the exact architecture (fresh random weights), trains it on the
FER-2013 Training split, validates on PublicTest and finally tests on
PrivateTest. The resulting history is shown on the app's Model insights page.

The app itself keeps using DeepFace's official pre-trained weights.

Example: python scripts/train_reproduction.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from fer2013 import DEFAULT_DATA_DIR, load_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "training_history.json")
    args = parser.parse_args()

    import tensorflow as tf

    from emotion_detector.detector import load_model

    tf.keras.utils.set_random_seed(args.seed)
    x_train, y_train = load_split("train", args.data_dir)
    x_val, y_val = load_split("publicTest", args.data_dir)
    x_test, y_test = load_split("privateTest", args.data_dir)

    def prepare(images: np.ndarray) -> np.ndarray:
        # Same scaling DeepFace applies before its emotion CNN.
        return (images.astype("float32") / 255.0)[..., None]

    # clone_model copies the layer graph only, so training starts from scratch.
    model = tf.keras.models.clone_model(load_model().model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    epoch_times: list[float] = []
    best = {"val_accuracy": -1.0, "weights": None, "epoch": 0}

    class Recorder(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            epoch_times.append(time.perf_counter() - self.started)
            if logs["val_accuracy"] > best["val_accuracy"]:
                best.update(val_accuracy=float(logs["val_accuracy"]), weights=model.get_weights(), epoch=epoch + 1)

    history = model.fit(
        prepare(x_train),
        tf.keras.utils.to_categorical(y_train, 7),
        validation_data=(prepare(x_val), tf.keras.utils.to_categorical(y_val, 7)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[Recorder()],
        verbose=2,
    )
    final_test = float((model.predict(prepare(x_test), verbose=0).argmax(1) == y_test).mean())
    model.set_weights(best["weights"])
    best_test = float((model.predict(prepare(x_test), verbose=0).argmax(1) == y_test).mean())

    epochs = [
        {
            "epoch": index + 1,
            "loss": float(history.history["loss"][index]),
            "accuracy": float(history.history["accuracy"][index]),
            "val_loss": float(history.history["val_loss"][index]),
            "val_accuracy": float(history.history["val_accuracy"][index]),
            "seconds": round(epoch_times[index], 2),
        }
        for index in range(len(history.history["loss"]))
    ]
    report = {
        "description": "DeepFace Emotion CNN architecture re-trained from random initialisation on FER-2013.",
        "config": {
            "train_split": f"FER-2013 Training ({len(y_train):,} faces)",
            "validation_split": f"FER-2013 PublicTest ({len(y_val):,} faces)",
            "test_split": f"FER-2013 PrivateTest ({len(y_test):,} faces)",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "loss": "categorical cross-entropy",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "augmentation": "none",
            "input": "48 x 48 x 1 grayscale, scaled to [0, 1]",
            "seed": args.seed,
            "trainable_parameters": int(model.count_params()),
        },
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "test_accuracy_best_epoch": best_test,
        "test_accuracy_final_epoch": final_test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best epoch {best['epoch']}: val {best['val_accuracy']:.2%}, test {best_test:.2%}. Saved {args.output}")


if __name__ == "__main__":
    main()
