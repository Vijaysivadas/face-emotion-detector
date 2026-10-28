"""Introspection of DeepFace's emotion CNN: layers, weights and Grad-CAM."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

LAYER_KINDS = {
    "Conv2D": "Convolution",
    "MaxPooling2D": "Max pooling",
    "AveragePooling2D": "Average pooling",
    "Flatten": "Flatten",
    "Dense": "Dense",
    "Dropout": "Dropout",
}


def _output_shape(layer: Any) -> tuple[int, ...]:
    shape = layer.output.shape if hasattr(layer, "output") else layer.output_shape
    return tuple(int(value) for value in shape[1:])


def layer_table(model: Any) -> pd.DataFrame:
    """One row per layer: type, output shape, activations and parameters."""
    rows = []
    total = model.count_params()
    for index, layer in enumerate(model.layers, start=1):
        kind = type(layer).__name__
        shape = _output_shape(layer)
        config = layer.get_config()
        detail = ""
        if kind == "Conv2D":
            detail = f"{config['filters']} filters · {config['kernel_size'][0]}×{config['kernel_size'][1]} · {config['activation']}"
        elif kind.endswith("Pooling2D"):
            detail = f"{config['pool_size'][0]}×{config['pool_size'][1]} · stride {config['strides'][0]}"
        elif kind == "Dense":
            detail = f"{config['units']} units · {config['activation']}"
        elif kind == "Dropout":
            detail = f"rate {config['rate']}"
        params = layer.count_params()
        rows.append(
            {
                "#": index,
                "layer": layer.name,
                "type": LAYER_KINDS.get(kind, kind),
                "details": detail,
                "output shape": " × ".join(map(str, shape)),
                "activations": int(np.prod(shape)),
                "parameters": int(params),
                "share of parameters": params / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def weight_samples(model: Any, per_layer: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A random sample of kernel weights per trainable layer, for histograms."""
    generator = np.random.default_rng(seed)
    frames = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0].ravel()
        sample = generator.choice(kernel, size=min(per_layer, kernel.size), replace=False)
        frames.append(pd.DataFrame({"layer": layer.name, "weight": sample}))
    return pd.concat(frames, ignore_index=True)


def weight_statistics(model: Any) -> pd.DataFrame:
    rows = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        rows.append(
            {
                "layer": layer.name,
                "kernel shape": " × ".join(map(str, kernel.shape)),
                "mean": float(kernel.mean()),
                "std": float(kernel.std()),
                "min": float(kernel.min()),
                "max": float(kernel.max()),
                "near-zero (|w| < 1e-3)": float((np.abs(kernel) < 1e-3).mean()),
            }
        )
    return pd.DataFrame(rows)


def _diverging(values: np.ndarray) -> np.ndarray:
    """Map [-1, 1] to blue → light grey → orange (RGB uint8)."""
    negative = np.array([42, 120, 214], dtype=float)
    middle = np.array([226, 226, 222], dtype=float)
    positive = np.array([235, 104, 52], dtype=float)
    values = np.clip(values, -1, 1)[..., None]
    colors = np.where(values < 0, middle + (negative - middle) * -values, middle + (positive - middle) * values)
    return colors.astype(np.uint8)


def first_layer_filters(model: Any, cell: int = 30, gap: int = 4) -> np.ndarray:
    """The 64 learned 5×5 filters of the first convolution as an 8×8 grid."""
    kernel = model.layers[0].get_weights()[0][:, :, 0, :]  # (5, 5, filters)
    filters = kernel.shape[-1]
    columns = 8
    rows = int(np.ceil(filters / columns))
    grid = np.full((rows * (cell + gap) - gap, columns * (cell + gap) - gap, 3), 255, np.uint8)
    for index in range(filters):
        weights = kernel[:, :, index]
        scaled = weights / (np.abs(weights).max() + 1e-9)
        tile = cv2.resize(_diverging(scaled), (cell, cell), interpolation=cv2.INTER_NEAREST)
        row, column = divmod(index, columns)
        grid[row * (cell + gap) : row * (cell + gap) + cell, column * (cell + gap) : column * (cell + gap) + cell] = tile
    return grid


_grad_models: dict[int, Any] = {}


def _grad_model(model: Any) -> tuple[Any, Any]:
    """A model returning the deepest 8×8-or-larger feature map and the logits input."""
    import tensorflow as tf

    key = id(model)
    if key not in _grad_models:
        conv_layers = [layer for layer in model.layers if type(layer).__name__ == "Conv2D"]
        target = [layer for layer in conv_layers if _output_shape(layer)[0] >= 8][-1]
        _grad_models[key] = tf.keras.Model(model.inputs, [target.output, model.layers[-2].output])
    return _grad_models[key], model.layers[-1]


def grad_cam(model: Any, face_input: np.ndarray, class_index: int) -> np.ndarray:
    """Grad-CAM heat map (values in [0, 1]) for one 48×48 face and one class.

    Gradients are taken on the class logit, before the softmax, so a
    confident prediction does not saturate them.
    """
    import tensorflow as tf

    grad_model, output_layer = _grad_model(model)
    kernel, bias = output_layer.get_weights()
    batch = tf.convert_to_tensor(face_input[None, ..., None], tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, hidden = grad_model(batch, training=False)
        logit = tf.matmul(hidden, kernel)[:, class_index] + bias[class_index]
    gradients = tape.gradient(logit, feature_maps)[0]
    weights = tf.reduce_mean(gradients, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feature_maps[0] * weights, axis=-1)).numpy()
    return cam / (cam.max() + 1e-9)


def cam_overlay(face_uint8: np.ndarray, cam: np.ndarray, size: int = 160) -> np.ndarray:
    """Blend a Grad-CAM map over the grayscale model input (RGB uint8)."""
    face = cv2.resize(face_uint8, (size, size), interpolation=cv2.INTER_CUBIC)
    heat = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 0.55, colored, 0.45, 0)
