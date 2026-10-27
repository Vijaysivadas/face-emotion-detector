"""OpenCV visualisations for facial-emotion results."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EMOTION_STYLE
from .detector import FaceResult


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def crop_face(image_rgb: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray:
    """Extract a padded crop safely, including faces near an image edge."""
    x, y, width, height = box
    padding_x, padding_y = round(width * pad), round(height * pad)
    left, top = max(0, x - padding_x), max(0, y - padding_y)
    right = min(image_rgb.shape[1], x + width + padding_x)
    bottom = min(image_rgb.shape[0], y + height + padding_y)
    return image_rgb[top:bottom, left:right].copy()


def annotate(image_rgb: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
    """Draw accessible colour-coded boxes and labels without mutating the input."""
    output = image_rgb.copy()
    height, width = output.shape[:2]
    line_width = max(2, round(min(height, width) / 350))
    font_scale = max(0.45, min(height, width) / 950)
    font_thickness = max(1, round(font_scale * 2))

    for index, face in enumerate(faces, start=1):
        x, y, box_width, box_height = face.box
        style = EMOTION_STYLE[face.dominant_emotion]
        color = hex_to_bgr(style["color"])
        cv2.rectangle(output, (x, y), (x + box_width, y + box_height), color, line_width)
        label = f"#{index} {face.dominant_emotion.title()} {face.confidence:.0f}%"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        label_top = max(0, y - label_height - baseline - 10)
        label_bottom = label_top + label_height + baseline + 10
        cv2.rectangle(output, (x, label_top), (x + label_width + 12, label_bottom), color, -1)
        text_color = (20, 20, 20) if face.dominant_emotion in {"happy", "neutral"} else (255, 255, 255)
        cv2.putText(
            output, label, (x + 6, label_bottom - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, font_thickness, cv2.LINE_AA,
        )
    return output
