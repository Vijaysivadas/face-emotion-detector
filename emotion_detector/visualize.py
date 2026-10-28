"""OpenCV visualisations for facial-emotion results (all images are RGB)."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EMOTION_STYLE
from .detector import FaceResult

FONT = cv2.FONT_HERSHEY_SIMPLEX


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _ink_for(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Dark text on light fills and white text on dark fills."""
    red, green, blue = (channel / 255 for channel in color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (20, 20, 24) if luminance > 0.42 else (255, 255, 255)


def crop_face(image_rgb: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray:
    """Extract a padded crop safely, including faces near an image edge."""
    x, y, width, height = box
    padding_x, padding_y = round(width * pad), round(height * pad)
    left, top = max(0, x - padding_x), max(0, y - padding_y)
    right = min(image_rgb.shape[1], x + width + padding_x)
    bottom = min(image_rgb.shape[0], y + height + padding_y)
    return image_rgb[top:bottom, left:right].copy()


def _label(
    image: np.ndarray, text: str, anchor_x: int, anchor_y: int, color: tuple[int, int, int], scale: float, thickness: int
) -> int:
    """Draw a filled label whose bottom-left sits at the anchor; returns its height."""
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    pad = max(4, round(text_height * 0.45))
    height, width = image.shape[:2]
    box_height = text_height + baseline + pad * 2
    top = anchor_y - box_height
    if top < 0:  # no room above the face: draw inside its top edge
        top = anchor_y
    left = min(max(0, anchor_x), max(0, width - text_width - pad * 2))
    cv2.rectangle(image, (left, top), (left + text_width + pad * 2, top + box_height), color, -1)
    cv2.putText(
        image, text, (left + pad, top + pad + text_height), FONT, scale, _ink_for(color), thickness, cv2.LINE_AA
    )
    return box_height


def annotate(image_rgb: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
    """Draw numbered, colour-coded boxes and labels without mutating the input."""
    output = image_rgb.copy()
    height, width = output.shape[:2]
    line_width = max(2, round(min(height, width) / 300))
    font_scale = max(0.45, min(height, width) / 900)
    font_thickness = max(1, round(font_scale * 2))

    for index, face in enumerate(faces, start=1):
        x, y, box_width, box_height = face.box
        color = hex_to_rgb(EMOTION_STYLE[face.dominant_emotion]["color"])
        cv2.rectangle(output, (x, y), (x + box_width, y + box_height), color, line_width)
        label = f"#{index} {face.dominant_emotion.title()} {face.confidence:.0f}%"
        _label(output, label, x, y, color, font_scale, font_thickness)
    return output


def _corner_brackets(image: np.ndarray, box: tuple[int, int, int, int], color, thickness: int) -> None:
    x, y, width, height = box
    arm = max(8, round(min(width, height) * 0.22))
    for corner_x, corner_y, dx, dy in (
        (x, y, 1, 1),
        (x + width, y, -1, 1),
        (x, y + height, 1, -1),
        (x + width, y + height, -1, -1),
    ):
        cv2.line(image, (corner_x, corner_y), (corner_x + dx * arm, corner_y), color, thickness, cv2.LINE_AA)
        cv2.line(image, (corner_x, corner_y), (corner_x, corner_y + dy * arm), color, thickness, cv2.LINE_AA)


def _blend_rect(image: np.ndarray, top_left, bottom_right, color, alpha: float) -> None:
    """Translucent filled rectangle, blending only the covered region."""
    left, top = max(0, top_left[0]), max(0, top_left[1])
    right, bottom = min(image.shape[1], bottom_right[0]), min(image.shape[0], bottom_right[1])
    if right <= left or bottom <= top:
        return
    region = image[top:bottom, left:right]
    overlay = np.empty_like(region)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


def draw_live_overlay(frame_rgb: np.ndarray, faces: list[FaceResult], hud: str | None = None) -> np.ndarray:
    """Draw live-video overlays in place: brackets, label and a confidence bar."""
    height, width = frame_rgb.shape[:2]
    scale = min(height, width) / 480
    font_scale = max(0.42, 0.6 * scale)
    thickness = max(1, round(1.6 * scale))
    line = max(2, round(3 * scale))

    for face in faces:
        x, y, box_width, box_height = face.box
        color = hex_to_rgb(EMOTION_STYLE[face.dominant_emotion]["color"])
        _corner_brackets(frame_rgb, face.box, color, line)
        _label(frame_rgb, f"{face.dominant_emotion.title()} {face.confidence:.0f}%", x, y - line, color, font_scale, thickness)
        # Confidence bar under the face.
        bar_top = min(height - 1, y + box_height + line * 2)
        bar_height = max(4, round(5 * scale))
        _blend_rect(frame_rgb, (x, bar_top), (x + box_width, bar_top + bar_height), (20, 20, 24), 0.55)
        filled = round(box_width * min(face.confidence, 100) / 100)
        cv2.rectangle(frame_rgb, (x, bar_top), (x + filled, bar_top + bar_height), color, -1)

    if hud:  # bottom-left, where it rarely covers a face label
        (text_width, text_height), baseline = cv2.getTextSize(hud, FONT, font_scale * 0.85, thickness)
        pad = round(8 * scale)
        dot = max(4, round(5 * scale))
        right = pad * 3 + dot * 2 + text_width
        top = height - pad * 3 - text_height - baseline
        _blend_rect(frame_rgb, (pad, top), (right, height - pad), (16, 16, 20), 0.6)
        centre_y = top + pad + text_height // 2 + baseline // 2
        cv2.circle(frame_rgb, (pad * 2 + dot, centre_y), dot, (230, 72, 72), -1, cv2.LINE_AA)
        cv2.putText(
            frame_rgb, hud, (pad * 2 + dot * 2 + pad // 2, top + pad + text_height), FONT, font_scale * 0.85,
            (245, 245, 245), thickness, cv2.LINE_AA,
        )
    return frame_rgb
