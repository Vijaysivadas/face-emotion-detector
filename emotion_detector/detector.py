"""DeepFace integration and image-preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import EMOTIONS, MAX_IMAGE_SIDE


@dataclass(frozen=True)
class FaceResult:
    """A serialisable, UI-friendly result for one detected face."""

    box: tuple[int, int, int, int]
    dominant_emotion: str
    scores: dict[str, float]
    face_confidence: float | None = None

    @property
    def confidence(self) -> float:
        return float(self.scores.get(self.dominant_emotion, 0.0))


def load_model() -> Any:
    """Build the emotion network so its weights download/load only once per app."""
    # DeepFace logs emoji while downloading its weights. Windows PowerShell may
    # otherwise use CP1252 and raise UnicodeEncodeError before the download starts.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    from deepface import DeepFace

    # DeepFace 0.0.103 identifies this network within its facial-attribute
    # model family (rather than its default facial-recognition family).
    return DeepFace.build_model(task="facial_attribute", model_name="Emotion")


def prepare_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return an RGB image with orientation corrected and a bounded long side."""
    if isinstance(image, Image.Image):
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(prepared)
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        elif rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected a grayscale, RGB, or RGBA image.")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    height, width = rgb.shape[:2]
    longest_side = max(height, width)
    if longest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest_side
        rgb = cv2.resize(
            rgb,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(rgb)


def _no_face_error(error: Exception) -> bool:
    """DeepFace has changed its no-face exception path over releases."""
    name = type(error).__name__.lower()
    message = str(error).lower()
    return "facenotdetected" in name or "face could not be detected" in message


def _parse_result(raw: dict[str, Any], image_shape: tuple[int, ...]) -> FaceResult:
    region = raw.get("region") or {}
    image_height, image_width = image_shape[:2]
    x = min(image_width - 1, max(0, int(round(float(region.get("x", 0))))))
    y = min(image_height - 1, max(0, int(round(float(region.get("y", 0))))))
    width = max(1, int(round(float(region.get("w", image_width)))))
    height = max(1, int(round(float(region.get("h", image_height)))))
    width = min(width, image_width - x)
    height = min(height, image_height - y)

    raw_scores = raw.get("emotion") or {}
    scores = {emotion: float(raw_scores.get(emotion, 0.0)) for emotion in EMOTIONS}
    dominant = str(raw.get("dominant_emotion") or max(scores, key=scores.get)).lower()
    if dominant not in scores:
        dominant = max(scores, key=scores.get)
    face_confidence = raw.get("face_confidence")
    return FaceResult(
        box=(x, y, width, height),
        dominant_emotion=dominant,
        scores=scores,
        face_confidence=None if face_confidence is None else float(face_confidence),
    )


def analyze(
    image_rgb: np.ndarray,
    detector: str = "opencv",
    align: bool = True,
    whole_image_fallback: bool = False,
) -> list[FaceResult]:
    """Detect and classify every face, returning results from left to right."""
    from deepface import DeepFace

    image_rgb = prepare_image(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    options = {
        "actions": ["emotion"],
        "detector_backend": detector,
        "align": align,
        "enforce_detection": True,
        "silent": True,
    }
    try:
        response = DeepFace.analyze(img_path=image_bgr, **options)
    except Exception as error:
        if not _no_face_error(error):
            raise
        if not whole_image_fallback:
            return []
        options["detector_backend"] = "skip"
        options["enforce_detection"] = False
        response = DeepFace.analyze(img_path=image_bgr, **options)

    records = response if isinstance(response, list) else [response]
    faces = [_parse_result(record, image_rgb.shape) for record in records]
    return sorted(faces, key=lambda face: (face.box[0], face.box[1]))


def overall_mood(faces: list[FaceResult]) -> tuple[str, dict[str, float]] | None:
    """Average individual emotion distributions for a group-photo summary."""
    if not faces:
        return None
    averages = {
        emotion: sum(face.scores.get(emotion, 0.0) for face in faces) / len(faces)
        for emotion in EMOTIONS
    }
    return max(averages, key=averages.get), averages
