from __future__ import annotations

import numpy as np
from PIL import Image

from emotion_detector.detector import FaceResult, _parse_result, overall_mood, prepare_image
from emotion_detector.visualize import annotate, crop_face


def sample_face() -> FaceResult:
    return FaceResult((20, 20, 40, 40), "happy", {"angry": 0, "disgust": 0, "fear": 0, "happy": 93, "sad": 1, "surprise": 4, "neutral": 2})


def test_parses_deepface_record_and_uses_argmax() -> None:
    face = _parse_result({"region": {"x": 2, "y": 3, "w": 10, "h": 11}, "emotion": {"happy": 75, "sad": 25}}, (30, 40, 3))
    assert face.box == (2, 3, 10, 11)
    assert face.dominant_emotion == "happy"
    assert sum(face.scores.values()) == 100


def test_prepare_image_downscales_large_image() -> None:
    image = Image.fromarray(np.zeros((2000, 4000, 3), dtype=np.uint8))
    prepared = prepare_image(image)
    assert max(prepared.shape[:2]) == 1600


def test_annotation_keeps_shape_and_draws_pixels() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = annotate(image, [sample_face()])
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_crop_and_overall_mood() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_face(image, sample_face().box).size > 0
    mood = overall_mood([sample_face()])
    assert mood is not None and mood[0] == "happy"
