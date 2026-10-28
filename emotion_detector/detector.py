"""Face detection, alignment and DeepFace emotion classification."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

# RetinaFace switches TensorFlow to the legacy tf-keras API when it is imported.
# Every model in the process must agree, so choose it before TensorFlow loads.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import (
    ALIGNED_FACE_SIZE,
    CROP_SCALE,
    DETECTOR_FALLBACK_ORDER,
    EMOTIONS,
    LIVE_DETECTION_WIDTH,
    LIVE_MIN_FACE_SIZE,
    LIVE_SCORE_THRESHOLD,
    MAX_ALIGN_ANGLE,
    MAX_IMAGE_SIDE,
    MODEL_INPUT_SIZE,
    YUNET_MAX_SIDE,
    YUNET_NMS_THRESHOLD,
    YUNET_SCORE_THRESHOLD,
)

Point = tuple[float, float]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """A face box from a detector, with eye landmarks when available."""

    box: Box
    confidence: float | None = None
    eyes: tuple[Point, Point] | None = None


@dataclass(frozen=True)
class FaceResult:
    """A serialisable, UI-friendly result for one detected face."""

    box: Box
    dominant_emotion: str
    scores: dict[str, float]
    face_confidence: float | None = None
    eyes: tuple[Point, Point] | None = None
    detector: str | None = None

    @property
    def confidence(self) -> float:
        return float(self.scores.get(self.dominant_emotion, 0.0))

    def ranked(self) -> list[tuple[str, float]]:
        """Emotions sorted from most to least likely."""
        return sorted(self.scores.items(), key=lambda item: item[1], reverse=True)


def _utf8_console() -> None:
    """DeepFace logs emoji while downloading weights; Windows' CP1252 console
    would otherwise raise UnicodeEncodeError before the download starts."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def load_model() -> Any:
    """Build DeepFace's emotion network; its weights download on first use."""
    _utf8_console()
    from deepface import DeepFace

    # DeepFace 0.0.103 files this network under its facial-attribute models.
    return DeepFace.build_model(task="facial_attribute", model_name="Emotion")


class EmotionClassifier:
    """DeepFace's pre-trained emotion CNN behind a compiled TensorFlow call.

    DeepFace runs a single face eagerly (~13 ms on a laptop CPU). Tracing the
    same Keras model once with a fixed input signature runs any batch in
    ~2.5 ms, which is what keeps live video responsive.
    """

    def __init__(self, keras_model: Any | None = None) -> None:
        import tensorflow as tf

        self.model = keras_model if keras_model is not None else load_model().model
        signature = [tf.TensorSpec([None, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1], tf.float32)]
        self._call = tf.function(lambda batch: self.model(batch, training=False), input_signature=signature)
        # Trace now rather than on the first video frame.
        self._call(np.zeros((1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1), np.float32))

    def predict(self, faces: np.ndarray, tta: bool = True) -> np.ndarray:
        """Return (n, 7) probabilities for (n, 48, 48) faces scaled to [0, 1].

        With ``tta`` each face is also classified mirrored and the two
        distributions are averaged (+0.7 pp accuracy on FER-2013 PrivateTest).
        """
        faces = np.asarray(faces, dtype=np.float32)
        if faces.ndim == 2:
            faces = faces[None]
        if len(faces) == 0:
            return np.zeros((0, len(EMOTIONS)), np.float32)
        batch = faces[..., None]
        if tta:
            batch = np.concatenate([batch, batch[:, :, ::-1]])
        probabilities = self._call(np.ascontiguousarray(batch)).numpy()
        if tta:
            probabilities = (probabilities[: len(faces)] + probabilities[len(faces) :]) / 2
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-9)


_classifier: EmotionClassifier | None = None
_classifier_lock = threading.Lock()


def emotion_classifier() -> EmotionClassifier:
    """Process-wide classifier; the lock stops two sessions tracing it twice."""
    global _classifier
    with _classifier_lock:
        if _classifier is None:
            _classifier = EmotionClassifier()
        return _classifier


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


def face_to_model_input(
    gray: np.ndarray,
    box: Box,
    eyes: tuple[Point, Point] | None = None,
    align: bool = True,
    scale: float = CROP_SCALE,
) -> np.ndarray:
    """Cut a square, eye-levelled face and return the CNN's 48x48 input in [0, 1].

    DeepFace crops the detector box and pads it to a square with black bars.
    Taking a square region of the photo instead keeps the FER-2013 framing,
    and one affine warp performs rotation, crop and the resize to 224 px.
    """
    x, y, width, height = box
    centre_x, centre_y = x + width / 2, y + height / 2
    side = max(width, height) * scale
    angle = 0.0
    if align and eyes is not None:
        (left_x, left_y), (right_x, right_y) = sorted(eyes)
        angle = math.degrees(math.atan2(right_y - left_y, right_x - left_x))
        if abs(angle) > MAX_ALIGN_ANGLE:
            angle = 0.0
    matrix = cv2.getRotationMatrix2D((centre_x, centre_y), angle, ALIGNED_FACE_SIZE / side)
    matrix[0, 2] += ALIGNED_FACE_SIZE / 2 - centre_x
    matrix[1, 2] += ALIGNED_FACE_SIZE / 2 - centre_y
    face = cv2.warpAffine(
        gray, matrix, (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    # DeepFace's own last step: bilinear resize to the 48x48 network input.
    face = cv2.resize(face, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    return face.astype(np.float32) / 255.0


def _yunet_weights() -> str:
    """Path to DeepFace's YuNet weights, downloading them on first use."""
    _utf8_console()
    from deepface.commons import weight_utils
    from deepface.models.face_detection.YuNet import WEIGHTS_URL

    return weight_utils.download_weights_if_necessary(
        file_name="face_detection_yunet_2023mar.onnx", source_url=WEIGHTS_URL
    )


class YuNetDetector:
    """OpenCV's YuNet CNN face detector (a DeepFace backend), used directly.

    Calling it directly lets us choose the detection resolution: small for
    live video, large for group photos. Instances are not thread-safe.
    """

    def __init__(self, score_threshold: float = YUNET_SCORE_THRESHOLD) -> None:
        self._size = (320, 320)
        self._model = cv2.FaceDetectorYN.create(
            _yunet_weights(), "", self._size, score_threshold, YUNET_NMS_THRESHOLD, 100
        )

    def detect(self, image_bgr: np.ndarray, max_side: int) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1:
            image_bgr = cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        size = (image_bgr.shape[1], image_bgr.shape[0])
        if size != self._size:
            self._model.setInputSize(size)
            self._size = size
        _, rows = self._model.detect(image_bgr)
        if rows is None:
            return []
        detections = []
        for row in rows:
            values = row[:8] / scale
            x, y, box_width, box_height = values[:4]
            left, top = max(0.0, x), max(0.0, y)
            right, bottom = min(width, x + box_width), min(height, y + box_height)
            if right - left < 2 or bottom - top < 2:
                continue
            # Columns 4-7 hold the right then left eye (from the person's view).
            eyes = ((float(values[4]), float(values[5])), (float(values[6]), float(values[7])))
            detections.append(
                Detection(
                    (round(left), round(top), round(right - left), round(bottom - top)), float(row[14]), eyes
                )
            )
        return detections


class HaarDetector:
    """OpenCV's bundled Haar cascade; the offline fallback for live video."""

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self._cascade.empty():
            raise RuntimeError("OpenCV's Haar cascade could not be loaded.")

    def detect(self, image_bgr: np.ndarray, max_side: int) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if scale < 1:
            gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
        return [Detection(tuple(round(value / scale) for value in box)) for box in boxes]


def live_face_detector() -> YuNetDetector | HaarDetector:
    """YuNet for live video, or the Haar cascade if its weights are unavailable."""
    try:
        return YuNetDetector(score_threshold=LIVE_SCORE_THRESHOLD)
    except Exception:
        return HaarDetector()


_yunet: YuNetDetector | None = None
_detector_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _detector_lock(name: str) -> threading.Lock:
    with _locks_guard:
        return _detector_locks.setdefault(name, threading.Lock())


def _deepface_detect(backend: str, image_bgr: np.ndarray) -> list[Detection]:
    _utf8_console()
    from deepface import DeepFace

    model = DeepFace.build_model(task="face_detector", model_name=backend)
    with _detector_lock(backend):
        regions = model.detect_faces(image_bgr)
    height, width = image_bgr.shape[:2]
    detections = []
    for region in regions:
        left, top = max(0, int(region.x)), max(0, int(region.y))
        right, bottom = min(width, int(region.x + region.w)), min(height, int(region.y + region.h))
        if right - left < 2 or bottom - top < 2:
            continue
        eyes = None
        if region.left_eye is not None and region.right_eye is not None:
            eyes = (tuple(map(float, region.right_eye)), tuple(map(float, region.left_eye)))
        confidence = None if region.confidence is None else float(region.confidence)
        detections.append(Detection((left, top, right - left, bottom - top), confidence, eyes))
    return detections


def detect_faces(image_rgb: np.ndarray, detector: str = "yunet") -> list[Detection]:
    """Find faces with a DeepFace detector backend (``skip`` finds none)."""
    if detector == "skip":
        return []
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if detector != "yunet":
        return _deepface_detect(detector, image_bgr)

    global _yunet
    with _detector_lock("yunet"):
        if _yunet is None:
            _yunet = YuNetDetector()
        detections = _yunet.detect(image_bgr, YUNET_MAX_SIDE)
        if detections:
            return detections
        # Tightly cropped portraits leave no margin around the face, which
        # YuNet needs; retry once with a replicated border.
        height, width = image_bgr.shape[:2]
        pad = max(height, width) // 4
        padded = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        detections = _yunet.detect(padded, YUNET_MAX_SIDE)
    shifted = []
    for detection in detections:
        x, y, box_width, box_height = detection.box
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(width, x - pad + box_width), min(height, y - pad + box_height)
        if right - left < 2 or bottom - top < 2:
            continue
        eyes = tuple((eye_x - pad, eye_y - pad) for eye_x, eye_y in detection.eyes) if detection.eyes else None
        shifted.append(Detection((left, top, right - left, bottom - top), detection.confidence, eyes))
    return shifted


def classify_faces(
    image_rgb: np.ndarray,
    detections: list[Detection],
    align: bool = True,
    tta: bool = True,
    detector: str | None = None,
    classifier: EmotionClassifier | None = None,
) -> list[FaceResult]:
    """Run DeepFace's emotion CNN on every detected face as one batch."""
    if not detections:
        return []
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    inputs = np.stack([face_to_model_input(gray, item.box, item.eyes, align) for item in detections])
    probabilities = (classifier or emotion_classifier()).predict(inputs, tta=tta) * 100
    faces = [
        FaceResult(
            box=item.box,
            dominant_emotion=EMOTIONS[int(np.argmax(scores))],
            scores={emotion: float(scores[index]) for index, emotion in enumerate(EMOTIONS)},
            face_confidence=item.confidence,
            eyes=item.eyes,
            detector=detector,
        )
        for item, scores in zip(detections, probabilities)
    ]
    return sorted(faces, key=lambda face: (face.box[0], face.box[1]))


def analyze(
    image_rgb: np.ndarray,
    detector: str = "yunet",
    align: bool = True,
    whole_image_fallback: bool = False,
    tta: bool = True,
    detector_fallback: bool = True,
) -> list[FaceResult]:
    """Detect and classify every face in a photo, returning them left to right.

    If the chosen detector finds nothing (or fails), the other detectors in
    ``DETECTOR_FALLBACK_ORDER`` are tried before giving up.
    """
    image_rgb = prepare_image(image_rgb)
    candidates = [detector]
    if detector_fallback and detector != "skip":
        candidates += [name for name in DETECTOR_FALLBACK_ORDER if name != detector]
    first_error: Exception | None = None
    any_detector_ran = False
    for name in candidates:
        try:
            detections = detect_faces(image_rgb, name)
        except Exception as error:
            first_error = first_error or error
            continue
        any_detector_ran = True
        if detections:
            return classify_faces(image_rgb, detections, align, tta, detector=name)
    if whole_image_fallback or detector == "skip":
        height, width = image_rgb.shape[:2]
        return classify_faces(image_rgb, [Detection((0, 0, width, height))], False, tta, detector="whole image")
    if first_error is not None and not any_detector_ran:
        raise first_error
    return []


def analyze_frame(
    frame_rgb: np.ndarray,
    face_detector: YuNetDetector | HaarDetector,
    classifier: EmotionClassifier,
    tta: bool = True,
) -> tuple[list[FaceResult], dict[str, float]]:
    """Live-video inference: detect on a small copy, classify full-resolution crops.

    Returns the faces and the time spent detecting and classifying (ms).
    """
    started = time.perf_counter()
    height, width = frame_rgb.shape[:2]
    scale = min(1.0, LIVE_DETECTION_WIDTH / width)
    small = cv2.resize(frame_rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    detections = [
        item
        for item in face_detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), LIVE_DETECTION_WIDTH)
        if min(item.box[2], item.box[3]) / scale >= LIVE_MIN_FACE_SIZE
    ]
    detections = [
        Detection(
            tuple(round(value / scale) for value in item.box),
            item.confidence,
            tuple((eye_x / scale, eye_y / scale) for eye_x, eye_y in item.eyes) if item.eyes else None,
        )
        for item in detections
    ]
    detected = time.perf_counter()
    faces = classify_faces(frame_rgb, detections, align=True, tta=tta, detector="live", classifier=classifier)
    finished = time.perf_counter()
    return faces, {"detect_ms": (detected - started) * 1000, "classify_ms": (finished - detected) * 1000}


def model_input_preview(image_rgb: np.ndarray, face: FaceResult, align: bool = True) -> np.ndarray:
    """The exact 48x48 grayscale image the CNN saw for a face (for display)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    use_eyes = align and face.detector != "whole image"
    return (face_to_model_input(gray, face.box, face.eyes, use_eyes) * 255).astype(np.uint8)


def overall_mood(faces: list[FaceResult]) -> tuple[str, dict[str, float]] | None:
    """Average individual emotion distributions for a group-photo summary."""
    if not faces:
        return None
    averages = {
        emotion: sum(face.scores.get(emotion, 0.0) for face in faces) / len(faces)
        for emotion in EMOTIONS
    }
    return max(averages, key=averages.get), averages
