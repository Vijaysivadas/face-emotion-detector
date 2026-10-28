from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from emotion_detector.config import EMOTION_STYLE, EMOTIONS
from emotion_detector.detector import FaceResult, face_to_model_input, overall_mood, prepare_image
from emotion_detector.live import BoxEaser, FaceTracker
from emotion_detector.visualize import annotate, crop_face, hex_to_rgb


def sample_face(box=(20, 20, 40, 40), emotion="happy") -> FaceResult:
    scores = {name: 1.0 for name in EMOTIONS}
    scores[emotion] = 94.0
    return FaceResult(box, emotion, scores)


def test_prepare_image_downscales_large_image() -> None:
    image = Image.fromarray(np.zeros((2000, 4000, 3), dtype=np.uint8))
    prepared = prepare_image(image)
    assert max(prepared.shape[:2]) == 1600


def test_annotation_keeps_shape_and_uses_rgb_colours() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = annotate(image, [sample_face(emotion="angry")])
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    # The left edge of the box is drawn in the emotion's colour, in RGB order.
    assert tuple(annotated[50, 20]) == hex_to_rgb(EMOTION_STYLE["angry"]["color"])


def test_crop_and_overall_mood() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_face(image, sample_face().box).size > 0
    mood = overall_mood([sample_face(), sample_face(emotion="sad")])
    assert mood is not None and mood[0] in {"happy", "sad"}


def test_ranked_orders_scores() -> None:
    ranked = sample_face().ranked()
    assert ranked[0][0] == "happy"
    assert ranked[0][1] >= ranked[1][1]


def test_model_input_is_scaled_48px_square() -> None:
    gray = np.full((120, 90), 200, dtype=np.uint8)
    face = face_to_model_input(gray, (10, 20, 60, 80))
    assert face.shape == (48, 48)
    assert face.dtype == np.float32
    # DeepFace's CNN expects [0, 1]; raw 0-255 input cost ~8 pp accuracy.
    assert 0.0 <= face.min() and face.max() <= 1.0


def _blob_rows(face: np.ndarray) -> list[float]:
    mask = (face > 0.5).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = sorted(range(1, count), key=lambda index: -stats[index, cv2.CC_STAT_AREA])[:2]
    return [centroids[index][1] for index in blobs]


def test_alignment_levels_the_eyes() -> None:
    gray = np.zeros((200, 200), dtype=np.uint8)
    left_eye, right_eye = (70.0, 90.0), (130.0, 110.0)  # head tilted ~18 degrees
    for x, y in (left_eye, right_eye):
        cv2.circle(gray, (int(x), int(y)), 8, 255, -1)
    box = (50, 50, 100, 100)
    tilted = _blob_rows(face_to_model_input(gray, box, (left_eye, right_eye), align=False))
    levelled = _blob_rows(face_to_model_input(gray, box, (right_eye, left_eye), align=True))
    assert abs(tilted[0] - tilted[1]) > 5
    assert abs(levelled[0] - levelled[1]) < 1.5


def test_tracker_keeps_identity_and_smooths_scores() -> None:
    tracker = FaceTracker(smoothing_seconds=0.5, timeout_seconds=0.5, min_hits=2)
    tracker.update([sample_face((100, 100, 80, 80), "happy")], now=0.0)
    assert tracker.visible() == []  # one sighting is not enough to draw
    tracker.update([sample_face((106, 102, 80, 80), "sad")], now=0.1)
    visible = tracker.visible()
    assert len(visible) == 1 and visible[0].track_id == 1
    # A single "sad" frame does not immediately overturn a "happy" face.
    assert visible[0].as_result().dominant_emotion == "happy"
    tracker.update([], now=1.0)
    assert tracker.tracks == []


def test_tracker_separates_distant_faces() -> None:
    tracker = FaceTracker(min_hits=1)
    tracker.update([sample_face((10, 10, 50, 50)), sample_face((300, 10, 50, 50))], now=0.0)
    tracker.update([sample_face((14, 10, 50, 50)), sample_face((296, 12, 50, 50))], now=0.1)
    assert sorted(track.track_id for track in tracker.visible()) == [1, 2]


def test_box_easer_moves_towards_target() -> None:
    tracker = FaceTracker(min_hits=1)
    easer = BoxEaser(follow=0.5)
    tracker.update([sample_face((0, 0, 40, 40))], now=0.0)
    easer.step(tracker.visible())
    tracker.tracks[0].box = np.array([20.0, 0.0, 40.0, 40.0])
    (_, face), = easer.step(tracker.visible())
    assert face.box[0] == 10


def test_local_hosts_skip_stun() -> None:
    from ui.live import _is_local

    assert _is_local("localhost") and _is_local("127.0.0.1") and _is_local("192.168.1.20")
    assert not _is_local("emotion-detection.streamlit.app")
    assert not _is_local(None)


def test_turn_relay_from_secrets(monkeypatch) -> None:
    import ui.live as live

    for name in ("CLOUDFLARE_TURN_KEY_ID", "CLOUDFLARE_TURN_KEY_API_TOKEN", "TURN_URLS", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert live._turn_servers() == ([], None)

    monkeypatch.setenv("TURN_URLS", "turn:relay.example:80, turns:relay.example:443")
    monkeypatch.setenv("TURN_USERNAME", "user")
    monkeypatch.setenv("TURN_CREDENTIAL", "secret")
    servers, problem = live._turn_servers()
    assert problem is None
    assert servers == [{"urls": ["turn:relay.example:80", "turns:relay.example:443"], "username": "user", "credential": "secret"}]


def test_failing_relay_is_reported(monkeypatch) -> None:
    import ui.live as live

    for name in ("CLOUDFLARE_TURN_KEY_ID", "CLOUDFLARE_TURN_KEY_API_TOKEN", "TURN_URLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def unavailable(token):
        raise ValueError("Failed to get credentials from HF turn server")

    monkeypatch.setattr(live, "get_hf_ice_servers", unavailable)
    servers, problem = live._turn_servers()
    assert servers == []
    assert "Hugging Face" in problem


def test_classifier_averages_with_mirror_image() -> None:
    tf = pytest.importorskip("tensorflow")
    from emotion_detector.detector import EmotionClassifier

    def stub(batch, training=False):
        # Class 0 for a bright left half, class 1 for a bright right half.
        left = tf.reduce_mean(batch[:, :, :24, 0], axis=[1, 2])
        right = tf.reduce_mean(batch[:, :, 24:, 0], axis=[1, 2])
        logits = tf.stack([left, right] + [tf.zeros_like(left)] * 5, axis=1) * 20
        return tf.nn.softmax(logits)

    classifier = EmotionClassifier(keras_model=stub)
    faces = np.zeros((3, 48, 48), np.float32)
    faces[:, :, :24] = 1.0
    single = classifier.predict(faces, tta=False)
    mirrored = classifier.predict(faces, tta=True)
    assert single.shape == mirrored.shape == (3, 7)
    np.testing.assert_allclose(mirrored.sum(axis=1), 1.0, rtol=1e-5)
    assert single[0, 0] > 0.9 and abs(mirrored[0, 0] - mirrored[0, 1]) < 1e-5
    assert classifier.predict(np.zeros((0, 48, 48), np.float32)).shape == (0, 7)


WEIGHTS = Path.home() / ".deepface" / "weights" / "facial_expression_model_weights.h5"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="DeepFace emotion weights not downloaded")
def test_end_to_end_on_sample_portrait() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from emotion_detector.detector import analyze

    portrait = Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    image = prepare_image(Image.open(portrait))
    faces = analyze(image)
    assert len(faces) == 1
    assert faces[0].detector == "yunet"
    assert abs(sum(faces[0].scores.values()) - 100) < 0.01
    blank = np.zeros((120, 120, 3), dtype=np.uint8)
    assert analyze(blank) == []
    assert analyze(blank, whole_image_fallback=True)[0].detector == "whole image"
