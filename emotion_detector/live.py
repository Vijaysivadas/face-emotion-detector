"""Real-time emotion tracking for the live camera view.

The WebRTC frame callback must return in a few milliseconds, so it only hands
the newest frame to :class:`LiveEmotionEngine` and draws the latest results.
A background thread analyses frames as fast as the rate cap allows, always
skipping to the newest one, and a tracker smooths boxes and emotion scores
over time so labels follow faces without flickering.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import (
    EMOTIONS,
    LIVE_DEFAULT_RATE_HZ,
    LIVE_DEFAULT_SMOOTHING_SECONDS,
    LIVE_HISTORY_SECONDS,
    LIVE_MIN_HITS,
    LIVE_TRACK_TIMEOUT_SECONDS,
)
from .detector import EmotionClassifier, FaceResult, analyze_frame, live_face_detector


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    scores: np.ndarray
    face_confidence: float | None
    last_seen: float
    hits: int = 1

    def as_result(self) -> FaceResult:
        x, y, width, height = (int(round(value)) for value in self.box)
        scores = {emotion: float(self.scores[index]) for index, emotion in enumerate(EMOTIONS)}
        return FaceResult((x, y, width, height), max(scores, key=scores.get), scores, self.face_confidence, None, "live")


class FaceTracker:
    """Associates faces between updates and smooths their boxes and scores.

    Emotion scores use a time-based exponential moving average, so the
    smoothing feels the same whatever the inference rate is.
    """

    def __init__(
        self,
        smoothing_seconds: float = LIVE_DEFAULT_SMOOTHING_SECONDS,
        timeout_seconds: float = LIVE_TRACK_TIMEOUT_SECONDS,
        min_hits: int = LIVE_MIN_HITS,
    ) -> None:
        self.smoothing_seconds = smoothing_seconds
        self.timeout_seconds = timeout_seconds
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _distance(track: Track, face: FaceResult) -> float:
        """Centre distance relative to face size; inf when sizes disagree."""
        x, y, width, height = face.box
        tx, ty, tw, th = track.box
        size_ratio = max(width, height) / max(tw, th, 1)
        if not 0.5 <= size_ratio <= 2.0:
            return math.inf
        dx = (x + width / 2) - (tx + tw / 2)
        dy = (y + height / 2) - (ty + th / 2)
        return math.hypot(dx, dy) / max(tw, th, 1)

    def update(self, faces: list[FaceResult], now: float) -> None:
        pairs = sorted(
            (self._distance(track, face), track_index, face_index)
            for track_index, track in enumerate(self.tracks)
            for face_index, face in enumerate(faces)
        )
        matched_tracks: set[int] = set()
        matched_faces: set[int] = set()
        for distance, track_index, face_index in pairs:
            if distance > 0.75 or track_index in matched_tracks or face_index in matched_faces:
                continue
            matched_tracks.add(track_index)
            matched_faces.add(face_index)
            track, face = self.tracks[track_index], faces[face_index]
            elapsed = max(now - track.last_seen, 1e-3)
            keep = math.exp(-elapsed / self.smoothing_seconds) if self.smoothing_seconds > 0 else 0.0
            measured = np.array([face.scores[emotion] for emotion in EMOTIONS])
            track.scores = keep * track.scores + (1 - keep) * measured
            # Boxes follow quickly; the video overlay eases between updates.
            track.box = 0.3 * track.box + 0.7 * np.asarray(face.box, dtype=float)
            track.face_confidence = face.face_confidence
            track.last_seen = now
            track.hits += 1
        for face_index, face in enumerate(faces):
            if face_index not in matched_faces:
                self.tracks.append(
                    Track(
                        self._next_id,
                        np.asarray(face.box, dtype=float),
                        np.array([face.scores[emotion] for emotion in EMOTIONS]),
                        face.face_confidence,
                        now,
                    )
                )
                self._next_id += 1
        self.tracks = [track for track in self.tracks if now - track.last_seen <= self.timeout_seconds]

    def visible(self) -> list[Track]:
        return [track for track in self.tracks if track.hits >= self.min_hits]


class BoxEaser:
    """Eases drawn boxes toward the latest tracked boxes on every video frame."""

    def __init__(self, follow: float = 0.5) -> None:
        self.follow = follow
        self._shown: dict[int, np.ndarray] = {}

    def step(self, tracks: list[Track]) -> list[tuple[int, FaceResult]]:
        shown: dict[int, np.ndarray] = {}
        results = []
        for track in tracks:
            previous = self._shown.get(track.track_id)
            box = track.box.copy() if previous is None else previous + (track.box - previous) * self.follow
            shown[track.track_id] = box
            result = track.as_result()
            eased_box = tuple(int(round(value)) for value in box)
            results.append(
                (track.track_id, FaceResult(eased_box, result.dominant_emotion, result.scores, result.face_confidence))
            )
        self._shown = shown
        return results


@dataclass
class LiveSettings:
    tta: bool = True
    rate_hz: float = LIVE_DEFAULT_RATE_HZ
    smoothing_seconds: float = LIVE_DEFAULT_SMOOTHING_SECONDS


@dataclass
class LiveSnapshot:
    """A thread-safe copy of the engine's state for the dashboard."""

    faces: list[tuple[int, FaceResult]]
    history: list[tuple[float, np.ndarray | None]]
    analysis_hz: float
    detect_ms: float
    classify_ms: float
    video_fps: float
    elapsed: float
    analysed_frames: int
    detector_name: str
    error: str | None = None
    time_share: dict[str, float] = field(default_factory=dict)


class LiveEmotionEngine:
    """Background analysis of the newest camera frame, one frame at a time."""

    def __init__(self, classifier: EmotionClassifier, settings: LiveSettings | None = None) -> None:
        self.classifier = classifier
        self.settings = settings or LiveSettings()
        self.tracker = FaceTracker(self.settings.smoothing_seconds)
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stopped = threading.Event()
        self._frame: np.ndarray | None = None
        self.latest_frame: np.ndarray | None = None
        self.started = time.monotonic()
        self._history: deque[tuple[float, np.ndarray | None]] = deque(maxlen=LIVE_HISTORY_SECONDS * 30)
        self._time_share = np.zeros(len(EMOTIONS))
        self._timings: deque[tuple[float, float]] = deque(maxlen=30)
        self._update_times: deque[float] = deque(maxlen=30)
        self._frame_times: deque[float] = deque(maxlen=60)
        self._analysed = 0
        self._last_update = time.monotonic()
        self._error: str | None = None
        self._detector_name = "loading"
        self._thread = threading.Thread(target=self._run, name="emotion-live-analysis", daemon=True)
        self._thread.start()

    def submit(self, frame_rgb: np.ndarray) -> None:
        """Offer the newest frame; an unanalysed older frame is simply replaced."""
        now = time.monotonic()
        with self._lock:
            self._frame = frame_rgb
            self.latest_frame = frame_rgb
            self._frame_times.append(now)
        self._frame_ready.set()

    def _run(self) -> None:
        try:
            face_detector = live_face_detector()
        except Exception as error:  # pragma: no cover - only without OpenCV data files
            self._error = f"Face detector unavailable: {error}"
            return
        self._detector_name = "YuNet" if type(face_detector).__name__ == "YuNetDetector" else "Haar cascade"
        next_start = 0.0
        while not self._stopped.is_set():
            if not self._frame_ready.wait(timeout=0.5):
                continue
            delay = next_start - time.monotonic()
            if delay > 0 and self._stopped.wait(delay):
                break
            with self._lock:
                frame, self._frame = self._frame, None
                self._frame_ready.clear()
            if frame is None:
                continue
            started = time.monotonic()
            try:
                faces, timing = analyze_frame(frame, face_detector, self.classifier, tta=self.settings.tta)
            except Exception as error:  # keep the stream alive after a bad frame
                self._error = f"{type(error).__name__}: {error}"
                continue
            now = time.monotonic()
            with self._lock:
                self.tracker.smoothing_seconds = self.settings.smoothing_seconds
                self.tracker.update(faces, now)
                self._error = None
                self._analysed += 1
                self._timings.append((timing["detect_ms"], timing["classify_ms"]))
                self._update_times.append(now)
                visible = self.tracker.visible()
                primary = max(visible, key=lambda track: track.box[2] * track.box[3], default=None)
                self._history.append((now - self.started, None if primary is None else primary.scores.copy()))
                if primary is not None:
                    # Cap the step so a stall (or the first update) is not over-counted.
                    self._time_share[int(np.argmax(primary.scores))] += min(now - self._last_update, 0.5)
                self._last_update = now
            next_start = started + 1.0 / max(self.settings.rate_hz, 1.0)

    def visible_tracks(self) -> list[Track]:
        with self._lock:
            return [
                Track(track.track_id, track.box.copy(), track.scores.copy(), track.face_confidence, track.last_seen, track.hits)
                for track in self.tracker.visible()
            ]

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2 or times[-1] == times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def rates(self) -> tuple[float, float]:
        """(video frames per second, emotion updates per second) — cheap, for the HUD."""
        with self._lock:
            return self._rate(self._frame_times), self._rate(self._update_times)

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            tracks = sorted(self.tracker.visible(), key=lambda track: -track.box[2] * track.box[3])
            timings = np.array(self._timings) if self._timings else np.zeros((1, 2))
            share_total = self._time_share.sum()
            return LiveSnapshot(
                faces=[(track.track_id, track.as_result()) for track in tracks],
                history=list(self._history),
                analysis_hz=self._rate(self._update_times),
                detect_ms=float(timings[:, 0].mean()),
                classify_ms=float(timings[:, 1].mean()),
                video_fps=self._rate(self._frame_times),
                elapsed=time.monotonic() - self.started,
                analysed_frames=self._analysed,
                detector_name=self._detector_name,
                error=self._error,
                time_share={
                    emotion: float(self._time_share[index] / share_total) if share_total else 0.0
                    for index, emotion in enumerate(EMOTIONS)
                },
            )

    def stop(self) -> None:
        self._stopped.set()
        self._frame_ready.set()
