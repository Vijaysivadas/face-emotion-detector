"""Shared UI helpers: styling, cached models, settings and small components."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from emotion_detector.config import (
    DEFAULT_DETECTOR,
    DISPLAY_ORDER,
    EMOTION_STYLE,
    EMOTIONS,
    LIVE_DEFAULT_RATE_HZ,
    LIVE_DEFAULT_RESOLUTION,
    LIVE_DEFAULT_SMOOTHING_SECONDS,
    PROJECT,
    TEAM_MEMBERS,
)
from emotion_detector.detector import EmotionClassifier, FaceResult, emotion_classifier

REPORTS = Path(__file__).resolve().parents[1] / "reports"

PHOTO_DEFAULTS = {
    "opt_detector": DEFAULT_DETECTOR,
    "opt_align": True,
    "opt_tta": True,
    "opt_detector_fallback": True,
    "opt_whole_image": False,
    "opt_gradcam": True,
}
LIVE_DEFAULTS = {
    "live_resolution": LIVE_DEFAULT_RESOLUTION,
    "live_rate": LIVE_DEFAULT_RATE_HZ,
    "live_smoothing": LIVE_DEFAULT_SMOOTHING_SECONDS,
    "live_tta": True,
    "live_hud": True,
}

CSS = """
<style>
.block-container { padding-bottom: 3rem; max-width: 1320px; }
.hero { margin: 0.4rem 0 1.4rem; }
.hero .eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; color: #6E6CF0; }
.hero h1 { font-size: clamp(1.7rem, 3.2vw, 2.35rem); line-height: 1.12; margin: 0.35rem 0 0.4rem; padding: 0; }
.hero p { opacity: 0.78; max-width: 74ch; margin: 0; font-size: 1.02rem; }
.chip { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.18rem 0.65rem; margin: 0 0.3rem 0.3rem 0;
  border-radius: 999px; font-size: 0.86rem; font-weight: 600; white-space: nowrap;
  background: color-mix(in srgb, var(--c) 15%, transparent); border: 1px solid color-mix(in srgb, var(--c) 50%, transparent); }
.emotion-hero { display: flex; align-items: center; gap: 0.95rem; padding: 0.85rem 1.05rem; border-radius: 14px; margin-bottom: 0.6rem;
  background: color-mix(in srgb, var(--c) 12%, transparent); border: 1px solid color-mix(in srgb, var(--c) 42%, transparent); }
.emotion-hero .emoji { font-size: 2.5rem; line-height: 1; }
.emotion-hero .name { font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; line-height: 1.15; }
.emotion-hero .sub { opacity: 0.78; font-size: 0.9rem; }
.panel { border: 1px dashed rgba(128, 128, 145, 0.5); border-radius: 14px; padding: 1.15rem 1.25rem; }
.panel h4 { margin: 0 0 0.6rem; padding: 0; }
.step { display: flex; gap: 0.65rem; align-items: flex-start; margin: 0.5rem 0; }
.step b { flex: none; display: inline-flex; width: 1.55rem; height: 1.55rem; border-radius: 50%; align-items: center;
  justify-content: center; background: rgba(110, 108, 240, 0.16); color: #6E6CF0; font-size: 0.8rem; }
.note { font-size: 0.88rem; opacity: 0.8; }
.status-dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; margin-right: 0.4rem; vertical-align: middle; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def keep_settings() -> None:
    """Initialise settings and keep widget values alive across pages.

    Streamlit drops a widget's state when its page is not shown; writing the
    value back each run keeps the user's choices when they switch pages.
    """
    for key, default in {**PHOTO_DEFAULTS, **LIVE_DEFAULTS}.items():
        st.session_state[key] = st.session_state.get(key, default)


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def emotion_color(emotion: str) -> str:
    return EMOTION_STYLE[emotion]["dark" if is_dark() else "color"]


def chip(emotion: str, text: str) -> str:
    style = EMOTION_STYLE[emotion]
    return f'<span class="chip" style="--c:{emotion_color(emotion)}">{style["emoji"]} {html.escape(text)}</span>'


def emotion_hero(face: FaceResult, caption: str | None = None) -> None:
    """A large, colour-coded headline for the leading emotion."""
    style = EMOTION_STYLE[face.dominant_emotion]
    runner_up, runner_score = face.ranked()[1]
    sub = caption or f"{face.confidence:.0f}% confidence · next: {runner_up} {runner_score:.0f}%"
    st.markdown(
        f'<div class="emotion-hero" style="--c:{emotion_color(face.dominant_emotion)}">'
        f'<div class="emoji">{style["emoji"]}</div><div><div class="name">{face.dominant_emotion.title()}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_classifier() -> EmotionClassifier:
    """DeepFace's emotion CNN, loaded and compiled once per server process."""
    return emotion_classifier()


@st.cache_resource(show_spinner=False)
def warm_live_detector() -> bool:
    """Download YuNet's weights before streaming so the first frames aren't delayed."""
    from emotion_detector.detector import live_face_detector

    live_face_detector()
    return True


def ensure_model() -> bool:
    """Load the model with a friendly spinner; show the error if it fails."""
    try:
        with st.spinner("Loading DeepFace's emotion model (the first start may download weights)…"):
            get_classifier()
        return True
    except Exception as error:
        st.error("The emotion model could not be loaded. Check the internet connection for the first weight download, then reload.")
        with st.expander("Technical details"):
            st.exception(error)
        return False


def result_rows(faces: list[FaceResult], source: str | None = None) -> pd.DataFrame:
    rows = []
    for number, face in enumerate(faces, start=1):
        row: dict[str, float | int | str] = {}
        if source is not None:
            row["image"] = source
        row.update(
            {
                "face": number,
                "emotion": face.dominant_emotion,
                "confidence_percent": round(face.confidence, 2),
                "runner_up": face.ranked()[1][0],
                "detector": face.detector or "",
                "detection_confidence": "" if face.face_confidence is None else round(face.face_confidence, 4),
                "x": face.box[0],
                "y": face.box[1],
                "width": face.box[2],
                "height": face.box[3],
            }
        )
        row.update({f"{emotion}_percent": round(face.scores[emotion], 2) for emotion in EMOTIONS})
        rows.append(row)
    return pd.DataFrame(rows)


def png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict | None:
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        st.markdown(f"**{PROJECT['group']}** · {PROJECT['course']}")
        if TEAM_MEMBERS:
            st.markdown("\n".join(f"- {member}" for member in TEAM_MEMBERS))
        st.caption("Privacy: images and video are processed in memory and never stored.")


def legend_html() -> str:
    """Emotion colour key, in the chart order."""
    return "".join(chip(emotion, emotion.title()) for emotion in DISPLAY_ORDER)
