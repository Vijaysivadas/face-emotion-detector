"""Streamlit interface for the Group 5 facial emotion detection project."""

from __future__ import annotations

import io
import os
from typing import Callable

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from emotion_detector.config import (
    DETECTORS,
    EMOTIONS,
    EMOTION_STYLE,
    LIVE_ANALYSIS_WIDTH,
    LIVE_ANALYZE_EVERY_N_FRAMES,
    PROJECT,
    TEAM_MEMBERS,
)
from emotion_detector.detector import FaceResult, analyze, load_model, overall_mood, prepare_image
from emotion_detector.visualize import annotate, crop_face


st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_model() -> object:
    return load_model()


def ensure_model() -> None:
    """Warm the CNN once, making the first inference failure understandable."""
    with st.spinner("Loading DeepFace emotion model (first use may download weights)…"):
        cached_model()


def result_rows(faces: list[FaceResult]) -> pd.DataFrame:
    rows = []
    for number, face in enumerate(faces, start=1):
        row: dict[str, float | int | str] = {
            "face": number,
            "emotion": face.dominant_emotion,
            "emotion_confidence_percent": round(face.confidence, 2),
            "face_detection_confidence": "" if face.face_confidence is None else round(face.face_confidence, 4),
            "x": face.box[0],
            "y": face.box[1],
            "width": face.box[2],
            "height": face.box[3],
        }
        row.update({f"{emotion}_percent": round(face.scores[emotion], 2) for emotion in EMOTIONS})
        rows.append(row)
    return pd.DataFrame(rows)


def png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def emotion_chart(face: FaceResult) -> alt.Chart:
    chart_data = pd.DataFrame(
        {
            "Emotion": [emotion.title() for emotion in EMOTIONS],
            "Confidence": [face.scores[emotion] for emotion in EMOTIONS],
            "Colour": [EMOTION_STYLE[emotion]["color"] for emotion in EMOTIONS],
        }
    )
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Confidence:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Emotion:N", sort="-x", title=None),
            color=alt.Color("Colour:N", scale=None, legend=None),
            tooltip=["Emotion:N", alt.Tooltip("Confidence:Q", format=".1f")],
        )
        .properties(height=185)
    )


def render_results(image_rgb: np.ndarray, faces: list[FaceResult], key: str, source_name: str = "image") -> None:
    if not faces:
        st.warning("No face was detected. Try a clearer, front-facing image, select another detector, or enable whole-image fallback.")
        st.image(image_rgb, caption="Original image", use_container_width=True)
        return

    annotated = annotate(image_rgb, faces)
    mood = overall_mood(faces)
    image_col, details_col = st.columns((1.25, 1), gap="large")
    with image_col:
        st.image(annotated, caption="Detected faces and predicted expressions", use_container_width=True)
        st.download_button(
            "Download annotated PNG", png_bytes(annotated), f"{source_name}_annotated.png", "image/png", key=f"png-{key}"
        )
    with details_col:
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Faces found", len(faces))
        if mood:
            mood_name, mood_scores = mood
            metric_right.metric("Overall mood", f"{EMOTION_STYLE[mood_name]['emoji']} {mood_name.title()}", f"{mood_scores[mood_name]:.0f}%")
        st.caption("Overall mood is the average of all detected facial-expression scores.")
        st.download_button(
            "Download results CSV",
            result_rows(faces).to_csv(index=False).encode("utf-8"),
            f"{source_name}_results.csv",
            "text/csv",
            key=f"csv-{key}",
        )

    st.markdown("#### Face-by-face analysis")
    for number, face in enumerate(faces, start=1):
        with st.container(border=True):
            preview_col, chart_col = st.columns((0.32, 0.68), gap="medium")
            with preview_col:
                thumbnail = crop_face(image_rgb, face.box)
                st.image(thumbnail, caption=f"Face #{number}", use_container_width=True)
                style = EMOTION_STYLE[face.dominant_emotion]
                st.markdown(f"### {style['emoji']} {face.dominant_emotion.title()}")
                st.metric("Model confidence", f"{face.confidence:.1f}%")
            with chart_col:
                st.altair_chart(emotion_chart(face), use_container_width=True)


def process_image(image: Image.Image, source_name: str, key: str, settings: dict[str, object]) -> None:
    try:
        image_rgb = prepare_image(image)
    except (ValueError, UnidentifiedImageError, OSError) as error:
        st.error(f"Could not read {source_name}: {error}")
        return
    try:
        ensure_model()
        faces = analyze(image_rgb, **settings)
    except Exception as error:
        st.error("Analysis could not be completed. Check that the selected detector's weights can be downloaded, then try again.")
        with st.expander("Technical details"):
            st.exception(error)
        return
    render_results(image_rgb, faces, key, source_name)


def render_live_video(settings: dict[str, object]) -> None:
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    except ImportError:
        st.info("Live video needs `streamlit-webrtc` and `av`. Install the project requirements to enable it.")
        return

    st.success(
        f"Optimised live mode: OpenCV fast detection, {LIVE_ANALYSIS_WIDTH}px analysis frames, and one inference every "
        f"{LIVE_ANALYZE_EVERY_N_FRAMES} frames. Labels stay smooth between inferences."
    )
    try:
        ensure_model()
    except Exception as error:
        st.error("The emotion model could not be loaded. Complete its first-run weight download, then restart the app.")
        with st.expander("Technical details"):
            st.exception(error)
        return

    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.frame_number = 0
            self.faces: list[FaceResult] = []

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            rgb = frame.to_ndarray(format="rgb24")
            self.frame_number += 1
            height, width = rgb.shape[:2]
            analysis_width = min(LIVE_ANALYSIS_WIDTH, width)
            scale = analysis_width / width
            analysis_frame = (
                cv2.resize(rgb, (analysis_width, round(height * scale)), interpolation=cv2.INTER_AREA)
                if scale < 1
                else rgb
            )
            if self.frame_number % LIVE_ANALYZE_EVERY_N_FRAMES == 1:
                try:
                    detected_faces = analyze(analysis_frame, **settings)
                    if scale < 1:
                        inverse_scale = 1 / scale
                        self.faces = [
                            FaceResult(
                                box=tuple(round(value * inverse_scale) for value in face.box),
                                dominant_emotion=face.dominant_emotion,
                                scores=face.scores,
                                face_confidence=face.face_confidence,
                            )
                            for face in detected_faces
                        ]
                    else:
                        self.faces = detected_faces
                except Exception:
                    # Keep the most recently successful labels visible rather
                    # than interrupting the video stream on a transient frame.
                    pass
                    self.faces = []
            return av.VideoFrame.from_ndarray(annotate(rgb, self.faces), format="rgb24")

    webrtc_streamer(
        key="emotion-live-video",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )


def render_about() -> None:
    st.subheader("About the project")
    st.markdown(
        """**Problem statement.** This application detects each face in a photograph and classifies its visible
        facial expression as angry, disgust, fear, happy, sad, surprise, or neutral. It supports awareness and
        human–computer interaction research; it does not determine a person's true inner feelings."""
    )
    st.graphviz_chart(
        "digraph { rankdir=LR; node [shape=box style=rounded]; Photo -> Detection -> Alignment -> "
        "Preprocessing -> {Emotion_CNN [label=\"48×48 emotion CNN\"]} -> Softmax -> Results }"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Dataset")
        st.dataframe(
            pd.DataFrame(
                [["FER-2013", "35,887", "48 × 48 grayscale", "7"], ["FER+", "FER-2013 relabelled", "10 annotators/image", "8 incl. contempt"]],
                columns=["Dataset", "Images", "Format", "Classes"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("DeepFace's built-in model uses FER-2013 labels. FER+ is an improved re-labelling used in the literature survey.")
    with right:
        st.markdown("#### Model")
        st.markdown("DeepFace runs a pre-trained 7-class CNN on a 48 × 48 grayscale face crop. Its softmax output is shown as percentages; the largest score is the displayed expression.")
        st.markdown("Detection → alignment → crop/pre-process → CNN → softmax → visual result")

    st.markdown("#### Academic mapping")
    st.dataframe(
        pd.DataFrame([[PROJECT["course"], PROJECT["co"], PROJECT["pos"], PROJECT["psos"], PROJECT["sdg"]]], columns=["Course", "CO", "POs", "PSOs", "SDG"]),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("#### Ethics and limitations")
    st.warning("A facial expression is not a diagnosis or proof of emotion. FER-2013 is low-resolution and imbalanced, and results can vary across people, lighting, pose, and culture. Use only with consent. Images are processed in memory and are not stored by this app.")
    st.markdown("#### References")
    st.markdown(
        "1. Goodfellow et al. (2013), *Challenges in Representation Learning* (FER-2013).  \n"
        "2. Barsoum et al. (2016), *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution* (FER+).  \n"
        "3. Serengil & Ozpinar (2021), *HyperExtended LightFace*.  \n"
        "4. Li & Deng (2022), *Deep Facial Expression Recognition: A Survey*.  \n"
        "5. Ekman & Friesen (1971), *Constants across cultures in the face and emotion*."
    )


def main() -> None:
    st.title("🎭 Emotion Detection from Facial Images")
    st.caption("PCCSL 508 · DeepFace pre-trained CNN · FER-2013 · SDG 3")

    with st.sidebar:
        st.header("Analysis settings")
        detector = st.selectbox("Face detector", options=list(DETECTORS), format_func=DETECTORS.get)
        align = st.checkbox("Align faces", value=True, help="Rotate detected faces using landmarks when the detector supports it.")
        fallback = st.checkbox("Treat whole image as a face if none found", help="Useful for already-cropped face images.")
        st.divider()
        st.markdown(f"**{PROJECT['group']}**")
        st.caption("7-class facial-expression model · 48 × 48 grayscale input")
        if TEAM_MEMBERS:
            st.markdown("**Team members**\n\n" + "\n\n".join(f"- {member}" for member in TEAM_MEMBERS))
        st.caption("Privacy: images are processed in memory and are not saved by the app.")

    settings: dict[str, object] = {"detector": detector, "align": align, "whole_image_fallback": fallback}
    upload_tab, camera_tab, live_tab, about_tab = st.tabs(["📁 Upload images", "📸 Webcam snapshot", "🎥 Live video", "ℹ️ About"])
    with upload_tab:
        uploads = st.file_uploader("Choose one or more facial images", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True)
        if uploads:
            for index, upload in enumerate(uploads):
                st.markdown(f"### {upload.name}")
                try:
                    image = Image.open(upload)
                    process_image(image, upload.name.rsplit(".", 1)[0], f"upload-{index}", settings)
                except (UnidentifiedImageError, OSError) as error:
                    st.error(f"Could not open {upload.name}: {error}")
    with camera_tab:
        snapshot = st.camera_input("Take a photo")
        if snapshot:
            process_image(Image.open(snapshot), "webcam_snapshot", "camera", settings)
    with live_tab:
        # A live stream prioritises stable frame rate. Upload/snapshot retain
        # the detector and alignment choices selected in the sidebar.
        render_live_video({"detector": "opencv", "align": False, "whole_image_fallback": False})
    with about_tab:
        render_about()


if __name__ == "__main__":
    main()
