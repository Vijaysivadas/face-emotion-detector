"""Upload and snapshot pages, plus the detailed results view they share."""

from __future__ import annotations

import hashlib
import io
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from emotion_detector import charts
from emotion_detector.config import DETECTORS, EMOTION_STYLE, EMOTIONS, MODEL_INPUT_SIZE
from emotion_detector.detector import FaceResult, analyze, model_input_preview, overall_mood, prepare_image
from emotion_detector.explain import cam_overlay, grad_cam
from emotion_detector.visualize import annotate, crop_face

from .common import (
    chip,
    emotion_hero,
    ensure_model,
    get_classifier,
    is_dark,
    page_header,
    png_bytes,
    result_rows,
    sidebar_footer,
)

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


def photo_settings() -> dict[str, object]:
    """Sidebar controls shared by the upload and snapshot pages."""
    with st.sidebar:
        st.subheader("Photo analysis")
        st.selectbox("Face detector", options=list(DETECTORS), format_func=DETECTORS.get, key="opt_detector")
        st.toggle("Align faces by the eyes", key="opt_align", help="Rotates each face so the eyes are level before classification.")
        st.toggle(
            "Mirror test-time augmentation",
            key="opt_tta",
            help="Classifies each face and its mirror image and averages the two (+0.7 pp accuracy on FER-2013).",
        )
        st.toggle(
            "Try other detectors if none found",
            key="opt_detector_fallback",
            help="Falls back to YuNet → MTCNN → OpenCV when the chosen detector finds no face.",
        )
        st.toggle("Treat whole image as a face if none found", key="opt_whole_image", help="For photos that are already cropped to one face.")
        st.toggle("Show model attention (Grad-CAM)", key="opt_gradcam", help="Highlights the regions that pushed the model towards its prediction.")
    sidebar_footer()
    return {
        "detector": st.session_state.opt_detector,
        "align": st.session_state.opt_align,
        "tta": st.session_state.opt_tta,
        "detector_fallback": st.session_state.opt_detector_fallback,
        "whole_image_fallback": st.session_state.opt_whole_image,
    }


@st.cache_data(show_spinner=False, max_entries=64)
def _analyse(digest: str, _image_rgb: np.ndarray, settings: tuple[tuple[str, object], ...]) -> list[FaceResult]:
    """Cached by image digest and settings, so reruns never recompute."""
    get_classifier()
    return analyze(_image_rgb, **dict(settings))


def analyse_image(image_rgb: np.ndarray, settings: dict[str, object]) -> list[FaceResult]:
    digest = hashlib.sha1(np.ascontiguousarray(image_rgb).tobytes()).hexdigest()
    return _analyse(digest, image_rgb, tuple(sorted(settings.items())))


@st.cache_data(show_spinner=False, max_entries=512)
def _attention(model_input: np.ndarray, class_index: int) -> np.ndarray:
    cam = grad_cam(get_classifier().model, model_input.astype(np.float32) / 255.0, class_index)
    return cam_overlay(model_input, cam)


def _enlarge(image: np.ndarray, size: int = 160) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


def face_card(image_rgb: np.ndarray, face: FaceResult, number: int, align: bool, show_attention: bool) -> None:
    with st.container(border=True):
        model_input = model_input_preview(image_rgb, face, align)
        columns = st.columns(3 if show_attention else 2, gap="small")
        columns[0].image(crop_face(image_rgb, face.box), caption=f"Face #{number}", width="stretch")
        columns[1].image(_enlarge(model_input), caption=f"CNN input · {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}", width="stretch")
        if show_attention:
            attention = _attention(model_input, EMOTIONS.index(face.dominant_emotion))
            columns[2].image(attention, caption="Attention (Grad-CAM)", width="stretch")
        emotion_hero(face)
        notes = []
        if face.confidence < 45:
            second = face.ranked()[1][0]
            notes.append(f"Low confidence: the expression sits between {face.dominant_emotion} and {second}.")
        if min(face.box[2], face.box[3]) < MODEL_INPUT_SIZE:
            notes.append("Small face (under 48 px), so detail is limited.")
        if notes:
            st.caption(" ".join(notes))
        st.altair_chart(charts.emotion_bars(face.scores, is_dark(), height=200), width="stretch")


def render_results(
    image_rgb: np.ndarray, faces: list[FaceResult], key: str, source_name: str, settings: dict[str, object]
) -> None:
    """Annotated image, summary, downloads and one card per face."""
    if not faces:
        st.warning(
            "No face was found. Try a clearer, front-facing photo, choose **RetinaFace** in the sidebar, "
            "or turn on **Treat whole image as a face** for pre-cropped faces."
        )
        st.image(image_rgb, caption="Original image", width="stretch")
        return

    dark = is_dark()
    annotated = annotate(image_rgb, faces)
    mood_name, mood_scores = overall_mood(faces)
    image_column, summary_column = st.columns((1.35, 1), gap="large")
    with image_column:
        st.image(annotated, caption="Detected faces and predicted expressions", width="stretch")
        left, right = st.columns(2)
        left.download_button(
            "Annotated PNG", png_bytes(annotated), f"{source_name}_annotated.png", "image/png",
            key=f"png-{key}", icon=":material/download:", width="stretch", on_click="ignore",
        )
        right.download_button(
            "Results CSV", result_rows(faces).to_csv(index=False).encode("utf-8"), f"{source_name}_results.csv", "text/csv",
            key=f"csv-{key}", icon=":material/table:", width="stretch", on_click="ignore",
        )
    with summary_column:
        first, second = st.columns(2)
        first.metric("Faces found", len(faces), border=True)
        style = EMOTION_STYLE[mood_name]
        second.metric(
            "Overall mood" if len(faces) > 1 else "Expression",
            f"{style['emoji']} {mood_name.title()}",
            f"{mood_scores[mood_name]:.0f}% {'average' if len(faces) > 1 else 'confidence'}",
            delta_color="off", delta_arrow="off",
            border=True,
        )
        used = faces[0].detector
        if used and used != settings["detector"]:
            st.caption(f"Found by the fallback: **{used}** (the selected detector found no face).")
        st.markdown("**Average emotion mix**" if len(faces) > 1 else "**Emotion probabilities**")
        st.altair_chart(charts.emotion_bars(mood_scores, dark), width="stretch")
        if len(faces) > 1:
            st.markdown("".join(chip(face.dominant_emotion, f"#{n} {face.dominant_emotion.title()}") for n, face in enumerate(faces, 1)), unsafe_allow_html=True)

    st.markdown("#### Face-by-face analysis")
    columns = st.columns(2, gap="medium") if len(faces) > 1 else [st.container()]
    for number, face in enumerate(faces, start=1):
        with columns[(number - 1) % len(columns)]:
            face_card(image_rgb, face, number, bool(settings["align"]), st.session_state.get("opt_gradcam", True))


def _empty_state() -> None:
    st.markdown(
        '<div class="panel"><h4>Tips for reliable results</h4>'
        '<div class="step"><b>1</b><span>Use clear, evenly lit photos where faces are at least ~60 px wide.</span></div>'
        '<div class="step"><b>2</b><span>Group photos work: every face gets its own prediction and the overall mood is averaged.</span></div>'
        '<div class="step"><b>3</b><span>For tilted or partly turned faces, try <b style="all:unset;font-weight:600">RetinaFace</b> in the sidebar.</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def upload_page() -> None:
    page_header("Photos", "Analyse photos", "Upload one or more photos. Every face is detected, aligned and classified into seven expressions.")
    settings = photo_settings()
    uploads = st.file_uploader(
        "Drop face photos here", type=IMAGE_TYPES, accept_multiple_files=True,
        help="JPG, PNG, WEBP or BMP. Large photos are resized to 1600 px on the long side.",
    )
    if not uploads:
        _empty_state()
        return
    if not ensure_model():
        return

    analysed: list[tuple[str, np.ndarray, list[FaceResult]]] = []
    progress = st.progress(0.0, text="Analysing…") if len(uploads) > 1 else None
    for index, upload in enumerate(uploads):
        name = upload.name.rsplit(".", 1)[0]
        try:
            image_rgb = prepare_image(Image.open(io.BytesIO(upload.getvalue())))
            faces = analyse_image(image_rgb, settings)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Could not read {upload.name}: {error}")
            continue
        except Exception as error:
            st.error(f"Analysis of {upload.name} failed. The selected detector's weights may need to download first.")
            with st.expander("Technical details"):
                st.exception(error)
            continue
        analysed.append((name, image_rgb, faces))
        if progress is not None:
            progress.progress((index + 1) / len(uploads), text=f"Analysed {index + 1} of {len(uploads)}")
    if progress is not None:
        progress.empty()
    if not analysed:
        return

    if len(analysed) > 1:
        _batch_summary(analysed)
        tabs = st.tabs([name if len(name) <= 24 else name[:22] + "…" for name, _, _ in analysed])
        for index, (tab, (name, image_rgb, faces)) in enumerate(zip(tabs, analysed)):
            with tab:
                render_results(image_rgb, faces, f"upload-{index}", name, settings)
    else:
        name, image_rgb, faces = analysed[0]
        render_results(image_rgb, faces, "upload-0", name, settings)


def _batch_summary(analysed: list[tuple[str, np.ndarray, list[FaceResult]]]) -> None:
    all_faces = [face for _, _, faces in analysed for face in faces]
    counts = Counter(face.dominant_emotion for face in all_faces)
    with st.container(border=True):
        st.markdown("#### Batch summary")
        first, second, third = st.columns(3)
        first.metric("Images", len(analysed))
        second.metric("Faces", len(all_faces))
        if counts:
            top, _ = counts.most_common(1)[0]
            third.metric("Most common", f"{EMOTION_STYLE[top]['emoji']} {top.title()}")
        chart_column, table_column = st.columns((1, 1.2), gap="large")
        with chart_column:
            st.altair_chart(charts.emotion_counts(dict(counts), is_dark()), width="stretch")
        table = pd.DataFrame(
            [
                {
                    "image": name,
                    "faces": len(faces),
                    "overall mood": overall_mood(faces)[0].title() if faces else "—",
                    "average confidence": np.mean([face.confidence for face in faces]) / 100 if faces else None,
                }
                for name, _, faces in analysed
            ]
        )
        with table_column:
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={"average confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
            )
            combined = pd.concat([result_rows(faces, name) for name, _, faces in analysed if faces], ignore_index=True)
            st.download_button(
                "All results (CSV)", combined.to_csv(index=False).encode("utf-8"), "emotion_results.csv", "text/csv",
                icon=":material/table:", on_click="ignore",
            )


def snapshot_page() -> None:
    page_header("Snapshot", "Take a photo", "Capture a still from your webcam for a detailed, full-accuracy analysis.")
    settings = photo_settings()
    snapshot = st.camera_input("Take a photo", label_visibility="collapsed")
    if not snapshot:
        st.caption("Allow camera access in your browser, look at the camera and press **Take Photo**.")
        return
    if not ensure_model():
        return
    try:
        image_rgb = prepare_image(Image.open(io.BytesIO(snapshot.getvalue())))
        faces = analyse_image(image_rgb, settings)
    except Exception as error:
        st.error("The snapshot could not be analysed.")
        with st.expander("Technical details"):
            st.exception(error)
        return
    render_results(image_rgb, faces, "camera", "webcam_snapshot", settings)
