"""Streamlit interface for the Group 5 facial emotion detection project."""

from __future__ import annotations

import os

# Must precede TensorFlow: RetinaFace needs the legacy tf-keras API, and every
# model in the process has to agree with it.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st

from ui.about import about_page
from ui.common import inject_css, keep_settings
from ui.insights import insights_page
from ui.live import live_page
from ui.photos import snapshot_page, upload_page


def main() -> None:
    st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="wide")
    inject_css()
    keep_settings()
    # Only the selected page runs, so the live camera never waits for photo
    # analysis and no hidden camera widget competes for the webcam.
    page = st.navigation(
        [
            st.Page(live_page, title="Live camera", icon=":material/videocam:", url_path="live", default=True),
            st.Page(upload_page, title="Photos", icon=":material/photo_library:", url_path="photos"),
            st.Page(snapshot_page, title="Snapshot", icon=":material/photo_camera:", url_path="snapshot"),
            st.Page(insights_page, title="Model insights", icon=":material/insights:", url_path="insights"),
            st.Page(about_page, title="About", icon=":material/info:", url_path="about"),
        ],
        position="top",
    )
    with st.sidebar:
        st.markdown("### 🎭 Emotion Detection")
        st.caption("DeepFace CNN · FER-2013 · 7 expressions")
    page.run()


if __name__ == "__main__":
    main()
