"""About page: problem statement, method, ethics and references."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from emotion_detector.config import PROJECT, TEAM_MEMBERS

from .common import legend_html, page_header, sidebar_footer


def about_page() -> None:
    page_header("About", PROJECT["title"], f"{PROJECT['course']} · {PROJECT['department']}, {PROJECT['institute']}")
    sidebar_footer()
    left, right = st.columns((1.3, 1), gap="large")
    with left:
        st.markdown("#### Problem statement")
        st.markdown(
            "Detect every face in a photo or a live video and classify its visible expression as one of seven "
            "classes: angry, disgust, fear, happy, sad, surprise or neutral. The project supports awareness and "
            "human–computer-interaction research. It recognises **expressions**, not a person's inner feelings."
        )
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.markdown("#### Method")
        st.markdown(
            "1. **Detect** faces with a DeepFace detector (YuNet by default; RetinaFace, MTCNN, SSD or Haar optional).\n"
            "2. **Align** each face so the eyes are level and take a square crop around it.\n"
            "3. **Pre-process** to a 48×48 grayscale image scaled to [0, 1], as in FER-2013.\n"
            "4. **Classify** with DeepFace's pre-trained CNN, averaged with the mirror image (test-time augmentation).\n"
            "5. **Live video** tracks faces between frames and smooths scores over time to avoid flicker."
        )
    with right:
        st.markdown("#### Dataset")
        st.dataframe(
            pd.DataFrame(
                [["FER-2013", "35,887", "48 × 48 grayscale", "7"], ["FER+", "FER-2013 relabelled", "10 annotators/image", "8 incl. contempt"]],
                columns=["Dataset", "Images", "Format", "Classes"],
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("DeepFace's model uses FER-2013 labels. FER+ is an improved relabelling cited in the literature survey.")
        st.markdown("#### Academic mapping")
        st.dataframe(
            pd.DataFrame([[PROJECT["co"], PROJECT["pos"], PROJECT["psos"], PROJECT["sdg"]]], columns=["CO", "POs", "PSOs", "SDG"]),
            hide_index=True,
            width="stretch",
        )
        if TEAM_MEMBERS:
            st.markdown("#### Team")
            st.markdown("\n".join(f"- {member}" for member in TEAM_MEMBERS))
    st.markdown("#### Ethics and limitations")
    st.warning(
        "A facial expression is not a diagnosis or proof of emotion. FER-2013 is low-resolution and imbalanced, and "
        "results vary with lighting, pose, age, skin tone and culture. Use only with consent. Images and video are "
        "processed in memory and never stored.",
        icon=":material/policy:",
    )
    st.markdown("#### References")
    st.markdown(
        "1. Goodfellow et al. (2013), *Challenges in Representation Learning: A report on three machine learning contests* (FER-2013).  \n"
        "2. Barsoum et al. (2016), *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution* (FER+).  \n"
        "3. Serengil & Ozpinar (2021), *HyperExtended LightFace: A Facial Attribute Analysis Framework* (DeepFace).  \n"
        "4. Wu et al. (2023), *YuNet: A Tiny Millisecond-level Face Detector*.  \n"
        "5. Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.  \n"
        "6. Li & Deng (2022), *Deep Facial Expression Recognition: A Survey*.  \n"
        "7. Ekman & Friesen (1971), *Constants across cultures in the face and emotion*."
    )
