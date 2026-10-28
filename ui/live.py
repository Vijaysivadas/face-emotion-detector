"""Live camera page: WebRTC streaming, overlays and a real-time dashboard."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

from emotion_detector import charts
from emotion_detector.config import EMOTIONS, LIVE_RESOLUTIONS, PUBLIC_STUN_SERVERS
from emotion_detector.detector import emotion_classifier
from emotion_detector.live import BoxEaser, LiveEmotionEngine
from emotion_detector.visualize import draw_live_overlay

from .common import chip, emotion_hero, ensure_model, is_dark, page_header, sidebar_footer, warm_live_detector
from .photos import analyse_image, render_results

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    from streamlit_webrtc.credentials import get_cloudflare_ice_servers, get_hf_ice_servers
except ImportError:  # pragma: no cover - optional dependency
    av = None


if av is not None:

    class EmotionVideoProcessor(VideoProcessorBase):
        """Draws the latest tracked emotions on every frame.

        ``recv`` only hands the frame to the engine and draws, so video keeps
        its frame rate however long inference takes.
        """

        def __init__(self) -> None:
            self.engine = LiveEmotionEngine(emotion_classifier())
            self.easer = BoxEaser()
            self.show_hud = True

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            self.engine.submit(image)
            output = image.copy()  # the engine may still be reading ``image``
            faces = [face for _, face in self.easer.step(self.engine.visible_tracks())]
            hud = None
            if self.show_hud:
                video_fps, analysis_hz = self.engine.rates()
                hud = f"LIVE  {video_fps:.0f} fps  |  AI {analysis_hz:.0f}/s  |  {len(faces)} face{'' if len(faces) == 1 else 's'}"
            draw_live_overlay(output, faces, hud)
            return av.VideoFrame.from_ndarray(output, format="rgb24")

        def on_ended(self) -> None:
            self.engine.stop()


def _is_local(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost" or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _secret(name: str) -> str:
    """A root-level Streamlit secret, or an environment variable of the same name."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        value = None
    return str(value or os.environ.get(name, "")).strip()


def _turn_servers() -> tuple[list[dict], str | None]:
    """TURN relays from the first provider whose credentials work, plus why others failed.

    Secrets are read on every run (the provider helpers cache per credential),
    so a newly added secret works at once; streamlit-webrtc's own
    auto-detection would keep a "no relay" result cached for an hour.
    """
    problems = []
    cloudflare = (_secret("CLOUDFLARE_TURN_KEY_ID"), _secret("CLOUDFLARE_TURN_KEY_API_TOKEN"))
    if all(cloudflare):
        try:
            return list(get_cloudflare_ice_servers(*cloudflare)), None
        except Exception as error:
            problems.append(f"Cloudflare: {error}")
    if _secret("TURN_URLS"):
        urls = [url.strip() for url in _secret("TURN_URLS").split(",") if url.strip()]
        return [{"urls": urls, "username": _secret("TURN_USERNAME"), "credential": _secret("TURN_CREDENTIAL")}], None
    if _secret("HF_TOKEN"):
        try:
            return list(get_hf_ice_servers(_secret("HF_TOKEN"))), None
        except Exception as error:
            problems.append(f"Hugging Face: {error}")
    return [], "; ".join(problems) or None


def ice_configuration() -> tuple[dict, dict, str, str | None]:
    """WebRTC ICE settings for the browser and the server, a mode label and any relay problem.

    On localhost or a LAN the browser and server reach each other directly,
    and STUN look-ups only delay the connection (each unreachable STUN
    server can add ~5 s on the server side), so none are used. Remote
    deployments need a TURN relay from the app's secrets; without one only
    public STUN servers are offered, which many networks cannot get through.
    """
    host = urlparse(st.context.url or "").hostname
    if _is_local(host):
        return {"iceServers": []}, {"iceServers": []}, "direct", None
    turn, problem = _turn_servers()
    stun = {"urls": list(PUBLIC_STUN_SERVERS)}
    frontend = {"iceServers": turn + [stun]}
    server = {"iceServers": turn + [{"urls": PUBLIC_STUN_SERVERS[0]}]}
    return frontend, server, "turn" if turn else "stun", problem


def live_settings() -> None:
    with st.sidebar:
        st.subheader("Live settings")
        st.selectbox(
            "Camera quality", list(LIVE_RESOLUTIONS), key="live_resolution",
            help="Takes effect when the camera (re)starts. Choose Fast on slow networks or older laptops.",
        )
        st.slider(
            "Emotion updates per second", 5, 30, key="live_rate",
            help="How often the CNN re-scores faces. Video stays smooth either way; higher uses more CPU.",
        )
        st.slider(
            "Smoothing (seconds)", 0.0, 1.5, step=0.05, key="live_smoothing",
            help="Averages scores over time so labels don't flicker. 0 shows raw per-frame predictions.",
        )
        st.toggle("Mirror test-time augmentation", key="live_tta", help="Averages each face with its mirror image.")
        st.toggle("Status overlay on video", key="live_hud")
    sidebar_footer()


def _idle_panel() -> None:
    st.markdown(
        '<div class="panel"><h4>Start the live view</h4>'
        '<div class="step"><b>1</b><span>Press <b style="all:unset;font-weight:600">Start camera</b> and allow camera access.</span></div>'
        '<div class="step"><b>2</b><span>Face the camera in even light; several faces are tracked at once.</span></div>'
        '<div class="step"><b>3</b><span>Watch the live emotion mix here, or capture a frame for a detailed analysis.</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


# Every 2 s: rebuilding charts more often made video delivery burstier.
@st.fragment(run_every=2.0)
def live_dashboard(ctx) -> None:
    """Refreshes on its own without rerunning the page or touching the video."""
    processor = ctx.video_processor if ctx.state.playing else None
    if processor is None:
        _idle_panel()
        return
    snapshot = processor.engine.snapshot()
    dark = is_dark()
    first, second, third = st.columns(3)
    first.metric("Video", f"{snapshot.video_fps:.0f} fps")
    second.metric("AI updates", f"{snapshot.analysis_hz:.0f}/s")
    third.metric(
        "Latency", f"{snapshot.detect_ms + snapshot.classify_ms:.0f} ms",
        help=f"{snapshot.detector_name} face detection {snapshot.detect_ms:.1f} ms + emotion CNN {snapshot.classify_ms:.1f} ms",
    )
    if snapshot.error:
        st.caption(f"Last frame skipped: {snapshot.error}")
    if not snapshot.faces:
        st.info("Looking for a face… face the camera in good, even light.", icon=":material/face:")
    else:
        _, face = snapshot.faces[0]
        emotion_hero(face)
        if len(snapshot.faces) > 1:
            st.markdown(
                "".join(chip(f.dominant_emotion, f"Face {i + 1} · {f.dominant_emotion.title()} {f.confidence:.0f}%") for i, (_, f) in enumerate(snapshot.faces)),
                unsafe_allow_html=True,
            )
        st.altair_chart(charts.emotion_bars(face.scores, dark, height=185), width="stretch")
    timeline = charts.emotion_timeline(snapshot.history, dark)
    if timeline is not None:
        st.markdown("**Emotion mix, last 60 seconds** · largest face")
        st.altair_chart(timeline, width="stretch")
    if any(snapshot.time_share.values()):
        with st.expander("Session summary", icon=":material/timer:"):
            st.altair_chart(charts.time_share(snapshot.time_share, dark), width="stretch")
            rows = [
                {"seconds": round(t, 2), **{f"{e}_percent": round(float(s[i]), 2) for i, e in enumerate(EMOTIONS)}}
                for t, s in snapshot.history
                if s is not None
            ]
            if rows:
                st.download_button(
                    "Session log (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), "live_session.csv", "text/csv",
                    icon=":material/download:", on_click="ignore",
                )


def _connection_help(mode: str, problem: str | None) -> None:
    labels = {
        "direct": ("#1BAF7A", "Direct local connection", "Browser and server are on the same network, so no relay is needed."),
        "turn": ("#1BAF7A", "TURN relay configured", "Connections work behind strict firewalls and mobile networks."),
        "stun": ("#EDA100", "STUN only", "No TURN relay, so the camera may not connect on many networks (see below)."),
    }
    color, title, detail = labels[mode]
    with st.expander("Camera not showing or connection slow?", icon=":material/help:", expanded=problem is not None):
        st.markdown(
            f'<span class="status-dot" style="background:{color}"></span>**{title}**: {detail}',
            unsafe_allow_html=True,
        )
        if problem:
            st.error(f"A TURN relay is configured but could not be reached. {problem}", icon=":material/error:")
        st.markdown(
            "- Allow camera access in the address bar, and close other apps or tabs using the camera.\n"
            "- Use **Switch camera** if your laptop has more than one camera.\n"
            "- Pick **Fast · 480×360** in the sidebar on slow connections, then restart the camera.\n"
            "- Chrome and Edge give the most reliable WebRTC support.\n"
            "- **Deployed apps:** if you see *Connection is taking longer than expected*, add a TURN relay to the app's "
            "secrets: Cloudflare `CLOUDFLARE_TURN_KEY_ID` + `CLOUDFLARE_TURN_KEY_API_TOKEN`; any TURN service's "
            "`TURN_URLS` + `TURN_USERNAME` + `TURN_CREDENTIAL`; or `HF_TOKEN` (Hugging Face)."
        )


def live_page() -> None:
    page_header(
        "Live",
        "Live emotion recognition",
        "Faces are tracked on every video frame while DeepFace's emotion CNN re-scores them in the background, "
        "so the video stays smooth and labels stay steady.",
    )
    live_settings()
    if av is None:
        st.error("Live video needs `streamlit-webrtc` and `av`. Install the project requirements to enable it.")
        return
    if not ensure_model():
        return
    try:
        warm_live_detector()
    except Exception:
        pass  # the engine falls back to OpenCV's bundled Haar cascade

    frontend_config, server_config, mode, relay_problem = ice_configuration()
    width, height = LIVE_RESOLUTIONS[st.session_state.live_resolution]
    video_column, dashboard_column = st.columns((1.4, 1), gap="large")
    with video_column:
        ctx = webrtc_streamer(
            key="emotion-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            # Size is only "ideal" (hard limits stop some cameras starting); the
            # frame-rate cap is safe because browsers just drop extra frames.
            media_stream_constraints={
                "video": {"width": {"ideal": width}, "height": {"ideal": height}, "frameRate": {"ideal": 24, "max": 24}},
                "audio": False,
            },
            frontend_rtc_configuration=frontend_config,
            server_rtc_configuration=server_config,
            async_processing=True,
            video_html_attrs={
                "autoPlay": True, "muted": True, "playsInline": True, "controls": False,
                "style": {"width": "100%", "borderRadius": "14px"},
            },
            translations={"start": "Start camera", "stop": "Stop camera", "select_device": "Switch camera"},
        )
        processor = ctx.video_processor
        if processor is not None:
            processor.engine.settings.rate_hz = float(st.session_state.live_rate)
            processor.engine.settings.smoothing_seconds = float(st.session_state.live_smoothing)
            processor.engine.settings.tta = bool(st.session_state.live_tta)
            processor.show_hud = bool(st.session_state.live_hud)
        if st.button(
            "Capture frame for detailed analysis", icon=":material/photo_camera:",
            disabled=processor is None, width="stretch",
        ):
            frame = processor.engine.latest_frame if processor is not None else None
            if frame is not None:
                st.session_state.live_capture = np.array(frame)
        _connection_help(mode, relay_problem)
    with dashboard_column:
        live_dashboard(ctx)

    capture = st.session_state.get("live_capture")
    if capture is not None:
        st.divider()
        header, clear = st.columns((4, 1), vertical_alignment="center")
        header.markdown("### Captured frame")
        if clear.button("Clear", icon=":material/close:", width="stretch"):
            del st.session_state.live_capture
            st.rerun()
        settings = {
            "detector": st.session_state.opt_detector,
            "align": st.session_state.opt_align,
            "tta": st.session_state.opt_tta,
            "detector_fallback": st.session_state.opt_detector_fallback,
            "whole_image_fallback": st.session_state.opt_whole_image,
        }
        render_results(capture, analyse_image(capture, settings), "live-capture", "live_capture", settings)
