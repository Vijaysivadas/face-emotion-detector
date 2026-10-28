"""Static configuration shared by the detector, the visualiser and the UI."""

# Order matches the output layer of DeepFace's emotion CNN (FER-2013 classes).
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# Order used for legends, stacked charts and score lists. Neighbouring colours
# in this order were checked for colour-vision-deficiency separation in both
# light and dark mode, so keep it when adding charts.
DISPLAY_ORDER = ("happy", "neutral", "surprise", "sad", "angry", "fear", "disgust")

EMOTION_STYLE = {
    "angry": {"emoji": "😠", "color": "#E34948", "dark": "#E66767"},
    "disgust": {"emoji": "🤢", "color": "#008300", "dark": "#008300"},
    "fear": {"emoji": "😨", "color": "#6250D6", "dark": "#9085E9"},
    "happy": {"emoji": "😄", "color": "#EDA100", "dark": "#C98500"},
    "sad": {"emoji": "😢", "color": "#2A78D6", "dark": "#3987E5"},
    "surprise": {"emoji": "😲", "color": "#EB6834", "dark": "#D95926"},
    "neutral": {"emoji": "😐", "color": "#1BAF7A", "dark": "#199E70"},
}

# Face detectors offered for photos. Each is a DeepFace detector backend; its
# weights download to ~/.deepface/weights the first time it is used.
DETECTORS = {
    "yunet": "YuNet — fast and accurate (recommended)",
    "retinaface": "RetinaFace — most accurate, slow (~3 s)",
    "mtcnn": "MTCNN — accurate",
    "ssd": "SSD ResNet-10 — fast",
    "opencv": "OpenCV Haar cascade — fastest, frontal faces only",
}
DEFAULT_DETECTOR = "yunet"
# Tried in turn when the chosen detector finds nothing.
DETECTOR_FALLBACK_ORDER = ("yunet", "mtcnn", "opencv")

# Very large photos are downscaled before analysis; detection time grows with
# pixel count and the emotion CNN only ever sees a 48x48 face crop anyway.
MAX_IMAGE_SIDE = 1600

# The CNN's input and DeepFace's intermediate face size (face -> 224 -> 48).
MODEL_INPUT_SIZE = 48
ALIGNED_FACE_SIZE = 224
# Side of the square face crop relative to the detector box's longer side.
# On FER-2013 PrivateTest, 1.0–1.1 matched the training framing best.
CROP_SCALE = 1.05
# Ignore eye landmarks implying a roll larger than this (usually a bad landmark).
MAX_ALIGN_ANGLE = 40.0

YUNET_SCORE_THRESHOLD = 0.75
YUNET_NMS_THRESHOLD = 0.3
# Group photos keep enough resolution for small faces at this size.
YUNET_MAX_SIDE = 1280

# Live video: faces are detected on a small copy of each frame, while the
# emotion crop is taken from the full-resolution frame.
LIVE_DETECTION_WIDTH = 320
LIVE_SCORE_THRESHOLD = 0.6
LIVE_MIN_FACE_SIZE = 28
LIVE_DEFAULT_RATE_HZ = 15
LIVE_DEFAULT_SMOOTHING_SECONDS = 0.35
# A face must be seen this many times before it is drawn (filters flickers)
# and is kept for this long after it was last seen (bridges missed frames).
LIVE_MIN_HITS = 2
LIVE_TRACK_TIMEOUT_SECONDS = 0.6
LIVE_HISTORY_SECONDS = 90
LIVE_RESOLUTIONS = {
    "Fast · 480×360": (480, 360),
    "Balanced · 640×480": (640, 480),
    "HD · 1280×720": (1280, 720),
}
LIVE_DEFAULT_RESOLUTION = "Balanced · 640×480"

# Extra public STUN servers handed to the browser. TURN relays come from
# streamlit-webrtc when Cloudflare/Twilio/Hugging Face credentials are set.
PUBLIC_STUN_SERVERS = (
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
    "stun:stun.cloudflare.com:3478",
)

PROJECT = {
    "group": "Group 5",
    "title": "Emotion Detection from Facial Images",
    "course": "PCCSL 508 - Machine Learning Lab",
    "institute": "Federal Institute of Science and Technology (FISAT)",
    "department": "Department of Computer Science and Engineering",
    "semester": "Semester V, CSE - Academic Year 2026-27",
    "model": "FER pre-trained CNN via DeepFace",
    "deployment": "Streamlit",
    "co": "CO4",
    "pos": "PO1, PO2, PO5, PO8",
    "psos": "PSO1, PSO2",
    "sdg": "SDG 3: Good Health and Well-being",
}

# Add your team members' names here; they appear in the sidebar and About tab.
TEAM_MEMBERS: list[str] = []
