"""Static configuration shared by the detector, the visualiser and the UI."""

# Order matches the output layer of DeepFace's emotion CNN (FER-2013 classes).
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

EMOTION_STYLE = {
    "angry": {"emoji": "😠", "color": "#E5484D"},
    "disgust": {"emoji": "🤢", "color": "#30A46C"},
    "fear": {"emoji": "😨", "color": "#8E4EC6"},
    "happy": {"emoji": "😄", "color": "#FFB224"},
    "sad": {"emoji": "😢", "color": "#3E63DD"},
    "surprise": {"emoji": "😲", "color": "#F76B15"},
    "neutral": {"emoji": "😐", "color": "#8B8D98"},
}

# Face detectors offered in the UI. All of them ship with DeepFace; weights are
# downloaded to ~/.deepface/weights the first time each one is used.
DETECTORS = {
    "opencv": "OpenCV Haar cascade (fastest)",
    "ssd": "SSD ResNet-10 (fast, handles angles better)",
    "mtcnn": "MTCNN (accurate, slower)",
    "retinaface": "RetinaFace (most accurate, slowest)",
    "yunet": "YuNet (modern and fast)",
}
DEFAULT_DETECTOR = "opencv"

# Very large photos are downscaled before analysis; detection time grows with
# pixel count and the emotion CNN only ever sees a 48x48 face crop anyway.
MAX_IMAGE_SIDE = 1600

# Live video is analysed on a smaller copy, then labels are drawn on the
# original frame. The emotion CNN ultimately sees 48x48 face crops, so this
# substantially improves responsiveness without reducing useful detail.
LIVE_ANALYSIS_WIDTH = 480
LIVE_ANALYZE_EVERY_N_FRAMES = 5

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
