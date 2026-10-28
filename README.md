# Emotion Detection from Facial Images

A Streamlit web application for Group 5's PCCSL 508 Machine Learning Lab micro-project. It uses DeepFace's pre-trained FER-2013 emotion CNN to identify the visible facial expression of every face in a photo or live video.

> The app recognises facial expressions, not a person's true feelings. It is an educational demonstration, not a diagnostic or surveillance tool.

## Features

- **Live camera**: real-time emotion recognition with face tracking, smoothed labels, a status overlay, a live emotion timeline and session summary, and one-click capture for a detailed analysis.
- **Photos**: upload one or many images (group photos included), with a batch summary, per-face cards, the exact 48×48 image the CNN saw, and a Grad-CAM attention map.
- **Snapshot**: take a webcam photo for a full-accuracy analysis.
- **Model insights**: architecture and parameter charts, learned first-layer filters, weight distributions, training curves, a confusion matrix, per-class precision/recall/F1, calibration, ablations and a face-detector benchmark.
- Seven expression classes: angry, disgust, fear, happy, sad, surprise and neutral.
- Detector choice (YuNet, RetinaFace, MTCNN, SSD, OpenCV) with automatic fallback, light and dark themes, PNG and CSV downloads.

## Run locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Open the URL Streamlit prints (normally `http://localhost:8501`). The first start downloads DeepFace's small emotion and YuNet weight files. Run the tests with `python -m pytest`.

## How it works

1. **Detect** faces with a DeepFace detector backend. YuNet is the default: it found 91% of benchmark faces in 16 ms, compared with 64% for the Haar cascade.
2. **Align and crop**: rotate so the eyes are level and take a square crop (1.05× the box), which matches FER-2013's framing.
3. **Pre-process** to 48×48 grayscale scaled to [0, 1], exactly as DeepFace does.
4. **Classify** with DeepFace's CNN, averaging each face with its mirror image (test-time augmentation).
5. **Live video** runs detection and classification on the newest frame in a background thread (the CNN is compiled once, cutting a call from about 13 ms to 4 ms). Faces are tracked between updates and scores are smoothed over time, so video stays fluid and labels do not flicker.

Measured on FER-2013 PrivateTest (3,589 held-out faces):

| Pre-processing | Accuracy |
| --- | --- |
| Raw 0–255 pixels (the previous live mode) | 48.5% |
| DeepFace's [0, 1] scaling | 56.5% |
| + mirror test-time augmentation (used by the app) | 57.3% |

## Live camera tips

- On `localhost` or a local network the app connects directly, with no STUN/TURN look-ups, so the camera starts in about 3 seconds.
- A **deployed** app needs a TURN relay, or you will see *"Connection is taking longer than expected"*. Add one of these to the app's Streamlit secrets as root-level keys:
  - Cloudflare Realtime TURN: `CLOUDFLARE_TURN_KEY_ID` + `CLOUDFLARE_TURN_KEY_API_TOKEN`
  - Any TURN service (for example Metered): `TURN_URLS` (comma-separated) + `TURN_USERNAME` + `TURN_CREDENTIAL`
  - Hugging Face: `HF_TOKEN`
- The live page's *"Camera not showing or connection slow?"* box shows whether a relay is active, and why it failed if it isn't.
- Pick **Fast · 480×360** in the sidebar on slow connections.

## Project structure

```text
app.py                         page navigation
ui/                            Streamlit pages: live, photos/snapshot, model insights, about
emotion_detector/
  config.py                    labels, colours, detector and live-video settings
  detector.py                  detection, alignment, DeepFace emotion classifier
  live.py                      background live analysis, face tracking and smoothing
  visualize.py                 OpenCV annotations and live overlay
  charts.py                    Altair charts
  explain.py                   layer table, weight statistics, Grad-CAM
scripts/
  build_model_report.py        evaluates the model → reports/model_report.json
  train_reproduction.py        re-trains the architecture → reports/training_history.json
reports/                       measured results shown on the Model insights page
evaluate.py                    evaluation on an image-folder FER-2013 test set
tests/test_detector.py         unit tests
```

## Reproducing the reports

Both scripts download FER-2013 (Training, PublicTest and PrivateTest splits) to `data/fer2013/` on first use:

```powershell
python scripts/build_model_report.py      # ~5 minutes on a laptop CPU
python scripts/train_reproduction.py      # ~15 minutes, 30 epochs
```

DeepFace publishes pre-trained weights but no training logs. `train_reproduction.py` re-trains the identical architecture from random weights to show its learning curves; the app always uses DeepFace's official weights. For an image-folder dataset (`test/<emotion>/*.jpg`) you can still run `python evaluate.py --dataset <folder>`.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose the repository with `app.py` as the entry point and Python 3.12.
3. Add a TURN credential to the app's secrets (see *Live camera tips*) so the live camera works on every network.

## Academic mapping and ethical use

| Course outcome | Programme outcomes | PSOs | Sustainable Development Goal |
| --- | --- | --- | --- |
| CO4 | PO1, PO2, PO5, PO8 | PSO1, PSO2 | SDG 3: Good Health and Well-being |

Use only with informed consent. Facial expression is not equivalent to a person's inner emotion; quality varies with pose, lighting, culture, age, skin tone and dataset representation. Never use this project for medical diagnosis or high-stakes decisions.

## References

1. Goodfellow et al. (2013), *Challenges in Representation Learning: A report on three machine learning contests*.
2. Barsoum et al. (2016), *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution*.
3. Serengil & Ozpinar (2021), *HyperExtended LightFace: A Facial Attribute Analysis Framework*.
4. Wu et al. (2023), *YuNet: A Tiny Millisecond-level Face Detector*.
5. Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.
6. Li & Deng (2022), *Deep Facial Expression Recognition: A Survey*.
7. Ekman & Friesen (1971), *Constants across cultures in the face and emotion*.
