# Emotion Detection from Facial Images

A Streamlit web application for Group 5's PCCSL 508 Machine Learning Lab micro-project. It uses DeepFace's pre-trained FER-2013 emotion CNN to identify the visible facial expression of every face in an image.

> The app recognises facial expressions, not a person's true feelings. It is an educational demonstration, not a diagnostic or surveillance tool.

## Features

- Upload JPG, JPEG, PNG, WEBP, or BMP images, including group photographs.
- Capture and analyse a browser webcam snapshot.
- Live webcam mode with results drawn on the video stream.
- Seven expression classes: angry, disgust, fear, happy, sad, surprise, and neutral.
- Colour-coded face boxes, per-face confidence charts, group mood summary, and face thumbnails.
- Detector selection: OpenCV, SSD, MTCNN, RetinaFace, and YuNet.
- Graceful no-face handling and an option for already-cropped faces.
- Download annotated PNG images and detailed CSV results.
- Self-contained About page covering the model, dataset, ethics, references, and CO/PO/PSO/SDG mapping.

## Run locally

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`). The first analysis downloads DeepFace's small emotion-model weight file. The live-video tab may need browser camera permission and works most reliably on localhost.

For Windows machines where `py` is not registered, run the Python executable directly:

```powershell
C:\path\to\python.exe -m venv .venv
```

## Project structure

```text
app.py                         Streamlit UI and all result views
emotion_detector/
  config.py                    labels, styles, project metadata, detector options
  detector.py                  DeepFace adapter, image preparation, result model
  visualize.py                 OpenCV annotations and face crops
evaluate.py                    optional FER-2013 evaluation utility
tests/test_detector.py         lightweight unit tests
```

## Model and dataset

DeepFace uses a pre-trained convolutional neural network with a `48 × 48 × 1` grayscale face input and seven softmax outputs. The pipeline is face detection → optional alignment → crop/pre-process → CNN → probability scores. The app uses inference only; it does not train a network or retain images.

The CNN is based on FER-2013: 35,887 48 × 48 grayscale faces in seven classes. FER+ re-labels FER-2013 with ten annotators per image and is cited as related work. FER-2013 is low-resolution and imbalanced (especially for disgust), so predictions should be interpreted carefully.

## Optional evaluation

Download an image-folder FER-2013 test set structured as `test/<emotion>/*.jpg`, then run:

```powershell
python evaluate.py --dataset C:\datasets\fer2013\test --output evaluation_output
```

It writes predictions, a precision/recall/F1 report, and a confusion matrix. The dataset is intentionally not committed to this repository.

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository and push this folder.
2. In Streamlit Community Cloud, choose the repository and set the entry point to `app.py`.
3. Select Python 3.12 in Advanced settings and deploy.

## Academic mapping and ethical use

| Course outcome | Programme outcomes | PSOs | Sustainable Development Goal |
| --- | --- | --- | --- |
| CO4 | PO1, PO2, PO5, PO8 | PSO1, PSO2 | SDG 3: Good Health and Well-being |

Use only with informed consent. Facial expression is not equivalent to a person's inner emotion; quality can vary with pose, lighting, culture, age, skin tone, and dataset representation. Never use this project for medical diagnosis or high-stakes decisions.

## References

1. Goodfellow et al. (2013), *Challenges in Representation Learning: A report on three machine learning contests*.
2. Barsoum et al. (2016), *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution*.
3. Serengil & Ozpinar (2021), *HyperExtended LightFace: A Facial Attribute Analysis Framework*.
4. Li & Deng (2022), *Deep Facial Expression Recognition: A Survey*.
5. Ekman & Friesen (1971), *Constants across cultures in the face and emotion*.
