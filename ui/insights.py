"""Model insights: architecture, weights, training curves, evaluation and dataset."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import streamlit as st

from emotion_detector import charts
from emotion_detector.config import CROP_SCALE, DISPLAY_ORDER, EMOTIONS
from emotion_detector.explain import first_layer_filters, layer_table, weight_samples, weight_statistics

from .common import ensure_model, get_classifier, is_dark, legend_html, load_report, page_header, sidebar_footer

REPORT_HINT = "Generate it with `python scripts/build_model_report.py` (downloads FER-2013 on first run)."


@st.cache_data(show_spinner=False)
def _architecture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, str, int]:
    model = get_classifier().model
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary(line_length=96)
    return (
        layer_table(model),
        weight_samples(model),
        weight_statistics(model),
        first_layer_filters(model),
        buffer.getvalue(),
        int(model.count_params()),
    )


def _pipeline_diagram() -> str:
    ink = "#C3C2B7" if is_dark() else "#52514E"
    nodes = [
        ("photo", "Photo / video frame"),
        ("detect", "Face detection\\nYuNet · RetinaFace · MTCNN · SSD · Haar"),
        ("align", f"Eye alignment +\\nsquare crop ({CROP_SCALE}× box)"),
        ("prep", "Grayscale · 48×48\\nscaled to [0, 1]"),
        ("cnn", "DeepFace emotion CNN\\n(+ mirrored copy)"),
        ("softmax", "Softmax over\\n7 expressions"),
        ("out", "Overlay · charts\\n(live: tracking + smoothing)"),
    ]
    body = "; ".join(f'{key} [label="{label}"]' for key, label in nodes)
    edges = " -> ".join(key for key, _ in nodes)
    return (
        "digraph { rankdir=LR; bgcolor=transparent; "
        f'node [shape=box style="rounded" color="{ink}" fontcolor="{ink}" fontname="Helvetica" fontsize=11 margin="0.18,0.1"]; '
        f'edge [color="{ink}" arrowsize=0.7]; {body}; {edges} }}'
    )


def _overview(report: dict | None, history: dict | None, total_params: int) -> None:
    cards = st.columns(5)
    cards[0].metric("Parameters", f"{total_params / 1e6:.2f} M", border=True, help="All trainable (float32).")
    cards[1].metric("Model size", f"{total_params * 4 / 1024**2:.1f} MB", border=True)
    cards[2].metric("Input", "48×48 gray", border=True)
    if report:
        test = report["test"]
        cards[3].metric("Test accuracy", f"{test['accuracy']:.1%}", border=True, help="FER-2013 PrivateTest, 3,589 faces, with mirror TTA.")
        cards[4].metric("Macro F1", f"{test['macro_f1']:.3f}", border=True)
    else:
        cards[3].metric("Classes", "7", border=True)
        cards[4].metric("Dataset", "FER-2013", border=True)
    st.markdown("#### How a prediction is made")
    st.graphviz_chart(_pipeline_diagram(), width="stretch")
    if report:
        ablation = {row["variant"]: row["accuracy"] for row in report["ablation"]}
        speed = {row["setup"]: row["ms"] for row in report["speed_ms"]}
        detectors = {row["detector"]: row for row in report["detectors"]}
        st.markdown("#### Key measurements")
        st.markdown(
            f"- Feeding raw 0–255 pixels (what the old live mode did) scores **{ablation['Raw 0–255 pixels (old live mode)']:.1%}**; "
            f"DeepFace's [0, 1] scaling scores **{ablation['DeepFace preprocessing ([0, 1])']:.1%}**, and averaging with the mirror image "
            f"reaches **{ablation['+ mirror test-time augmentation']:.1%}** on PrivateTest.\n"
            f"- Compiling the CNN cuts one face from **{speed['DeepFace default (eager, 1 face)']:.1f} ms** to "
            f"**{speed['Compiled call, 1 face']:.1f} ms**.\n"
            f"- YuNet finds **{detectors['YuNet']['detection_rate']:.0%}** of benchmark faces in "
            f"**{detectors['YuNet']['ms_per_image']:.0f} ms**, against **{detectors['OpenCV Haar']['detection_rate']:.0%}** for the Haar cascade."
        )
    if history:
        st.markdown(
            f"- Re-training the same architecture from scratch peaks at **{history['best_val_accuracy']:.1%}** validation accuracy "
            f"(epoch {history['best_epoch']}), close to DeepFace's published weights (see the **Training** tab)."
        )


def _architecture_tab(layers: pd.DataFrame, samples: pd.DataFrame, statistics: pd.DataFrame, filters: np.ndarray, summary: str) -> None:
    dark = is_dark()
    st.markdown("#### Layers")
    st.dataframe(
        layers.drop(columns=["activations"]),
        hide_index=True,
        width="stretch",
        column_config={
            "parameters": st.column_config.NumberColumn(format="localized"),
            "share of parameters": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )
    left, right = st.columns(2, gap="large")
    trainable = layers[layers["parameters"] > 0]
    with left:
        st.markdown("**Parameters per layer**")
        st.altair_chart(
            charts.single_bars(trainable, "layer", "parameters", dark, ",.0f", sort=list(trainable["layer"]), title="Parameters"),
            width="stretch",
        )
        largest = trainable.loc[trainable["parameters"].idxmax()]
        st.caption(f"`{largest['layer']}` alone holds {largest['share of parameters']:.0%} of all weights: the fully connected layers dominate.")
    with right:
        st.markdown("**Feature-map size per layer**")
        st.altair_chart(
            charts.single_bars(layers, "layer", "activations", dark, ",.0f", sort=list(layers["layer"]), title="Values per face"),
            width="stretch",
        )
        st.caption("Convolutions and pooling shrink a 2,304-pixel face into 128 features before classification.")

    st.markdown("#### What the first layer learned")
    filter_column, text_column = st.columns((1, 1.3), gap="large")
    filter_column.image(filters, caption="64 learned 5×5 filters of the first convolution", width="stretch")
    text_column.markdown(
        "Each tile is one filter. **Orange** weights respond to brighter pixels, **blue** to darker ones and "
        "**grey** is close to zero. Many filters are edge and gradient detectors at different angles: "
        "the building blocks for eyebrows, eye corners and mouth shape."
    )
    st.markdown("#### Weight distributions")
    st.altair_chart(charts.weight_histograms(samples, dark), width="content")
    st.dataframe(
        statistics, hide_index=True, width="stretch",
        column_config={column: st.column_config.NumberColumn(format="%.4f") for column in ("mean", "std", "min", "max")}
        | {"near-zero (|w| < 1e-3)": st.column_config.NumberColumn(format="percent")},
    )
    with st.expander("Keras model summary", icon=":material/data_object:"):
        st.code(summary, language=None)


def _training_tab(history: dict | None) -> None:
    if not history:
        st.info("No training run found. Run `python scripts/train_reproduction.py` to create `reports/training_history.json`.")
        return
    st.info(
        "DeepFace publishes pre-trained weights but no training logs. To show how this network learns, the identical "
        "architecture was **re-trained from random weights** on FER-2013 with the settings below. The app itself still "
        "uses DeepFace's official weights.",
        icon=":material/science:",
    )
    epochs = pd.DataFrame(history["epochs"])
    last = epochs.iloc[-1]
    cards = st.columns(4)
    cards[0].metric("Best validation accuracy", f"{history['best_val_accuracy']:.1%}", f"epoch {history['best_epoch']}", delta_color="off", delta_arrow="off", border=True)
    cards[1].metric("Test accuracy (best epoch)", f"{history['test_accuracy_best_epoch']:.1%}", border=True, help="FER-2013 PrivateTest.")
    cards[2].metric(
        "Final train vs validation", f"{last['accuracy']:.0%} / {last['val_accuracy']:.0%}",
        f"gap {last['accuracy'] - last['val_accuracy']:.0%}", delta_color="off", delta_arrow="off", border=True,
    )
    cards[3].metric("Training time", f"{epochs['seconds'].sum() / 60:.1f} min", f"{len(epochs)} epochs on CPU", delta_color="off", delta_arrow="off", border=True)
    dark = is_dark()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Loss**")
        st.altair_chart(charts.training_curve(epochs, "loss", history["best_epoch"], dark), width="stretch")
    with right:
        st.markdown("**Accuracy**")
        st.altair_chart(charts.training_curve(epochs, "accuracy", history["best_epoch"], dark), width="stretch")
    best_loss_epoch = int(epochs.loc[epochs["val_loss"].idxmin(), "epoch"])
    st.caption(
        f"Validation loss is lowest at epoch {best_loss_epoch} and rises afterwards while training loss keeps falling: "
        "the network starts memorising the training faces (overfitting). Data augmentation, more dropout or early "
        "stopping are the usual remedies."
    )
    config = history["config"]
    left, right = st.columns((1, 1.4), gap="large")
    with left:
        st.markdown("**Hyperparameters**")
        st.dataframe(
            pd.DataFrame({"setting": list(config), "value": [f"{value:,}" if isinstance(value, int) else str(value) for value in config.values()]}),
            hide_index=True, width="stretch",
        )
    with right:
        st.markdown("**Every epoch**")
        st.dataframe(
            epochs, hide_index=True, width="stretch", height=320,
            column_config={
                "loss": st.column_config.NumberColumn(format="%.3f"),
                "val_loss": st.column_config.NumberColumn(format="%.3f"),
                "accuracy": st.column_config.NumberColumn(format="percent"),
                "val_accuracy": st.column_config.NumberColumn(format="percent"),
                "seconds": st.column_config.NumberColumn(format="%.1f s"),
            },
        )


def _evaluation_tab(report: dict | None) -> None:
    if not report:
        st.info(f"No evaluation report found. {REPORT_HINT}")
        return
    dark = is_dark()
    labels = [emotion.title() for emotion in EMOTIONS]
    test = report["test"]
    cards = st.columns(5)
    cards[0].metric("Accuracy", f"{test['accuracy']:.1%}", border=True)
    cards[1].metric("Top-2 accuracy", f"{test['top2_accuracy']:.1%}", border=True, help="The true emotion is one of the two most likely.")
    cards[2].metric("Macro F1", f"{test['macro_f1']:.3f}", border=True)
    cards[3].metric("Weighted F1", f"{test['weighted_f1']:.3f}", border=True)
    cards[4].metric("Calibration error", f"{test['expected_calibration_error']:.3f}", border=True, help="Expected calibration error (lower is better).")
    st.caption("DeepFace's pre-trained weights on FER-2013 **PrivateTest** (3,589 held-out faces), using the app's mirror test-time augmentation.")

    st.markdown("#### Accuracy on each FER-2013 split")
    rows = []
    split_names = {"train": "Training", "publicTest": "PublicTest", "privateTest": "PrivateTest"}
    for split, values in report["accuracy_by_split"].items():
        rows.append({"Split": split_names[split], "Variant": "Single pass", "Accuracy": values["single"]})
        rows.append({"Split": split_names[split], "Variant": "With mirror TTA", "Accuracy": values["tta"]})
    st.altair_chart(
        charts.grouped_bars(pd.DataFrame(rows), "Split", "Variant", "Accuracy", ["Single pass", "With mirror TTA"], dark,
                            category_sort=list(split_names.values()), domain=(0, 0.8), height=240),
        width="stretch",
    )
    st.caption("Training-split accuracy is clearly higher than on the two test splits: the network has seen those faces.")

    left, right = st.columns((1.15, 1), gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        st.altair_chart(charts.confusion_heatmap(np.array(test["confusion_matrix"]), labels, dark), width="stretch")
        st.caption("Each row is a true emotion; cells show where its faces were predicted. The diagonal is recall.")
    with right:
        st.markdown("#### Precision, recall and F1")
        per_class = test["per_class"]
        metric_rows = [
            {"Emotion": emotion.title(), "Metric": metric.title() if metric != "f1" else "F1", "Score": per_class[emotion][metric]}
            for emotion in DISPLAY_ORDER
            for metric in ("precision", "recall", "f1")
        ]
        st.altair_chart(
            charts.grouped_bars(pd.DataFrame(metric_rows), "Emotion", "Metric", "Score", ["Precision", "Recall", "F1"], dark,
                                category_sort=[e.title() for e in DISPLAY_ORDER], domain=(0, 1), height=330),
            width="stretch",
        )
        best = max(per_class, key=lambda e: per_class[e]["f1"])
        worst = min(per_class, key=lambda e: per_class[e]["f1"])
        row = np.array(test["confusion_matrix"][EMOTIONS.index(worst)], dtype=float)
        row[EMOTIONS.index(worst)] = -1
        confused_with = EMOTIONS[int(row.argmax())]
        st.caption(
            f"**{best.title()}** is recognised best (F1 {per_class[best]['f1']:.2f}); **{worst.title()}** is hardest "
            f"(F1 {per_class[worst]['f1']:.2f}) and is most often mistaken for {confused_with}."
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Is the confidence trustworthy?")
        st.altair_chart(charts.reliability_diagram(pd.DataFrame(test["reliability"]), dark), width="stretch")
        st.caption("Points below the diagonal mean the model is over-confident at that confidence level.")
    with right:
        st.markdown("#### Confidence of right and wrong answers")
        histogram = test["confidence_histogram"]
        edges = histogram["edges"]
        rows = [
            {"bin_start": edges[i], "bin_end": edges[i + 1], "outcome": outcome, "count": counts[i]}
            for outcome, counts in (("Correct", histogram["correct"]), ("Incorrect", histogram["incorrect"]))
            for i in range(len(edges) - 1)
        ]
        st.altair_chart(charts.confidence_histogram(pd.DataFrame(rows), dark), width="stretch")
        st.caption("Wrong answers cluster at low confidence, which is why the app flags predictions under 45%.")

    st.markdown("#### What each improvement is worth")
    left, right = st.columns(2, gap="large")
    with left:
        ablation = pd.DataFrame(report["ablation"])
        st.altair_chart(
            charts.single_bars(ablation, "variant", "accuracy", dark, ".1%", highlight="+ mirror test-time augmentation",
                               sort=list(ablation["variant"]), title="PrivateTest accuracy"),
            width="stretch",
        )
        st.caption("Preprocessing ablation on PrivateTest; the highlighted bar is what the app uses.")
    with right:
        framing = report["framing"]
        st.altair_chart(
            charts.framing_chart(pd.DataFrame(framing["points"]), framing["deepface_letterbox"]["tta"], dark), width="stretch"
        )
        st.caption(
            f"Faces detected by YuNet in 4× upscaled FER faces ({framing['detected']:,} of {framing['total']:,}). "
            f"Square crops near the box size match FER-2013's framing; the app uses {report['crop_scale']}×."
        )

    st.markdown("#### Face detectors")
    detectors = pd.DataFrame(report["detectors"])
    left, middle, right = st.columns(3, gap="large")
    with left:
        st.altair_chart(charts.single_bars(detectors, "detector", "detection_rate", dark, ".0%", title="Faces found"), width="stretch")
    with middle:
        st.altair_chart(charts.single_bars(detectors, "detector", "ms_per_image", dark, ".0f", title="Milliseconds per image"), width="stretch")
    with right:
        st.altair_chart(charts.single_bars(detectors, "detector", "end_to_end_accuracy", dark, ".0%", title="End-to-end accuracy"), width="stretch")
    st.caption(
        "Measured on upscaled FER-2013 PrivateTest faces (RetinaFace on fewer samples because it is slow on CPU). "
        "End-to-end accuracy counts a missed face as wrong."
    )

    st.markdown("#### Inference speed")
    speed = pd.DataFrame(report["speed_ms"])
    st.altair_chart(charts.single_bars(speed, "setup", "ms", dark, ".1f", sort=list(speed["setup"]), title="Milliseconds per call"), width="stretch")
    versions = report["versions"]
    st.caption(f"Generated {report['generated']} · DeepFace {versions['deepface']} · TensorFlow {versions['tensorflow']} · OpenCV {versions['opencv']}.")


def _dataset_tab(report: dict | None) -> None:
    dark = is_dark()
    if report:
        dataset = report["dataset"]
    else:  # published FER-2013 counts
        dataset = {"train": dict(zip(EMOTIONS, (3995, 436, 4097, 7215, 4830, 3171, 4965)))}
    train = dataset["train"]
    total = sum(train.values())
    counts = pd.DataFrame(
        {"Emotion": [e.title() for e in DISPLAY_ORDER], "Faces": [train[e] for e in DISPLAY_ORDER], "Share": [train[e] / total for e in DISPLAY_ORDER]}
    )
    cards = st.columns(4)
    cards[0].metric("Images", f"{sum(sum(split.values()) for split in dataset.values()):,}", border=True)
    cards[1].metric("Training faces", f"{total:,}", border=True)
    cards[2].metric("Resolution", "48×48 gray", border=True)
    cards[3].metric("Largest ÷ smallest class", f"{max(train.values()) / min(train.values()):.1f}×", border=True, help="Happy vs disgust.")
    left, right = st.columns((1.4, 1), gap="large")
    with left:
        st.markdown("**FER-2013 training faces per emotion**")
        st.altair_chart(charts.class_distribution(counts, dark), width="stretch")
    with right:
        st.markdown("**All splits**")
        names = {"train": "Training", "publicTest": "PublicTest", "privateTest": "PrivateTest"}
        table = pd.DataFrame({names[split]: [values[e] for e in DISPLAY_ORDER] for split, values in dataset.items()}, index=[e.title() for e in DISPLAY_ORDER])
        table.loc["Total"] = table.sum()
        st.dataframe(table, width="stretch")
        st.markdown(
            "FER-2013 (Goodfellow et al., 2013) was collected with Google image search and labelled into seven expressions. "
            "**Disgust** is rare (about 1.5% of faces), so the model sees few examples and recalls it poorly. "
            "Labels are noisy too: human accuracy on FER-2013 is estimated at only about 65%."
        )
    st.markdown(legend_html(), unsafe_allow_html=True)


def insights_page() -> None:
    page_header(
        "Model",
        "Model insights",
        "Inside DeepFace's emotion CNN: its architecture and learned weights, a from-scratch training run, "
        "and how well it performs on FER-2013.",
    )
    sidebar_footer()
    if not ensure_model():
        return
    report = load_report("model_report.json")
    history = load_report("training_history.json")
    layers, samples, statistics, filters, summary, total_params = _architecture()
    overview, architecture, training, evaluation, dataset = st.tabs(
        [":material/dashboard: Overview", ":material/account_tree: Architecture", ":material/trending_up: Training",
         ":material/fact_check: Evaluation", ":material/dataset: Dataset"]
    )
    with overview:
        _overview(report, history, total_params)
    with architecture:
        _architecture_tab(layers, samples, statistics, filters, summary)
    with training:
        _training_tab(history)
    with evaluation:
        _evaluation_tab(report)
    with dataset:
        _dataset_tab(report)
