"""Altair chart builders shared by the app's pages.

Colour rules: emotions always use their own validated colour (identity
follows the emotion, never its rank), single-series charts use one hue, and
magnitudes use a one-hue blue ramp. Each chart has a light and dark variant.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .config import DISPLAY_ORDER, EMOTION_STYLE, EMOTIONS

SERIES = {
    "light": ("#2A78D6", "#EB6834", "#1BAF7A"),
    "dark": ("#3987E5", "#D95926", "#199E70"),
}
INK = {"light": ("#16161D", "#52514E"), "dark": ("#ECECF1", "#C3C2B7")}
SURFACE = {"light": "#F7F7FA", "dark": "#0F1016"}
MUTED = {"light": "#B9B9C3", "dark": "#4A4B57"}
RAMP = {"light": ["#EEF4FD", "#86B6EF", "#2A78D6", "#0D366B"], "dark": ["#172235", "#1C5CAB", "#5598E7", "#B7D3F6"]}


def _mode(dark: bool) -> str:
    return "dark" if dark else "light"


def emotion_scale(dark: bool = False) -> alt.Scale:
    key = "dark" if dark else "color"
    return alt.Scale(
        domain=[emotion.title() for emotion in DISPLAY_ORDER],
        range=[EMOTION_STYLE[emotion][key] for emotion in DISPLAY_ORDER],
    )


def emotion_bars(scores: dict[str, float], dark: bool = False, height: int = 210) -> alt.Chart:
    """Horizontal probability bars for one face, largest first."""
    data = pd.DataFrame(
        {"Emotion": [emotion.title() for emotion in EMOTIONS], "Probability": [scores[emotion] for emotion in EMOTIONS]}
    )
    base = alt.Chart(data).encode(
        y=alt.Y("Emotion:N", sort="-x", title=None, axis=alt.Axis(ticks=False, domain=False, labelPadding=6, labelOverlap=False)),
        x=alt.X("Probability:Q", title=None, scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(format=".0f", tickCount=5)),
        tooltip=["Emotion:N", alt.Tooltip("Probability:Q", format=".1f", title="Probability (%)")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.72}).encode(
        color=alt.Color("Emotion:N", scale=emotion_scale(dark), legend=None)
    )
    labels = base.transform_filter("datum.Probability >= 3").mark_text(
        align="left", dx=5, fontSize=12, color=INK[_mode(dark)][1]
    ).encode(text=alt.Text("Probability:Q", format=".0f"))
    return (bars + labels).properties(height=height)


def emotion_counts(counts: dict[str, int], dark: bool = False, title: str = "Faces") -> alt.Chart:
    data = pd.DataFrame({"Emotion": [emotion.title() for emotion in DISPLAY_ORDER], title: [counts.get(e, 0) for e in DISPLAY_ORDER]})
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.72})
        .encode(
            y=alt.Y("Emotion:N", sort=[emotion.title() for emotion in DISPLAY_ORDER], title=None, axis=alt.Axis(ticks=False, domain=False)),
            x=alt.X(f"{title}:Q", axis=alt.Axis(tickMinStep=1, format="d")),
            color=alt.Color("Emotion:N", scale=emotion_scale(dark), legend=None),
            tooltip=["Emotion:N", f"{title}:Q"],
        )
        .properties(height=220)
    )


def emotion_timeline(history: list[tuple[float, np.ndarray | None]], dark: bool = False, window: float = 60) -> alt.Chart | None:
    """Stacked area of the main face's emotion mix over the last ``window`` seconds."""
    if not history:
        return None
    latest = history[-1][0]
    points = [(t, scores) for t, scores in history if t >= latest - window and scores is not None]
    if len(points) < 2:
        return None
    # Thin to roughly 4 points per second so the chart stays light.
    times = np.array([t for t, _ in points])
    keep = np.unique(np.searchsorted(times, np.arange(times[0], times[-1] + 0.25, 0.25)).clip(0, len(points) - 1))
    rows = []
    for index in keep:
        t, scores = points[index]
        for emotion_index, emotion in enumerate(EMOTIONS):
            rows.append({"Seconds ago": round(t - latest, 2), "Emotion": emotion.title(), "Share": float(scores[emotion_index])})
    data = pd.DataFrame(rows)
    order = {emotion.title(): index for index, emotion in enumerate(DISPLAY_ORDER)}
    data["order"] = data["Emotion"].map(order)
    return (
        alt.Chart(data)
        .mark_area(interpolate="monotone", stroke=SURFACE[_mode(dark)], strokeWidth=1)
        .encode(
            x=alt.X("Seconds ago:Q", scale=alt.Scale(domain=[-window, 0]), axis=alt.Axis(tickCount=6)),
            y=alt.Y("Share:Q", stack="normalize", axis=alt.Axis(format="%", tickCount=4), title=None),
            color=alt.Color(
                "Emotion:N",
                scale=emotion_scale(dark),
                legend=alt.Legend(orient="bottom", title=None, columns=4, symbolType="square"),
            ),
            order=alt.Order("order:Q"),
            tooltip=["Emotion:N", alt.Tooltip("Share:Q", format=".1f", title="Probability (%)"), "Seconds ago:Q"],
        )
        .properties(height=230)
    )


def time_share(share: dict[str, float], dark: bool = False) -> alt.Chart:
    data = pd.DataFrame({"Emotion": [e.title() for e in DISPLAY_ORDER], "Share": [share.get(e, 0.0) for e in DISPLAY_ORDER]})
    base = alt.Chart(data).encode(
        y=alt.Y("Emotion:N", sort=[e.title() for e in DISPLAY_ORDER], title=None, axis=alt.Axis(ticks=False, domain=False)),
        x=alt.X("Share:Q", title="Share of session time", axis=alt.Axis(format="%", tickCount=4), scale=alt.Scale(domain=[0, 1])),
        tooltip=["Emotion:N", alt.Tooltip("Share:Q", format=".0%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.72}).encode(color=alt.Color("Emotion:N", scale=emotion_scale(dark), legend=None))
    labels = base.transform_filter("datum.Share >= 0.02").mark_text(align="left", dx=5, fontSize=12, color=INK[_mode(dark)][1]).encode(
        text=alt.Text("Share:Q", format=".0%")
    )
    return (bars + labels).properties(height=210)


def single_bars(data: pd.DataFrame, category: str, value: str, dark: bool = False, value_format: str = ",.0f",
                highlight: str | None = None, sort: str | list[str] | None = "-x", height: int | None = None,
                title: str | None = None) -> alt.Chart:
    """One-series horizontal bars; ``highlight`` names a bar to emphasise."""
    color = SERIES[_mode(dark)][0]
    encoding_color = (
        alt.condition(alt.datum[category] == highlight, alt.value(color), alt.value(MUTED[_mode(dark)]))
        if highlight is not None
        else alt.value(color)
    )
    base = alt.Chart(data).encode(
        y=alt.Y(f"{category}:N", sort=sort, title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=260)),
        x=alt.X(f"{value}:Q", title=title or value, axis=alt.Axis(format=value_format.lstrip(","), tickCount=5)),
        tooltip=[f"{category}:N", alt.Tooltip(f"{value}:Q", format=value_format)],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(color=encoding_color)
    labels = base.mark_text(align="left", dx=5, fontSize=12, color=INK[_mode(dark)][1]).encode(
        text=alt.Text(f"{value}:Q", format=value_format)
    )
    return (bars + labels).properties(height=height or max(140, 30 * len(data)))


def weight_histograms(samples: pd.DataFrame, dark: bool = False) -> alt.Chart:
    return (
        alt.Chart(samples)
        .mark_bar(color=SERIES[_mode(dark)][0], cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("weight:Q", bin=alt.Bin(maxbins=40), title="Weight value"),
            y=alt.Y("count():Q", title=None, axis=alt.Axis(tickCount=3)),
            tooltip=[alt.Tooltip("count():Q", title="Weights in bin")],
        )
        .properties(width=210, height=120)
        .facet(facet=alt.Facet("layer:N", title=None, sort=list(dict.fromkeys(samples["layer"]))), columns=4)
        .resolve_scale(x="independent", y="independent")
    )


def training_curve(epochs: pd.DataFrame, metric: str, best_epoch: int | None, dark: bool = False) -> alt.Chart:
    """Train vs validation curve for one metric (loss or accuracy)."""
    label = "Accuracy" if metric == "accuracy" else "Loss"
    data = pd.concat(
        [
            pd.DataFrame({"Epoch": epochs["epoch"], label: epochs[metric], "Split": "Training"}),
            pd.DataFrame({"Epoch": epochs["epoch"], label: epochs[f"val_{metric}"], "Split": "Validation (PublicTest)"}),
        ]
    )
    colors = SERIES[_mode(dark)]
    scale = alt.Scale(domain=["Training", "Validation (PublicTest)"], range=[colors[0], colors[1]])
    hover = alt.selection_point(fields=["Epoch"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(data).encode(x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1, format="d")))
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y(f"{label}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=".0%" if metric == "accuracy" else ".2f")),
        color=alt.Color("Split:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
    )
    points = base.mark_point(filled=True, size=60).encode(
        y=f"{label}:Q",
        color=alt.Color("Split:N", scale=scale, legend=None),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=["Epoch:Q", "Split:N", alt.Tooltip(f"{label}:Q", format=".2%" if metric == "accuracy" else ".3f")],
    ).add_params(hover)
    rule = base.mark_rule(color=MUTED[_mode(dark)]).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0))).transform_filter(hover)
    layers = [lines, points, rule]
    if best_epoch:
        best = alt.Chart(pd.DataFrame({"Epoch": [best_epoch], "label": [f"best val. epoch {best_epoch}"]}))
        layers.append(best.mark_rule(color=INK[_mode(dark)][1], strokeWidth=1).encode(x="Epoch:Q"))
        layers.append(
            best.mark_text(align="left", dx=4, dy=-4, baseline="top", fontSize=11, color=INK[_mode(dark)][1]).encode(
                x="Epoch:Q", y=alt.value(0), text="label:N"
            )
        )
    return alt.layer(*layers).properties(height=260)


def grouped_bars(data: pd.DataFrame, category: str, series: str, value: str, series_order: list[str],
                 dark: bool = False, value_format: str = ".0%", category_sort: list[str] | None = None,
                 domain: tuple[float, float] | None = (0, 1), height: int = 260) -> alt.Chart:
    """Vertical grouped bars for up to three series (fixed colour order)."""
    colors = SERIES[_mode(dark)][: len(series_order)]
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{category}:N", sort=category_sort, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset(f"{series}:N", sort=series_order),
            y=alt.Y(f"{value}:Q", axis=alt.Axis(format=value_format), scale=alt.Scale(domain=list(domain)) if domain else alt.Undefined),
            color=alt.Color(f"{series}:N", scale=alt.Scale(domain=series_order, range=list(colors)),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[f"{category}:N", f"{series}:N", alt.Tooltip(f"{value}:Q", format=".2%" if value_format == ".0%" else value_format)],
        )
        .properties(height=height)
    )


def confusion_heatmap(matrix: np.ndarray, labels: list[str], dark: bool = False) -> alt.Chart:
    """Row-normalised confusion matrix: each row is one true emotion."""
    matrix = np.asarray(matrix, dtype=float)
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    rows = [
        {"True": labels[i], "Predicted": labels[j], "Share": normalised[i, j], "Count": int(matrix[i, j])}
        for i in range(len(labels))
        for j in range(len(labels))
    ]
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=-35, labelOverlap=False, orient="top", title="Predicted")),
        y=alt.Y("True:N", sort=labels, title="True label"),
    )
    cells = base.mark_rect(stroke=SURFACE[_mode(dark)], strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("Share:Q", scale=alt.Scale(domain=[0, 1], range=RAMP[_mode(dark)]),
                        legend=alt.Legend(format="%", title="Row share", orient="right")),
        tooltip=["True:N", "Predicted:N", alt.Tooltip("Share:Q", format=".1%"), "Count:Q"],
    )
    light_ink, dark_ink = "#FFFFFF", "#16161D"
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text("Share:Q", format=".0%"),
        color=alt.condition(
            alt.datum.Share > 0.45,
            alt.value(dark_ink if dark else light_ink),
            alt.value(INK[_mode(dark)][0]),
        ),
    )
    return (cells + text).properties(height=380)


def reliability_diagram(bins: pd.DataFrame, dark: bool = False) -> alt.Chart:
    """Accuracy within each confidence bin against the ideal diagonal."""
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED[_mode(dark)], strokeWidth=1).encode(
        x=alt.X("x:Q", title="Model confidence", axis=alt.Axis(format="%")), y=alt.Y("y:Q", title="Actual accuracy", axis=alt.Axis(format="%"))
    )
    data = bins[bins["count"] > 0]
    color = SERIES[_mode(dark)][0]
    line = alt.Chart(data).mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True, color=color)).encode(
        x="confidence:Q",
        y="accuracy:Q",
        tooltip=[alt.Tooltip("confidence:Q", format=".0%", title="Mean confidence"), alt.Tooltip("accuracy:Q", format=".1%"), "count:Q"],
    )
    return (diagonal + line).properties(height=300)


def confidence_histogram(data: pd.DataFrame, dark: bool = False) -> alt.Chart:
    colors = SERIES[_mode(dark)]
    order = ["Correct", "Incorrect"]
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, stroke=SURFACE[_mode(dark)], strokeWidth=1)
        .encode(
            x=alt.X("bin_start:Q", bin="binned", title="Confidence of the predicted emotion", axis=alt.Axis(format="%")),
            x2="bin_end:Q",
            y=alt.Y("count:Q", stack=True, title="Faces"),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=order, range=[colors[0], colors[1]]), legend=alt.Legend(orient="bottom", title=None)),
            order=alt.Order("outcome:N"),
            tooltip=["outcome:N", alt.Tooltip("bin_start:Q", format=".0%", title="From"), alt.Tooltip("bin_end:Q", format=".0%", title="To"), "count:Q"],
        )
        .properties(height=300)
    )


def framing_chart(data: pd.DataFrame, reference: float, dark: bool = False) -> alt.Chart:
    colors = SERIES[_mode(dark)]
    order = ["Single pass", "With mirror TTA"]
    lines = alt.Chart(data).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=alt.X("scale:Q", title="Square crop side ÷ detector box side", scale=alt.Scale(zero=False)),
        y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        color=alt.Color("variant:N", scale=alt.Scale(domain=order, range=[colors[0], colors[1]]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["variant:N", "scale:Q", alt.Tooltip("accuracy:Q", format=".2%")],
    )
    rule_data = pd.DataFrame({"accuracy": [reference], "label": ["DeepFace letterbox crop (with TTA)"]})
    rule = alt.Chart(rule_data).mark_rule(color=INK[_mode(dark)][1], strokeWidth=1).encode(y="accuracy:Q")
    text = alt.Chart(rule_data).mark_text(align="right", dy=-6, fontSize=11, color=INK[_mode(dark)][1]).encode(
        y="accuracy:Q", x=alt.value(560), text="label:N"
    )
    return (lines + rule + text).properties(height=280)


def class_distribution(counts: pd.DataFrame, dark: bool = False) -> alt.Chart:
    """Training-split faces per emotion, coloured by emotion."""
    base = alt.Chart(counts).encode(
        x=alt.X("Emotion:N", sort=[e.title() for e in DISPLAY_ORDER], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Faces:Q", title="Training faces"),
        tooltip=["Emotion:N", alt.Tooltip("Faces:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(color=alt.Color("Emotion:N", scale=emotion_scale(dark), legend=None))
    labels = base.mark_text(dy=-6, fontSize=12, color=INK[_mode(dark)][1]).encode(text=alt.Text("Faces:Q", format=","))
    return (bars + labels).properties(height=300)
