"""
FER Studio - Condition-Aware Facial Emotion Recognition
Streamlit dashboard (research prototype).

Two independent outputs are produced by this app:
  * PDF 1 - the browser's Print / Save as PDF of the dashboard (print CSS).
  * PDF 2 - the dedicated image-analysis report built with ReportLab.

The model, checkpoint, preprocessing assumptions, class order and the
experimental CSV data are unchanged. Only presentation, reporting and
usability layers are different.
"""

import html
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Must be the first Streamlit command, and must be called only once.
st.set_page_config(
    page_title="FER Studio | Condition-Aware FER",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Dataset" / "fer_v8_best.pth"
FACE_MODEL_PATH = BASE_DIR / "blaze_face_short_range.tflite"

BASELINE_RESULTS_PATH = BASE_DIR / "Dataset" / "v8_baseline_condition_results.csv"
ADAPTIVE_RESULTS_PATH = BASE_DIR / "Dataset" / "v8_adaptive_processing_results.csv"

EMOTIONS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Heuristic thresholds used by assess_condition() (unchanged).
LOW_LIGHT_THRESHOLD = 85
BLUR_THRESHOLD = 450
NOISE_THRESHOLD = 16

BASELINE_COLOR = "#C7A4A8"
ADAPTIVE_COLOR = "#773B4A"


# Recent Streamlit versions replace use_container_width with width="stretch".
def _version_tuple(version_text):
    parts = []
    for piece in version_text.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


USE_WIDTH_ARG = _version_tuple(st.__version__) >= (1, 50)


# ============================================================

# ============================================================
# 2. V8 RESNET18 FER MODEL
# ============================================================

V8_CLASS_NAMES = [
    "anger",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

EMOTIONS = [
    "Anger",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]


def build_v8_model():
    model = tv_models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(512, 6),
    )

    return model


# ============================================================
# 3. LOAD TRAINED V8 MODEL
# ============================================================

@st.cache_resource
def load_fer_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V8 FER checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected V8 checkpoint format.")

    if "model_state_dict" not in checkpoint:
        raise ValueError("V8 model_state_dict is missing.")

    stored_classes = list(checkpoint.get("class_names", []))

    if stored_classes != V8_CLASS_NAMES:
        raise ValueError(
            f"V8 class order mismatch: {stored_classes}"
        )

    model = build_v8_model().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


# 4. LOAD MEDIAPIPE FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face detector model not found: {FACE_MODEL_PATH}"
        )

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH)
        ),
        min_detection_confidence=0.5,
    )

    return vision.FaceDetector.create_from_options(options)


# ============================================================
# 5. FACE DETECTION AND CROPPING  (unchanged logic)
# ============================================================

def detect_and_crop_face(image):

    rgb = np.asarray(image.convert("RGB"))

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )

    detector = load_face_detector()
    result = detector.detect(mp_image)

    if not result.detections:
        return (
            image.convert("L"),
            "No face detected; using full image",
        )

    detection = max(
        result.detections,
        key=lambda d: (
            d.bounding_box.width * d.bounding_box.height
        ),
    )

    bbox = detection.bounding_box

    x, y = bbox.origin_x, bbox.origin_y
    w, h = bbox.width, bbox.height

    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(rgb.shape[1], x + w + margin_x)
    y2 = min(rgb.shape[0], y + h + margin_y)

    if x2 <= x1 or y2 <= y1:
        return (
            image.convert("L"),
            "Invalid face crop; using full image",
        )

    face_crop = Image.fromarray(
        rgb[y1:y2, x1:x2]
    ).convert("L")

    return face_crop, "Face detected and cropped successfully"


# ============================================================
# 6. IMAGE CONDITION ASSESSMENT  (unchanged logic)
# ============================================================


def assess_condition(image):
    """
    Estimate brightness, sharpness and noise using heuristics.
    Partial occlusion is not reliably detected.
    """

    # Keep existing brightness and noise calculations unchanged.
    gray = np.asarray(
        ImageOps.grayscale(image).resize((48, 48)),
        dtype=np.float32,
    )

    brightness = float(gray.mean())

    # Calculate sharpness at higher resolution.
    sharp_gray = np.asarray(
        ImageOps.grayscale(image).resize((112, 112)),
        dtype=np.float32,
    )

    laplacian = (
        -4 * sharp_gray
        + np.roll(sharp_gray, 1, axis=0)
        + np.roll(sharp_gray, -1, axis=0)
        + np.roll(sharp_gray, 1, axis=1)
        + np.roll(sharp_gray, -1, axis=1)
    )

    sharpness = float(laplacian.var())

    smooth_image = Image.fromarray(
        np.uint8(np.clip(gray, 0, 255))
    ).filter(ImageFilter.GaussianBlur(radius=1))

    smooth = np.asarray(smooth_image, dtype=np.float32)
    noise = float(np.std(gray - smooth))

    low_light = brightness < LOW_LIGHT_THRESHOLD
    blurry = sharpness < BLUR_THRESHOLD
    noisy = noise > NOISE_THRESHOLD

    conditions = []

    if low_light:
        conditions.append("Low illumination")
    if blurry:
        conditions.append("Possible blur")
    if noisy:
        conditions.append("Possible noise")

    if not conditions:
        conditions = ["Normal / no strong quality warning"]

    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "noise": noise,
        "low_light": low_light,
        "blur": blurry,
        "noise_flag": noisy,
        "condition": " + ".join(conditions),
    }



# ============================================================
# 7. ADAPTIVE IMAGE PREPROCESSING  (unchanged logic)
# ============================================================

def adaptive_process(image, quality):

    processed = image.convert("L")
    strategies = []

    if quality["low_light"]:
        processed = ImageEnhance.Brightness(
            processed
        ).enhance(1.7)

        strategies.append("Brightness enhancement")

    if quality["noise_flag"]:
        processed = processed.filter(
            ImageFilter.MedianFilter(size=3)
        )

        strategies.append("Median-filter denoising")

    if quality["blur"]:
        processed = processed.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=150,
                threshold=3,
            )
        )

        strategies.append("Sharpening")

    if not strategies:
        strategies.append("Standard preprocessing")

    return processed, strategies


# ============================================================
# 8. IMAGE PREPARATION AND PREDICTION  (unchanged logic)
# ============================================================


# Exact V8 validation preprocessing.
V8_TRANSFORM = tv_transforms.Compose([
    tv_transforms.Resize((112, 112)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def image_to_tensor(image):

    # V8 training used grayscale converted to RGB.
    image = image.convert("L").convert("RGB")

    tensor = V8_TRANSFORM(image)
    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)



def predict_emotion(model, image):

    tensor = image_to_tensor(image)

    if DEVICE.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(tensor),
            dim=1,
        )[0].cpu().numpy()

    if DEVICE.type == "mps":
        torch.mps.synchronize()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    sorted_indices = np.argsort(probabilities)[::-1]

    index = int(sorted_indices[0])

    top_three = [
        {
            "emotion": EMOTIONS[int(class_index)],
            "score": float(probabilities[int(class_index)]),
        }
        for class_index in sorted_indices[:3]
    ]

    top_score = float(probabilities[sorted_indices[0]])
    second_score = float(probabilities[sorted_indices[1]])

    score_gap = top_score - second_score

    uncertain = (
        top_score < 0.35
        or score_gap < 0.08
    )

    return {
        "emotion": EMOTIONS[index],
        "confidence": top_score,
        "probabilities": probabilities,
        "top_three": top_three,
        "score_gap": score_gap,
        "uncertain": uncertain,
        "time_ms": elapsed_ms,
    }


# ============================================================
# 9. RENDERING HELPERS
# ============================================================

def esc(value):
    """HTML-escape any dynamic text before placing it in markup."""
    return html.escape(str(value), quote=True)


def render_html(markup):
    """
    Render one complete HTML block through st.markdown.

    Markdown treats indented lines as code blocks, and a blank line ends
    an HTML block. Stripping indentation and blank lines guarantees that
    the whole block is parsed as one raw-HTML element.
    """
    cleaned = "\n".join(
        line.strip()
        for line in markup.strip().splitlines()
        if line.strip()
    )

    st.markdown(cleaned, unsafe_allow_html=True)


def show_plotly(fig):
    config = {"displayModeBar": False}

    if USE_WIDTH_ARG:
        try:
            st.plotly_chart(fig, width="stretch", config=config)
            return
        except Exception:
            pass

    st.plotly_chart(fig, use_container_width=True, config=config)


def show_image(image, caption=None):

    if USE_WIDTH_ARG:
        try:
            st.image(image, caption=caption, width="stretch")
            return
        except Exception:
            pass

    st.image(image, caption=caption, use_container_width=True)


def section_header(anchor, label, title, lead=None, first=False):
    """
    Section label, native Streamlit heading (with a real anchor id) and an
    optional lead paragraph. The anchor= argument of st.header() is what
    the sidebar's markdown links (#anchor) point to.
    """

    css_class = "fer-sechead first" if first else "fer-sechead"

    render_html(
        f'<div class="{css_class}">'
        f'<div class="fer-section">{esc(label)}</div>'
        f'</div>'
    )

    st.header(title, anchor=anchor, divider=False)

    if lead:
        render_html(f'<div class="fer-lead">{esc(lead)}</div>')


def sub_header(text):
    render_html(f'<div class="fer-h3">{esc(text)}</div>')


def minor_header(text):
    render_html(f'<div class="fer-h4">{esc(text)}</div>')


def render_table(dataframe, numeric_columns=(), widths=None):
    """
    Render a DataFrame of display strings as a styled, print-safe table.

    widths: optional list of column widths in percent. When given, the
    table uses a fixed layout so it always fits its container (no
    horizontal scrolling on screen or in print).
    """

    colgroup = ""
    table_class = "fer-table"

    if widths:
        colgroup = "<colgroup>" + "".join(
            f'<col style="width:{width}%;">' for width in widths
        ) + "</colgroup>"
        table_class = "fer-table fixed"

    header_cells = "".join(
        f"<th class=\"{'num' if col in numeric_columns else ''}\">"
        f"{esc(col)}</th>"
        for col in dataframe.columns
    )

    body_rows = []

    for _, row in dataframe.iterrows():
        cells = "".join(
            f"<td class=\"{'num' if col in numeric_columns else ''}\">"
            f"{esc(row[col])}</td>"
            for col in dataframe.columns
        )
        body_rows.append(f"<tr>{cells}</tr>")

    render_html(
        '<div class="fer-table-wrap">'
        f'<table class="{table_class}">'
        f'{colgroup}'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
        '</div>'
    )


def render_print_bars(categories, series, max_value, value_format,
                      scale_note=""):
    """
    Print-only horizontal bar rendition of a chart.

    Plotly draws fixed-pixel SVG that cannot re-flow to the printed page
    width, so for the browser's Print / Save as PDF the interactive chart
    is hidden and this pure HTML/CSS version (which scales with the page)
    is shown instead. On screen this block is hidden.

    series: list of dicts with keys name, values, color.
    """

    legend = "".join(
        '<span class="fer-legend-item">'
        f'<span class="fer-swatch" style="background:{esc(item["color"])};">'
        f'</span>{esc(item["name"])}</span>'
        for item in series
    )

    if scale_note:
        legend += f'<span class="fer-legend-note">{esc(scale_note)}</span>'

    groups = []

    for index, category in enumerate(categories):

        rows = []

        for item in series:
            value = float(item["values"][index])

            if max_value > 0:
                width = max(0.0, min(100.0, value / max_value * 100.0))
            else:
                width = 0.0

            rows.append(
                '<div class="fer-hbar-row">'
                '<div class="fer-hbar-track">'
                f'<div class="fer-hbar-fill" style="width:{width:.1f}%;'
                f'background:{esc(item["color"])};"></div>'
                '</div>'
                f'<div class="fer-hbar-val">{esc(value_format(value))}</div>'
                '</div>'
            )

        groups.append(
            '<div class="fer-hbar-group">'
            f'<div class="fer-hbar-label">{esc(category)}</div>'
            f'<div class="fer-hbar-rows">{"".join(rows)}</div>'
            '</div>'
        )

    render_html(
        '<div class="fer-hbars print-only">'
        f'<div class="fer-hbar-legend">{legend}</div>'
        f'{"".join(groups)}'
        '</div>'
    )


def prediction_card_html(result, title, description):
    """Return the HTML for one prediction card (not rendered here)."""

    top_rows = "".join(
        f"<tr><td>{esc(item['emotion'])}</td>"
        f"<td class=\"num\">{item['score']:.1%}</td></tr>"
        for item in result["top_three"]
    )

    if result["uncertain"]:
        flag = (
            '<div class="fer-flag warn">'
            "Ambiguous prediction: the top score is low or close to the "
            "second-highest score."
            "</div>"
        )
    else:
        flag = (
            '<div class="fer-flag ok">'
            "The top score passes the current display rule. This does "
            "not guarantee prediction correctness."
            "</div>"
        )

    bar_width = max(0.0, min(100.0, result["confidence"] * 100))

    return (
        '<div class="fer-card">'
        f'<div class="fer-section">{esc(title)}</div>'
        f'<div class="fer-pred-emotion">{esc(result["emotion"])}</div>'
        f'<div class="fer-muted">{esc(description)}</div>'
        '<div class="fer-kicker">Top-class softmax score</div>'
        f'<div class="fer-pred-score">{result["confidence"]:.1%}</div>'
        '<div class="fer-bar">'
        f'<div class="fer-bar-fill" style="width:{bar_width:.1f}%;"></div>'
        '</div>'
        f'<div class="fer-muted small">Inference time: '
        f'{result["time_ms"]:.1f} ms</div>'
        '<div class="fer-kicker">Top 3 predictions</div>'
        '<table class="fer-mini-table">'
        '<thead><tr><th>Expression</th>'
        '<th class="num">Model score</th></tr></thead>'
        f'<tbody>{top_rows}</tbody>'
        '</table>'
        f'{flag}'
        '</div>'
    )


# ============================================================
# 10. EXPERIMENTAL RESULTS  (data and numbers unchanged)
# ============================================================

CONDITION_CHART_HEIGHT = 410


def _style_bar_figure(fig, y_title, y_max):
    """Shared styling so both condition charts look identical."""

    fig.update_layout(
        height=CONDITION_CHART_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#30272A",
            size=12,
        ),
        xaxis=dict(
            showgrid=False,
            title=dict(text="", font=dict(color="#30272A", size=12)),
            automargin=True,
            tickfont=dict(family="DM Sans, sans-serif", color="#30272A", size=12),
            color="#30272A",
            linecolor="#CDBFC0",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EAE1DC",
            title=dict(text=y_title, font=dict(family="DM Sans, sans-serif", color="#30272A", size=12)),
            automargin=True,
            tickfont=dict(family="DM Sans, sans-serif", color="#30272A", size=12),
            color="#30272A",
            linecolor="#CDBFC0",
            range=[0, y_max * 1.22],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            title=dict(font=dict(family="DM Sans, sans-serif", color="#30272A")),
            font=dict(family="DM Sans, sans-serif", color="#30272A", size=12),
        ),
        margin=dict(l=30, r=20, t=70, b=30),
        bargap=0.22,
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(family="DM Sans, sans-serif", color="#30272A", size=11),
        cliponaxis=False,
    )

    return fig


def _signed(value):
    return f"{value:+.2f}"


def _plain(value):
    return f"{value:.2f}"


def check_result_units(adaptive_results):
    """
    Read-only consistency check; it never changes a number.

    accuracy_change_pp should equal (adaptive - baseline) accuracy in
    percentage points, and macro_f1_change should equal the difference of
    the macro-F1 columns. If the CSV units differ from these assumptions,
    show a warning instead of silently mis-scaling.
    """

    try:
        accuracy_gap = (
            (
                adaptive_results["adaptive_accuracy"]
                - adaptive_results["baseline_accuracy"]
            ) * 100
            - adaptive_results["accuracy_change_pp"]
        ).abs().max()

        f1_gap = (
            adaptive_results["adaptive_macro_f1"]
            - adaptive_results["baseline_macro_f1"]
            - adaptive_results["macro_f1_change"]
        ).abs().max()
    except Exception:
        return

    messages = []

    if pd.notna(accuracy_gap) and accuracy_gap > 0.05:
        messages.append(
            "accuracy_change_pp does not match "
            "(adaptive_accuracy - baseline_accuracy) x 100"
        )

    if pd.notna(f1_gap) and f1_gap > 0.0005:
        messages.append(
            "macro_f1_change does not match "
            "(adaptive_macro_f1 - baseline_macro_f1)"
        )

    if messages:
        st.warning(
            "Unit check: " + "; ".join(messages) + ". The CSV values may "
            "use different units from the ones assumed by this "
            "dashboard, so please verify them before relying on the "
            "displayed percentages."
        )


def render_experimental_results():

    render_html(
        '<div class="fer-note">'
        "<b>How to read these results.</b> The live condition assessment "
        "on this page is heuristic-based: it measures brightness, "
        "sharpness and noise of the uploaded image. The V8 experiment "
        "applies the calibrated heuristic condition assessment to each "
        "test image before selecting adaptive preprocessing. Thresholds "
        "were selected using validation data and held fixed for "
        "independent test evaluation."
        "</div>"
    )

    if not BASELINE_RESULTS_PATH.exists():
        st.warning(
            f"Baseline results file not found: {BASELINE_RESULTS_PATH}"
        )
        return

    if not ADAPTIVE_RESULTS_PATH.exists():
        st.warning(
            f"Adaptive results file not found: {ADAPTIVE_RESULTS_PATH}"
        )
        return

    try:
        baseline_results = pd.read_csv(BASELINE_RESULTS_PATH)
        adaptive_results = pd.read_csv(ADAPTIVE_RESULTS_PATH)
    except Exception as exc:
        st.error(f"Could not read the experimental CSV files: {exc}")
        return

    required_baseline = {
        "condition", "accuracy", "macro_f1", "images"
    }

    required_adaptive = {
        "condition",
        "baseline_accuracy",
        "adaptive_accuracy",
        "accuracy_change_pp",
        "baseline_macro_f1",
        "adaptive_macro_f1",
        "macro_f1_change",
    }

    if not required_baseline.issubset(baseline_results.columns):
        st.error("The baseline CSV has unexpected columns.")
        return

    if not required_adaptive.issubset(adaptive_results.columns):
        st.error("The adaptive CSV has unexpected columns.")
        return

    check_result_units(adaptive_results)

    results = adaptive_results.copy()

    results["Condition"] = (
        results["condition"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    percent_columns = [
        "baseline_accuracy",
        "adaptive_accuracy",
        "baseline_macro_f1",
        "adaptive_macro_f1",
    ]

    for column in percent_columns:
        results[column] = results[column] * 100

    results["Accuracy Change (pp)"] = results["accuracy_change_pp"]

    results["Macro-F1 Change (pp)"] = (
        results["macro_f1_change"] * 100
    )

    conditions = results["Condition"].tolist()

    # ---------------- Accuracy comparison ----------------
    sub_header("Accuracy comparison")

    accuracy_data = results.melt(
        id_vars="Condition",
        value_vars=["baseline_accuracy", "adaptive_accuracy"],
        var_name="Model",
        value_name="Accuracy",
    )

    accuracy_data["Model"] = accuracy_data["Model"].map({
        "baseline_accuracy": "Baseline",
        "adaptive_accuracy": "Adaptive",
    })

    accuracy_fig = px.bar(
        accuracy_data,
        x="Condition",
        y="Accuracy",
        color="Model",
        barmode="group",
        color_discrete_map={
            "Baseline": BASELINE_COLOR,
            "Adaptive": ADAPTIVE_COLOR,
        },
        text_auto=".2f",
    )

    _style_bar_figure(
        accuracy_fig,
        "Accuracy (%)",
        float(accuracy_data["Accuracy"].max()),
    )

    show_plotly(accuracy_fig)

    render_print_bars(
        conditions,
        [
            {
                "name": "Baseline",
                "values": results["baseline_accuracy"].tolist(),
                "color": BASELINE_COLOR,
            },
            {
                "name": "Adaptive",
                "values": results["adaptive_accuracy"].tolist(),
                "color": ADAPTIVE_COLOR,
            },
        ],
        max_value=100.0,
        value_format=lambda value: f"{value:.2f}%",
        scale_note="Bar scale: 0–100%",
    )

    # ---------------- Macro-F1 comparison ----------------
    sub_header("Macro-F1 comparison")

    f1_data = results.melt(
        id_vars="Condition",
        value_vars=["baseline_macro_f1", "adaptive_macro_f1"],
        var_name="Model",
        value_name="Macro-F1",
    )

    f1_data["Model"] = f1_data["Model"].map({
        "baseline_macro_f1": "Baseline",
        "adaptive_macro_f1": "Adaptive",
    })

    f1_fig = px.bar(
        f1_data,
        x="Condition",
        y="Macro-F1",
        color="Model",
        barmode="group",
        color_discrete_map={
            "Baseline": BASELINE_COLOR,
            "Adaptive": ADAPTIVE_COLOR,
        },
        text_auto=".2f",
    )

    _style_bar_figure(
        f1_fig,
        "Macro-F1 (%)",
        float(f1_data["Macro-F1"].max()),
    )

    show_plotly(f1_fig)

    render_print_bars(
        conditions,
        [
            {
                "name": "Baseline",
                "values": results["baseline_macro_f1"].tolist(),
                "color": BASELINE_COLOR,
            },
            {
                "name": "Adaptive",
                "values": results["adaptive_macro_f1"].tolist(),
                "color": ADAPTIVE_COLOR,
            },
        ],
        max_value=100.0,
        value_format=lambda value: f"{value:.2f}%",
        scale_note="Bar scale: 0–100%",
    )

    # ---------------- Detailed results tables ----------------
    sub_header("Detailed results")

    table = results[
        [
            "Condition",
            "baseline_accuracy",
            "adaptive_accuracy",
            "Accuracy Change (pp)",
            "baseline_macro_f1",
            "adaptive_macro_f1",
            "Macro-F1 Change (pp)",
        ]
    ].copy()

    table = table.rename(
        columns={
            "baseline_accuracy": "Baseline Accuracy (%)",
            "adaptive_accuracy": "Adaptive Accuracy (%)",
            "baseline_macro_f1": "Baseline Macro-F1 (%)",
            "adaptive_macro_f1": "Adaptive Macro-F1 (%)",
        }
    )

    table = table.round(2)

    image_counts = (
        baseline_results
        .drop_duplicates("condition")
        .set_index("condition")["images"]
    )

    table["Test Images"] = (
        results["condition"]
        .map(image_counts)
        .fillna(0)
        .astype(int)
        .values
    )

    # The detailed table is shown as two smaller tables so that the
    # Condition column always stays visible and nothing needs scrolling.
    accuracy_table = pd.DataFrame({
        "Condition": table["Condition"],
        "Test images": table["Test Images"].astype(str),
        "Baseline (%)": table["Baseline Accuracy (%)"].map(_plain),
        "Adaptive (%)": table["Adaptive Accuracy (%)"].map(_plain),
        "Change (pp)": table["Accuracy Change (pp)"].map(_signed),
    })

    f1_table = pd.DataFrame({
        "Condition": table["Condition"],
        "Baseline (%)": table["Baseline Macro-F1 (%)"].map(_plain),
        "Adaptive (%)": table["Adaptive Macro-F1 (%)"].map(_plain),
        "Change (pp)": table["Macro-F1 Change (pp)"].map(_signed),
    })

    minor_header("Accuracy by condition")

    render_table(
        accuracy_table,
        numeric_columns=[
            "Test images", "Baseline (%)", "Adaptive (%)", "Change (pp)"
        ],
        widths=[32, 15, 18, 18, 17],
    )

    minor_header("Macro-F1 by condition")

    render_table(
        f1_table,
        numeric_columns=["Baseline (%)", "Adaptive (%)", "Change (pp)"],
        widths=[40, 20, 20, 20],
    )

    # ---------------- Key observations ----------------
    sub_header("Key observations")

    best_f1_row = results.loc[results["macro_f1_change"].idxmax()]
    best_accuracy_row = results.loc[results["accuracy_change_pp"].idxmax()]
    worst_accuracy_row = results.loc[results["accuracy_change_pp"].idxmin()]

    observation1, observation2, observation3 = st.columns(3, gap="medium")

    # Custom HTML cards replace st.metric here so Streamlit cannot inject
    # its directional arrow/square glyph or theme-dependent delta styling.
    observation_cards = [
        ("Largest Macro-F1 improvement", best_f1_row["Condition"],
         f"{best_f1_row['Macro-F1 Change (pp)']:+.2f} pp"),
        ("Largest accuracy improvement", best_accuracy_row["Condition"],
         f"{best_accuracy_row['Accuracy Change (pp)']:+.2f} pp"),
        ("Largest accuracy decrease", worst_accuracy_row["Condition"],
         f"{worst_accuracy_row['Accuracy Change (pp)']:+.2f} pp"),
    ]
    for column, (label, condition, change) in zip(
        (observation1, observation2, observation3), observation_cards
    ):
        with column:
            render_html(
                '<div class="fer-observation-card">'
                f'<div class="fer-observation-label">{esc(label)}</div>'
                f'<div class="fer-observation-condition">{esc(condition)}</div>'
                f'<div class="fer-observation-change">{esc(change)}</div>'
                '</div>'
            )

    render_html(
        '<div class="fer-card fer-spaced">'
        '<div class="fer-section">Research observations</div>'
        '<ul class="fer-list">'
        '<li>Clean images show no change because no adaptive correction '
        'is applied.</li>'
        '<li>Occlusion results remain unchanged because the current '
        'adaptive pipeline has no occlusion-specific correction.</li>'
        '<li>Accuracy and macro-F1 can move in different directions.</li>'
        '</ul>'
        '<p class="fer-conclusion">Conclusion: adaptive preprocessing '
        'produces mixed results. Its effect depends on image condition '
        'and evaluation metric.</p>'
        '</div>'
    )

    # CSV keeps the original single-table layout and numbers.
    st.download_button(
        label="Download experimental results as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="fer_experimental_results.csv",
        mime="text/csv",
        key="download_results_csv",
    )


# ============================================================
# 11. PDF 2 - DEDICATED REPORT (ReportLab Platypus)
# ============================================================

def create_pdf_report(
    original,
    face,
    quality,
    strategies,
    baseline,
    adaptive,
    face_status=None,
):
    """
    Generate the dedicated A4 image-analysis report.

    Returns the PDF as bytes, or None if reportlab cannot be imported.
    Page 1 normally holds the input images, the condition assessment and
    the prediction summary; page 2 holds the six-class scores and the
    research limitations.
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.platypus import (
            HRFlowable,
            Image as PDFImage,
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

    except ImportError:
        return None

    burgundy = colors.HexColor("#773B4A")
    burgundy_dark = colors.HexColor("#552B38")
    blush = colors.HexColor("#EAD9D7")
    header_fill = colors.HexColor("#F1E6E2")
    label_fill = colors.HexColor("#F8F1EE")
    cream = colors.HexColor("#FBF7F4")
    grid_color = colors.HexColor("#E3D8D2")
    note_fill = colors.HexColor("#F5EBE9")
    muted = colors.HexColor("#82767A")
    ink = colors.HexColor("#30272A")

    margin_side = 18 * mm
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin_side,
        rightMargin=margin_side,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        title="Condition-Aware FER - Image Analysis Report",
        author="FER Studio",
    )

    content_width = document.width

    # ---------------- Styles ----------------
    title_style = ParagraphStyle(
        "FERTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=burgundy,
        alignment=TA_LEFT,
        spaceAfter=1,
    )

    subtitle_style = ParagraphStyle(
        "FERSubtitle",
        fontName="Helvetica",
        fontSize=10.5,
        leading=13,
        textColor=burgundy_dark,
        spaceAfter=2,
    )

    meta_style = ParagraphStyle(
        "FERMeta",
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=muted,
    )

    heading_style = ParagraphStyle(
        "FERHeading",
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=burgundy,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "FERBody",
        fontName="Helvetica",
        fontSize=8.8,
        leading=12,
        textColor=ink,
    )

    cell_style = ParagraphStyle(
        "FERCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=ink,
    )

    cell_bold = ParagraphStyle(
        "FERCellBold",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )

    cell_emphasis = ParagraphStyle(
        "FERCellEmphasis",
        parent=cell_style,
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=burgundy,
    )

    head_cell = ParagraphStyle(
        "FERHeadCell",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=burgundy_dark,
    )

    caption_style = ParagraphStyle(
        "FERCaption",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=muted,
        alignment=1,
    )

    def cell(text, style=None):
        return Paragraph(xml_escape(str(text)), style or cell_style)

    def head(text):
        return Paragraph(xml_escape(str(text)), head_cell)

    def base_table_style(extra=None):
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), header_fill),
            ("GRID", (0, 0), (-1, -1), 0.5, grid_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]

        if extra:
            commands.extend(extra)

        return TableStyle(commands)

    def fitted_image(pil_image, max_w, max_h):
        """Embed an image scaled to fit the box, keeping its aspect ratio."""

        picture = pil_image.copy()

        if picture.mode not in ("RGB", "L"):
            picture = picture.convert("RGB")

        picture.thumbnail((1200, 1200), Image.Resampling.LANCZOS)

        pixel_w, pixel_h = picture.size
        scale = min(max_w / pixel_w, max_h / pixel_h)

        image_buffer = BytesIO()
        picture.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        flowable = PDFImage(
            image_buffer,
            width=pixel_w * scale,
            height=pixel_h * scale,
        )
        flowable.hAlign = "CENTER"

        return flowable

    class NumberedCanvas(rl_canvas.Canvas):
        """Canvas that writes 'Page X of Y' in the footer."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_pages = []

        def showPage(self):
            self._saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_pages)

            for state in self._saved_pages:
                self.__dict__.update(state)
                self._draw_footer(total)
                super().showPage()

            super().save()

        def _draw_footer(self, total):
            self.saveState()
            self.setStrokeColor(grid_color)
            self.setLineWidth(0.5)
            self.line(
                margin_side, 14 * mm,
                A4[0] - margin_side, 14 * mm,
            )
            self.setFont("Helvetica", 7.5)
            self.setFillColor(muted)
            self.drawString(
                margin_side,
                9.5 * mm,
                "FER Studio - Condition-Aware FER | Research prototype",
            )
            self.drawRightString(
                A4[0] - margin_side,
                9.5 * mm,
                f"Page {self._pageNumber} of {total}",
            )
            self.restoreState()

    story = []

    # ---------------- Title block ----------------
    story.append(
        Paragraph("Condition-Aware Facial Expression Recognition", title_style)
    )
    story.append(Paragraph("Image Analysis Report", subtitle_style))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')} "
            f"&nbsp;|&nbsp; Model: ResNet18 V8 (6 classes, "
            f"112 x 112 RGB input) &nbsp;|&nbsp; "
            f"Inference device: {xml_escape(DEVICE.type.upper())}",
            meta_style,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=burgundy,
            spaceAfter=2,
        )
    )

    # ---------------- Input images (side by side) ----------------
    half = content_width / 2
    image_box_w = half - 8 * mm
    image_box_h = 88 * mm

    original_caption = f"{original.size[0]} x {original.size[1]} px"
    face_caption = (
        f"{face.size[0]} x {face.size[1]} px"
        + (f" - {face_status}" if face_status else "")
    )

    image_table = Table(
        [
            [head("Original image"), head("Face crop used for analysis")],
            [
                fitted_image(original, image_box_w, image_box_h),
                fitted_image(face, image_box_w, image_box_h),
            ],
            [
                Paragraph(xml_escape(original_caption), caption_style),
                Paragraph(xml_escape(face_caption), caption_style),
            ],
        ],
        colWidths=[half, half],
    )

    image_table.setStyle(
        base_table_style([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 1), (-1, 1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
        ])
    )

    story.append(
        KeepTogether([
            Paragraph("Input image and detected face", heading_style),
            image_table,
        ])
    )

    # ---------------- Condition assessment ----------------
    def status_cell(flagged):
        if flagged:
            return Paragraph(
                '<font color="#9A3B2F"><b>Flagged</b></font>', cell_style
            )

        return Paragraph(
            '<font color="#2F6B45">Not flagged</font>', cell_style
        )

    condition_rows = [
        [
            head("Measurement"),
            head("Value"),
            head("Threshold"),
            head("Status"),
        ],
        [
            cell("Brightness (0-255)"),
            cell(f"{quality['brightness']:.2f}"),
            cell(f"Low illumination if < {LOW_LIGHT_THRESHOLD}"),
            status_cell(quality["low_light"]),
        ],
        [
            cell("Sharpness (Laplacian variance)"),
            cell(f"{quality['sharpness']:.2f}"),
            cell(f"Possible blur if < {BLUR_THRESHOLD}"),
            status_cell(quality["blur"]),
        ],
        [
            cell("Noise indicator"),
            cell(f"{quality['noise']:.2f}"),
            cell(f"Possible noise if > {NOISE_THRESHOLD}"),
            status_cell(quality["noise_flag"]),
        ],
        [
            cell("Detected condition", cell_bold),
            cell(quality["condition"], cell_emphasis),
            "",
            "",
        ],
        [
            cell("Selected preprocessing", cell_bold),
            cell(", ".join(strategies), cell_emphasis),
            "",
            "",
        ],
    ]

    condition_table = Table(
        condition_rows,
        colWidths=[
            content_width * 0.31,
            content_width * 0.13,
            content_width * 0.36,
            content_width * 0.20,
        ],
    )

    condition_table.setStyle(
        base_table_style([
            ("SPAN", (1, 4), (-1, 4)),
            ("SPAN", (1, 5), (-1, 5)),
            ("BACKGROUND", (0, 4), (0, 5), note_fill),
            ("BACKGROUND", (1, 4), (-1, 5), cream),
            ("TOPPADDING", (0, 4), (-1, 5), 6),
            ("BOTTOMPADDING", (0, 4), (-1, 5), 6),
        ])
    )

    story.append(
        KeepTogether([
            Paragraph("Image condition assessment", heading_style),
            condition_table,
            Spacer(1, 3),
            Paragraph(
                "Condition assessment uses simple heuristic measurements; "
                "it is not a trained condition classifier, and partial "
                "occlusion is not reliably detected.",
                meta_style,
            ),
        ])
    )

    # ---------------- Prediction summary ----------------
    def display_rule(result):
        return "Ambiguous" if result["uncertain"] else "Passes display rule"

    prediction_rows = [
        [
            head("Prediction"),
            head("Expression"),
            head("Top-class score"),
            head("Inference time"),
            head("Display rule"),
        ],
        [
            Paragraph(
                "<b>Baseline</b><br/>"
                '<font size="7" color="#82767A">Original face crop</font>',
                cell_style,
            ),
            cell(baseline["emotion"], cell_emphasis),
            cell(f"{baseline['confidence']:.2%}"),
            cell(f"{baseline['time_ms']:.1f} ms"),
            cell(display_rule(baseline)),
        ],
        [
            Paragraph(
                "<b>Adaptive</b><br/>"
                '<font size="7" color="#82767A">Processed face crop</font>',
                cell_style,
            ),
            cell(adaptive["emotion"], cell_emphasis),
            cell(f"{adaptive['confidence']:.2%}"),
            cell(f"{adaptive['time_ms']:.1f} ms"),
            cell(display_rule(adaptive)),
        ],
    ]

    prediction_table = Table(
        prediction_rows,
        colWidths=[
            content_width * 0.25,
            content_width * 0.19,
            content_width * 0.18,
            content_width * 0.17,
            content_width * 0.21,
        ],
    )

    prediction_table.setStyle(
        base_table_style([
            ("BACKGROUND", (0, 1), (0, 1), label_fill),
            ("BACKGROUND", (0, 2), (0, 2), note_fill),
        ])
    )

    story.append(
        KeepTogether([
            Paragraph("Expression recognition results", heading_style),
            prediction_table,
        ])
    )

    # ---------------- six-class scores ----------------
    baseline_top = int(np.argmax(baseline["probabilities"]))
    adaptive_top = int(np.argmax(adaptive["probabilities"]))

    probability_rows = [
        [
            head("Expression"),
            head("Baseline score"),
            head("Adaptive score"),
            head("Change (pp)"),
        ]
    ]

    for index, emotion in enumerate(EMOTIONS):

        base_value = float(baseline["probabilities"][index])
        adapt_value = float(adaptive["probabilities"][index])

        probability_rows.append([
            cell(emotion),
            cell(
                f"{base_value:.2%}",
                cell_bold if index == baseline_top else cell_style,
            ),
            cell(
                f"{adapt_value:.2%}",
                cell_bold if index == adaptive_top else cell_style,
            ),
            cell(f"{(adapt_value - base_value) * 100:+.2f}"),
        ])

    probability_table = Table(
        probability_rows,
        colWidths=[
            content_width * 0.34,
            content_width * 0.22,
            content_width * 0.22,
            content_width * 0.22,
        ],
        repeatRows=1,
    )

    highlight_commands = [
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("BACKGROUND", (1, baseline_top + 1), (1, baseline_top + 1), blush),
        ("BACKGROUND", (2, adaptive_top + 1), (2, adaptive_top + 1), blush),
    ]

    probability_table.setStyle(base_table_style(highlight_commands))

    story.append(
        KeepTogether([
            Paragraph("six-class model scores", heading_style),
            probability_table,
            Spacer(1, 3),
            Paragraph(
                "Shaded bold values mark the highest-scoring class in each "
                "prediction. Change = adaptive score minus baseline score, "
                "in percentage points.",
                meta_style,
            ),
        ])
    )

    # ---------------- Research limitations ----------------
    limitations_text = (
        "<b>Research limitations</b><br/>"
        "&bull; Condition assessment is heuristic-based; it is not a "
        "trained condition classifier.<br/>"
        "&bull; Partial occlusion is not reliably detected by the current "
        "quality-assessment implementation.<br/>"
        "&bull; Softmax scores are not calibrated probabilities.<br/>"
        "&bull; The system identifies visible facial expression "
        "categories; it does not determine a person's actual internal "
        "emotional state.<br/>"
        "&bull; Offline condition experiments use known test-condition "
        "labels, which differ from the live heuristic assessment shown "
        "here.<br/>"
        "&bull; This is a research prototype; predictions may be "
        "inaccurate, particularly for underrepresented expression "
        "classes."
    )

    limitations_table = Table(
        [[Paragraph(limitations_text, body_style)]],
        colWidths=[content_width],
    )

    limitations_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), note_fill),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, burgundy),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(Spacer(1, 10))
    story.append(KeepTogether([limitations_table]))

    document.build(story, canvasmaker=NumberedCanvas)

    return buffer.getvalue()


# ============================================================
# 12. DASHBOARD STYLING (screen + PDF 1 print layout)
# ============================================================

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

@page {
    size: A4;
    margin: 12mm;
}

:root {
    --cream: #F7F3EF;
    --white: #FFFFFF;
    --burgundy: #773B4A;
    --burgundy-dark: #552B38;
    --blush: #EAD9D7;
    --ink: #30272A;
    --muted: #82767A;
    --border: #EAE1DC;
    --sidebar-bg: #EFE7E2;
}

html, body, .stApp {
    background: var(--cream);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
    max-width: 100vw;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    overflow-x: hidden;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1360px;
    padding: 1.8rem 2.5rem 3rem 2.5rem;
}

h1, h2, h3, p, label, li {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

[data-testid="stMarkdownContainer"] {
    max-width: 100%;
    overflow-wrap: break-word;
}

[data-testid="stImage"] img {
    max-width: 100%;
    max-height: 520px;
    object-fit: contain;
    border-radius: 12px;
}

[data-testid="stImageCaption"] {
    color: var(--muted);
}

/* Print-only blocks are hidden on screen */
.print-only {
    display: none !important;
}

/* ---------- Sidebar (compact) ---------- */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebarHeader"] {
    padding: 0.5rem 1rem 0 1rem;
    min-height: 0;
    height: auto;
}

[data-testid="stSidebarUserContent"] {
    padding: 0.4rem 1rem 1rem 1rem;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.25rem;
}

.sb-brand {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--burgundy-dark);
    line-height: 1.2;
}

.sb-tag {
    font-size: 0.66rem;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2px 0 10px 0;
}

.sb-h {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--burgundy-dark);
    margin: 12px 0 4px 0;
    padding-top: 10px;
    border-top: 1px solid rgba(85, 43, 56, 0.14);
}

.sb-h.first {
    border-top: none;
    padding-top: 0;
    margin-top: 0;
}

/* Sidebar navigation: native markdown list of #anchor links */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul {
    list-style: none;
    margin: 0;
    padding: 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul li {
    margin: 0;
    padding: 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul li a {
    display: block;
    color: var(--burgundy-dark);
    text-decoration: none;
    padding: 5px 10px;
    margin: 1px 0;
    border-radius: 8px;
    font-size: 0.84rem;
    font-weight: 600;
    line-height: 1.25;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul li a:hover {
    background: var(--blush);
}

.sb-kv {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--ink);
}

.sb-kv span {
    color: var(--muted);
}

.sb-kv b {
    font-weight: 600;
    text-align: right;
}

.sb-pipe {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: 0.78rem;
    line-height: 1.55;
    color: var(--muted);
}

.sb-pipe li {
    color: var(--muted);
}

.sb-pipe b {
    color: var(--burgundy);
    font-weight: 700;
    margin-right: 6px;
}

.sb-status {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.78rem;
    line-height: 1.55;
    color: var(--ink);
}

.sb-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4F8A5B;
    flex: 0 0 auto;
}

.sb-dot.bad {
    background: #B3453F;
}

.sb-note {
    font-size: 0.7rem;
    line-height: 1.45;
    color: var(--muted);
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(85, 43, 56, 0.14);
}

/* ---------- Typography ---------- */
.fer-brand {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 3px;
    color: var(--burgundy);
    text-transform: uppercase;
}

.fer-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 4vw, 3.5rem);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
    margin: 8px 0 10px 0;
}

.fer-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 720px;
    margin-bottom: 20px;
}

.fer-section {
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    color: var(--burgundy);
    text-transform: uppercase;
}

[data-testid="stHeading"] h2,
.stApp h2 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.6rem, 2.6vw, 2.1rem);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.2;
    margin: 0;
    padding: 4px 0 6px 0;
    scroll-margin-top: 5rem;
}

.fer-h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--ink);
    margin: 18px 0 8px 0;
}

.fer-h4 {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--burgundy);
    margin: 12px 0 6px 0;
}

.fer-lead {
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.65;
    max-width: 820px;
    margin-bottom: 12px;
}

.fer-sechead {
    margin-top: 2.2rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--border);
    break-after: avoid;
    page-break-after: avoid;
}

.fer-sechead.first {
    margin-top: 1.6rem;
    padding-top: 0;
    border-top: none;
}

.fer-muted {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

.fer-muted.small {
    font-size: 0.8rem;
}

.fer-kicker {
    margin-top: 14px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
}

/* ---------- Cards ---------- */
.fer-grid {
    display: grid;
    gap: 18px;
    margin: 6px 0 10px 0;
    align-items: stretch;
}

.fer-grid-3 {
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
}

.fer-grid-2 {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.fer-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 5px 20px rgba(65, 40, 45, 0.04);
    box-sizing: border-box;
    min-width: 0;
}

.fer-card.fer-spaced {
    margin: 10px 0 14px 0;
}

.fer-card-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
    color: var(--burgundy);
    margin: 8px 0 6px 0;
}

.fer-card-value.sans {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
}

.fer-empty {
    text-align: center;
    padding: 38px 24px;
}

.fer-empty .star {
    font-size: 2.4rem;
    color: var(--burgundy);
    line-height: 1;
}

.fer-empty .headline {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--ink);
    margin: 12px 0 6px 0;
}

.fer-empty .flow {
    margin-top: 20px;
    color: var(--burgundy);
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.8px;
    line-height: 1.8;
}

.fer-pending {
    background: rgba(255, 255, 255, 0.6);
    border: 1px dashed #D9C7C5;
    border-radius: 14px;
    padding: 14px 18px;
    color: var(--muted);
    font-size: 0.88rem;
}

.fer-chip {
    display: inline-block;
    background: #F5EBE9;
    color: var(--burgundy-dark);
    border-radius: 999px;
    padding: 4px 12px;
    margin: 3px 6px 3px 0;
    font-size: 0.82rem;
    font-weight: 600;
}

.fer-pred-emotion {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--burgundy);
    margin: 10px 0 2px 0;
}

.fer-pred-score {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--ink);
    margin: 2px 0 6px 0;
}

.fer-bar {
    height: 8px;
    background: #F0E6E2;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 6px;
}

.fer-bar-fill {
    height: 100%;
    background: var(--burgundy);
    border-radius: 999px;
}

.fer-mini-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 6px;
    font-size: 0.86rem;
}

.fer-mini-table th {
    text-align: left;
    font-size: 0.7rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
}

.fer-mini-table td {
    padding: 5px 0;
    border-bottom: 1px solid #F3ECE8;
    color: var(--ink);
}

.fer-mini-table .num,
.fer-table .num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.fer-flag {
    margin-top: 12px;
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 0.82rem;
    line-height: 1.5;
}

.fer-flag.warn {
    background: #FFF4DC;
    color: #6B4E0A;
}

.fer-flag.ok {
    background: #EAF2EC;
    color: #2F5A3B;
}

.fer-note {
    background: #F5EBE9;
    border-left: 4px solid var(--burgundy);
    border-radius: 9px;
    padding: 13px 16px;
    margin: 6px 0 14px 0;
    color: var(--ink);
    font-size: 0.9rem;
    line-height: 1.65;
}

.fer-list {
    margin: 10px 0 0 0;
    padding-left: 1.2rem;
    line-height: 1.8;
    color: var(--muted);
    font-size: 0.92rem;
}

.fer-conclusion {
    color: var(--ink);
    font-weight: 600;
    margin: 12px 0 0 0;
    line-height: 1.6;
}

.fer-body {
    color: var(--muted);
    line-height: 1.7;
    font-size: 0.93rem;
    margin: 8px 0;
}

/* ---------- Tables ---------- */
.fer-table-wrap {
    overflow-x: auto;
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin: 6px 0 12px 0;
}

.fer-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}

.fer-table.fixed {
    table-layout: fixed;
}

.fer-table th {
    background: #F5EEEA;
    color: var(--burgundy-dark);
    text-align: left;
    font-size: 0.74rem;
    letter-spacing: 0.4px;
    font-weight: 700;
    padding: 10px 12px;
    white-space: normal;
}

.fer-table th.num {
    text-align: right;
}

.fer-table td {
    padding: 9px 12px;
    border-top: 1px solid #F3ECE8;
    color: var(--ink);
    overflow-wrap: break-word;
}

.fer-table tbody tr:nth-child(even) td {
    background: #FCFAF8;
}

/* ---------- Print-only bar charts (PDF 1) ---------- */
.fer-hbars {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 16px;
    margin: 6px 0 12px 0;
}

.fer-hbar-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 18px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 8px;
}

.fer-legend-item {
    display: inline-flex;
    align-items: center;
}

.fer-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 3px;
    margin-right: 6px;
}

.fer-legend-note {
    margin-left: auto;
    font-weight: 400;
    color: var(--muted);
}

.fer-hbar-group {
    display: grid;
    grid-template-columns: 26% 1fr;
    gap: 10px;
    align-items: center;
    padding: 6px 0;
    border-top: 1px solid #F3ECE8;
}

.fer-hbar-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--ink);
    overflow-wrap: break-word;
}

.fer-hbar-row {
    display: grid;
    grid-template-columns: 1fr 58px;
    gap: 8px;
    align-items: center;
    margin: 2px 0;
}

.fer-hbar-track {
    height: 10px;
    background: #F5EEEA;
    border-radius: 999px;
    overflow: hidden;
}

.fer-hbar-fill {
    height: 100%;
    border-radius: 999px;
}

.fer-hbar-val {
    font-size: 0.76rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
}

/* ---------- Streamlit widgets ---------- */
[data-testid="stFileUploader"] {
    background: var(--white);
    border: 1px dashed #C9A9AC;
    border-radius: 18px;
    padding: 16px 18px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FCF9F7;
    border: 1px solid #E5D6D3;
    border-radius: 12px;
}

/* File uploader: keep the dropzone copy and action button on-theme. */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] * {
    color: var(--burgundy-dark);
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
    background: var(--burgundy) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--burgundy) !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * {
    color: #FFFFFF !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover {
    background: var(--burgundy-dark) !important;
    border-color: var(--burgundy-dark) !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stCaption"],
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}

/* Neutral cream/burgundy face-detection status (instead of Streamlit green). */
.fer-face-status {
    background: #F1E6E2;
    color: var(--burgundy-dark);
    border: 1px solid #DCC6C5;
    border-left: 4px solid var(--burgundy);
    border-radius: 11px;
    padding: 0.7rem 0.95rem;
    font-weight: 650;
    line-height: 1.45;
    margin: 0.35rem 0 0.8rem 0;
}

[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 3px 12px rgba(65, 40, 45, 0.035);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--burgundy);
    font-weight: 800;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] *,
[data-testid="stMetricDelta"] svg {
    color: var(--burgundy-dark) !important;
    fill: var(--burgundy-dark) !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* Ensure Plotly SVG labels, ticks, titles and legend text remain dark. */
[data-testid="stPlotlyChart"] text,
.stPlotlyChart text,
.js-plotly-plot text {
    fill: #30272A !important;
    color: #30272A !important;
}

.stButton > button,
.stDownloadButton > button,
[data-testid="stDownloadButton"] button {
    background: var(--burgundy);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
}

.stButton > button p,
.stDownloadButton > button p,
[data-testid="stDownloadButton"] button p {
    color: #FFFFFF;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    background: var(--burgundy-dark);
    color: #FFFFFF;
    border: none;
}

[data-testid="stExpander"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--burgundy-dark);
}

[data-testid="stExpander"] summary:hover {
    background: #FBF7F4;
}

[data-testid="stAlert"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-baseweb="select"] > div {
    background: white;
    border-color: var(--border);
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

#MainMenu, footer {
    visibility: hidden;
}

/* ---------- Responsive ---------- */
@media screen and (max-width: 800px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .fer-card {
        padding: 16px;
    }

    .fer-grid-2,
    .fer-grid-3 {
        grid-template-columns: 1fr;
    }

    .fer-table th,
    .fer-table td {
        padding: 8px 8px;
        font-size: 0.8rem;
    }
}

/* ============================================================
   PRINT / SAVE AS PDF  (PDF 1: printed copy of the dashboard)
   ============================================================ */
@media print {

    /* Un-clip Streamlit's scrolling app shell so content flows over
       as many pages as it needs, with no blank trailing pages. */
    html, body, .stApp,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section.main, .main, .block-container {
        display: block !important;
        position: static !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        transform: none !important;
        background: #FFFFFF !important;
        color: #30272A !important;
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Controls that should not appear on paper */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stBottom"],
    [data-testid="stFileUploader"],
    [data-testid="stDownloadButton"],
    [data-testid="stSpinner"],
    .stButton,
    .stDownloadButton,
    footer,
    #MainMenu,
    .no-print {
        display: none !important;
    }

    /* Interactive Plotly charts are replaced by the HTML bar versions */
    [data-testid="stPlotlyChart"],
    .stPlotlyChart,
    .js-plotly-plot {
        display: none !important;
    }

    /* Streamlit leaves an empty wrapper behind a hidden element; hide the
       wrapper too so "keep with next" binds to the visible content. */
    [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]),
    [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]),
    [data-testid="stElementContainer"]:has([data-testid="stDownloadButton"]),
    [data-testid="stElementContainer"]:has([data-testid="stSpinner"]),
    [data-testid="element-container"]:has([data-testid="stPlotlyChart"]),
    [data-testid="element-container"]:has([data-testid="stFileUploader"]),
    [data-testid="element-container"]:has([data-testid="stDownloadButton"]),
    [data-testid="element-container"]:has([data-testid="stSpinner"]) {
        display: none !important;
    }

    .print-only {
        display: block !important;
    }

    * {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }

    body, p, li, td, th {
        font-size: 10.5pt;
        orphans: 3;
        widows: 3;
    }

    /* Block flow paginates predictably in every browser (Safari
       included), unlike flex columns. */
    [data-testid="stVerticalBlock"] {
        display: block !important;
        gap: 0 !important;
    }

    [data-testid="stElementContainer"],
    [data-testid="element-container"] {
        display: block !important;
        width: 100% !important;
        margin: 0 0 8px 0 !important;
        break-inside: avoid;
        page-break-inside: avoid;
    }

    /* Columns stay side by side and never overlap */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 12px !important;
        width: 100% !important;
    }

    [data-testid="stColumn"],
    [data-testid="column"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        width: auto !important;
    }

    /* Cards, metrics and tables stay in one piece */
    .fer-card,
    .fer-note,
    .fer-pending,
    .fer-hbars,
    .fer-table-wrap,
    .fer-table tr,
    .fer-hbar-group,
    [data-testid="stMetric"],
    [data-testid="stImage"],
    [data-testid="stExpander"],
    [data-testid="stAlert"] {
        break-inside: avoid;
        page-break-inside: avoid;
    }

    .fer-card,
    .fer-hbars,
    [data-testid="stMetric"] {
        box-shadow: none !important;
    }

    .fer-card {
        padding: 14px 16px;
        border-radius: 12px;
    }

    [data-testid="stMetric"] {
        padding: 10px 12px;
    }

    .fer-grid-3 {
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 12px;
    }

    .fer-grid-2 {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 12px;
    }

    .fer-title {
        font-size: 2rem;
    }

    .fer-card-value {
        font-size: 1.3rem;
    }

    .fer-pred-emotion {
        font-size: 1.5rem;
    }

    .fer-empty {
        padding: 18px;
    }

    /* Tables fit the page width, wrap their text, never scroll */
    .fer-table-wrap {
        overflow: visible !important;
    }

    .fer-table {
        width: 100% !important;
        table-layout: fixed;
        font-size: 9.5pt;
    }

    .fer-table th,
    .fer-table td {
        padding: 5px 8px;
        white-space: normal !important;
        overflow-wrap: break-word;
        word-break: normal;
    }

    .fer-mini-table {
        font-size: 9.5pt;
    }

    /* Images never exceed the printable width or a sensible height */
    [data-testid="stImage"] {
        width: 100% !important;
        text-align: center;
    }

    [data-testid="stImage"] img {
        display: block;
        margin: 0 auto;
        width: auto !important;
        height: auto !important;
        max-width: 100% !important;
        max-height: 85mm !important;
        object-fit: contain;
    }

    /* Keep headings with the content that follows them. Streamlit wraps
       every element in its own container, so the rule must sit on the
       wrapper (via :has) and not only on the heading inside it. */
    [data-testid="stHeading"],
    .fer-sechead,
    .fer-h3,
    .fer-h4,
    .fer-lead,
    [data-testid="stElementContainer"]:has(> [data-testid="stHeading"]),
    [data-testid="stElementContainer"]:has(.fer-sechead),
    [data-testid="stElementContainer"]:has(.fer-h3),
    [data-testid="stElementContainer"]:has(.fer-h4),
    [data-testid="stElementContainer"]:has(.fer-lead),
    [data-testid="element-container"]:has(> [data-testid="stHeading"]),
    [data-testid="element-container"]:has(.fer-sechead),
    [data-testid="element-container"]:has(.fer-h3),
    [data-testid="element-container"]:has(.fer-h4),
    [data-testid="element-container"]:has(.fer-lead) {
        break-after: avoid;
        page-break-after: avoid;
    }

    [data-testid="stHeading"] h2,
    .stApp h2 {
        font-size: 1.45rem;
    }

    .fer-sechead {
        margin-top: 14px;
        padding-top: 10px;
    }

    .fer-sechead.first {
        margin-top: 0;
    }
}

/* ---------- Final UI overrides: upload chip, expanders, observations ---------- */
.fer-observation-card {
    min-height: 112px;
    box-sizing: border-box;
    padding: 18px 20px;
    background: #FFFFFF !important;
    border: 1px solid #EAE1DC;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(65, 40, 45, 0.035);
}
.fer-observation-label {
    color: #55484B !important;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.fer-observation-condition {
    color: #30272A !important;
    font-size: clamp(1.15rem, 1.6vw, 1.65rem);
    line-height: 1.2;
    font-weight: 800;
    overflow-wrap: anywhere;
    margin-bottom: 10px;
}
.fer-observation-change {
    display: inline-block;
    color: #552B38 !important;
    background: #F1E6E2 !important;
    border: 1px solid #DCC6C5;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.95rem;
    line-height: 1.2;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}

/* Streamlit uploader's selected-file chip (current and alternate DOMs). */
[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div > div {
    background: #F1E6E2 !important;
    background-color: #F1E6E2 !important;
    color: #552B38 !important;
    border: 1px solid #DCC6C5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"] * {
    color: #552B38 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"] svg {
    color: #773B4A !important;
    fill: #773B4A !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"] button {
    background: #FBF7F4 !important;
    border: 1px solid #DCC6C5 !important;
    color: #773B4A !important;
}

/* Expander header: explicitly override Streamlit/BaseWeb dark-theme surfaces. */
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary > div,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    background: #F1E6E2 !important;
    background-color: #F1E6E2 !important;
    color: #552B38 !important;
    border-color: #DCC6C5 !important;
    opacity: 1 !important;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover > div {
    background: #EAD9D7 !important;
    color: #552B38 !important;
}
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary svg * {
    color: #773B4A !important;
    fill: #773B4A !important;
    stroke: #773B4A !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    color: #30272A !important;
}

/* ---------- Final contrast fixes: uploaded-file chip + metric deltas ---------- */
@media screen {
    /* Selected upload: Streamlit versions vary in the nested chip markup.
       Force the chip and its non-icon descendants onto the cream palette. */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFile"] > div > div,
    [data-testid="stFileUploaderFile"] [data-testid],
    [data-testid="stFileUploaderFile"] [role="button"] {
        background: #F1E6E2 !important;
        background-color: #F1E6E2 !important;
        color: #552B38 !important;
        border-color: #DCC6C5 !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileSize"] {
        color: #552B38 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderFile"] button {
        background: #FBF7F4 !important;
        color: #773B4A !important;
        border: 1px solid #DCC6C5 !important;
    }

    [data-testid="stFileUploaderFile"] button svg {
        color: #773B4A !important;
        fill: #773B4A !important;
        background: transparent !important;
    }

    /* Neutral metric delta pill; hide Streamlit's directional icon,
       which appears as an unwanted solid square with the theme overrides. */
    [data-testid="stMetricDelta"] {
        display: inline-flex !important;
        width: fit-content !important;
        align-items: center !important;
        background: #F1E6E2 !important;
        color: #552B38 !important;
        border: 1px solid #DCC6C5 !important;
        border-radius: 999px !important;
        padding: 3px 9px !important;
        font-weight: 750 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricDelta"] > div,
    [data-testid="stMetricDelta"] span {
        background: transparent !important;
        color: #552B38 !important;
        fill: #552B38 !important;
        font-weight: 750 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricDelta"] svg {
        display: none !important;
    }
}
</style>
"""


def inject_styles():
    render_html(DASHBOARD_CSS)


# ============================================================
# 13. SIDEBAR (compact, functional navigation)
# ============================================================

NAV_ITEMS = [
    ("01", "image-analysis", "Image analysis"),
    ("02", "face-detection", "Face detection"),
    ("03", "condition-assessment", "Condition assessment"),
    ("04", "prediction-comparison", "Prediction comparison"),
    ("05", "expression-distribution", "Expression distribution"),
    ("06", "research-evaluation", "Research evaluation"),
    ("07", "research-notes", "Research notes"),
]


def render_sidebar():

    # Native markdown links to st.header(anchor=...) targets; no script and
    # no access to the parent document is needed.
    nav_markdown = "\n".join(
        f"- [{number} · {label}](#{anchor})"
        for number, anchor, label in NAV_ITEMS
    )

    def status_row(label, exists):
        dot = "sb-dot" if exists else "sb-dot bad"
        text = "Found" if exists else "Missing"
        return (
            f'<div class="sb-status"><span class="{dot}"></span>'
            f'{esc(label)}: {text}</div>'
        )

    # Only file existence is checked here, not whether anything loaded.
    status_rows = "".join([
        status_row("Model checkpoint", MODEL_PATH.exists()),
        status_row("Face detector", FACE_MODEL_PATH.exists()),
        status_row("Baseline results", BASELINE_RESULTS_PATH.exists()),
        status_row("Adaptive results", ADAPTIVE_RESULTS_PATH.exists()),
    ])

    pipeline_steps = [
        "Facial image input",
        "Face detection and cropping",
        "Image-quality assessment",
        "Adaptive preprocessing",
        "Expression classification",
    ]

    pipeline_html = "".join(
        f"<li><b>{index:02d}</b>{esc(step)}</li>"
        for index, step in enumerate(pipeline_steps, start=1)
    )

    with st.sidebar:
        render_html(
            '<div>'
            '<div class="sb-brand">✦ FER Studio</div>'
            '<div class="sb-tag">Condition-aware vision system</div>'
            '<div class="sb-h first">Navigation</div>'
            '</div>'
        )

        st.markdown(nav_markdown)

        render_html(
            '<div>'
            '<div class="sb-h">Model information</div>'
            '<div class="sb-kv"><span>Model</span>'
            '<b>ResNet18 V8</b></div>'
            '<div class="sb-kv"><span>Classes</span>'
            '<b>6 expression classes</b></div>'
            '<div class="sb-kv"><span>Input</span>'
            '<b>112 × 112 RGB</b></div>'
            f'<div class="sb-kv"><span>Device</span>'
            f'<b>{esc(DEVICE.type.upper())}</b></div>'
            '<div class="sb-h">Processing pipeline</div>'
            f'<ul class="sb-pipe">{pipeline_html}</ul>'
            '<div class="sb-h">System status</div>'
            '<div class="sb-status"><span class="sb-dot"></span>'
            'Application interface active</div>'
            f'{status_rows}'
            '<div class="sb-note">Status shows whether files exist, not '
            'that they loaded successfully. Research prototype: '
            'image-condition assessment is heuristic-based, and model '
            'scores are not calibrated measures of actual emotional '
            'states.</div>'
            '</div>'
        )


# ============================================================
# 14. PAGE HEADER
# ============================================================

def render_page_header():

    render_html(
        '<div>'
        '<div class="fer-brand">✦ Intelligent Vision Research</div>'
        '<div class="fer-title">Condition-Aware FER</div>'
        '<div class="fer-subtitle">A condition-aware facial expression '
        'recognition system combining ResNet18-based deep learning, facial '
        'analysis and adaptive image preprocessing.</div>'
        '<div class="fer-grid fer-grid-3">'
        '<div class="fer-card">'
        '<div class="fer-section">Architecture</div>'
        '<div class="fer-card-value">ResNet18</div>'
        '<div class="fer-muted">V8 facial expression model</div>'
        '</div>'
        '<div class="fer-card">'
        '<div class="fer-section">Input resolution</div>'
        '<div class="fer-card-value">112 × 112</div>'
        '<div class="fer-muted">RGB facial images</div>'
        '</div>'
        '<div class="fer-card">'
        '<div class="fer-section">Classification</div>'
        '<div class="fer-card-value">06 Classes</div>'
        '<div class="fer-muted">Facial expression categories</div>'
        '</div>'
        '</div>'
        '</div>'
    )


# ============================================================
# 15. IMAGE ANALYSIS SECTIONS (02-05 + report download)
# ============================================================

def render_empty_state():

    render_html(
        '<div class="fer-card fer-empty">'
        '<div class="star">✦</div>'
        '<div class="headline">Your analysis starts here</div>'
        '<div class="fer-muted">Upload a facial image to begin the '
        'condition-aware emotion recognition pipeline.</div>'
        '<div class="flow">IMAGE INPUT → FACE DETECTION → QUALITY '
        'ASSESSMENT → ADAPTIVE PROCESSING → CLASSIFICATION</div>'
        '</div>'
    )


def render_pending_sections():
    """Placeholders so every navigation link has a destination."""

    pending = [
        ("face-detection", "02 — Face detection", "Face detection"),
        ("condition-assessment", "03 — Condition assessment",
         "Condition assessment"),
        ("prediction-comparison", "04 — Prediction comparison",
         "Prediction comparison"),
        ("expression-distribution", "05 — Expression distribution",
         "six-class expression distribution"),
    ]

    for anchor, label, title in pending:
        section_header(anchor, label, title)
        render_html(
            '<div class="fer-pending">Available after a facial image '
            'is uploaded.</div>'
        )


def load_uploaded_image(uploaded_file):
    """Open the upload, apply EXIF orientation and convert to RGB."""

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    return image.convert("RGB")


def render_analysis(uploaded_file):

    try:
        original = load_uploaded_image(uploaded_file)
    except Exception as exc:
        st.error(f"The uploaded file could not be read as an image: {exc}")
        return

    # ---------------- 02 Face detection ----------------
    section_header(
        "face-detection",
        "02 — Face detection",
        "Input and detected face",
    )

    try:
        with st.spinner("Detecting and cropping face..."):
            face, face_status = detect_and_crop_face(original)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    col1, col2 = st.columns(2, gap="large")

    with col1:
        show_image(original, caption="Original image")

    with col2:
        show_image(face, caption="Face crop used for analysis")

    if face_status.startswith("Face detected"):
        render_html(
            f'<div class="fer-face-status">{esc(face_status)}</div>'
        )
    else:
        st.warning(face_status + ". Prediction may be less reliable.")

    # ---------------- 03 Condition assessment ----------------
    quality = assess_condition(face)
    processed, strategies = adaptive_process(face, quality)

    section_header(
        "condition-assessment",
        "03 — Condition assessment",
        "Image quality intelligence",
        lead=(
            "Heuristic measurements of the face crop. This is not a "
            "trained condition classifier."
        ),
    )

    q1, q2, q3 = st.columns(3, gap="medium")

    with q1:
        st.metric(
            "Brightness",
            f"{quality['brightness']:.1f}",
            help="Average image brightness on a 0–255 scale.",
        )

    with q2:
        st.metric(
            "Sharpness",
            f"{quality['sharpness']:.1f}",
            help="Laplacian variance used as a sharpness indicator.",
        )

    with q3:
        st.metric(
            "Noise indicator",
            f"{quality['noise']:.1f}",
            help="Estimated image noise indicator.",
        )

    chips = "".join(
        f'<span class="fer-chip">{esc(name)}</span>'
        for name in strategies
    )

    render_html(
        '<div class="fer-grid fer-grid-2">'
        '<div class="fer-card">'
        '<div class="fer-section">Detected condition</div>'
        f'<div class="fer-card-value sans">{esc(quality["condition"])}</div>'
        '<div class="fer-muted small">Heuristic thresholds: brightness '
        f'below {LOW_LIGHT_THRESHOLD}, Laplacian variance below '
        f'{BLUR_THRESHOLD}, noise indicator above {NOISE_THRESHOLD}. '
        'Partial occlusion is not reliably detected.</div>'
        '</div>'
        '<div class="fer-card">'
        '<div class="fer-section">Selected processing</div>'
        f'<div style="margin:10px 0 8px 0;">{chips}</div>'
        '<div class="fer-muted small">Applied to the face crop before '
        'the adaptive prediction.</div>'
        '</div>'
        '</div>'
    )

    with st.expander("View adaptively processed face"):
        st.image(
            processed,
            caption="Face after adaptive preprocessing",
            width=260,
        )

    # ---------------- 04 Prediction comparison ----------------
    section_header(
        "prediction-comparison",
        "04 — Prediction comparison",
        "Baseline and adaptive predictions",
    )

    try:
        with st.spinner("Running the FER model..."):
            model = load_fer_model()
            baseline = predict_emotion(model, face)
            adaptive = predict_emotion(model, processed)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    baseline_card = prediction_card_html(
        baseline,
        "BASELINE MODEL",
        "Prediction using the original face crop.",
    )

    adaptive_card = prediction_card_html(
        adaptive,
        "ADAPTIVE MODEL",
        "Prediction using the adaptively processed face.",
    )

    render_html(
        f'<div class="fer-grid fer-grid-2">{baseline_card}'
        f'{adaptive_card}</div>'
    )

    st.caption(
        "The uncertainty warning uses a heuristic: top score below "
        "35% or a gap below 8 percentage points between the top two "
        "scores. This rule is not calibrated or validated. Softmax "
        "scores are not probabilities that a person truly experiences "
        "an emotion."
    )

    # ---------------- 05 Expression distribution ----------------
    section_header(
        "expression-distribution",
        "05 — Expression distribution",
        "six-class expression distribution",
    )

    chart_data = pd.DataFrame({
        "Expression": EMOTIONS,
        "Baseline": baseline["probabilities"],
        "Adaptive": adaptive["probabilities"],
    })

    chart_long = chart_data.melt(
        id_vars="Expression",
        value_vars=["Baseline", "Adaptive"],
        var_name="Model",
        value_name="Score",
    )

    # Horizontal chart keeps all six labels readable.
    fig = px.bar(
        chart_long,
        x="Score",
        y="Expression",
        color="Model",
        barmode="group",
        orientation="h",
        text=chart_long["Score"].map(lambda value: f"{value:.1%}"),
        color_discrete_map={
            "Baseline": BASELINE_COLOR,
            "Adaptive": ADAPTIVE_COLOR,
        },
        category_orders={"Expression": EMOTIONS[::-1]},
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
        marker_line_width=0,
    )

    max_score = float(chart_long["Score"].max())

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#30272A",
            size=12,
        ),
        xaxis=dict(
            title="Softmax score",
            tickformat=".0%",
            gridcolor="#EAE1DC",
            automargin=True,
            range=[0, min(1.0, max_score * 1.2 + 0.02)],
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            automargin=True,
            categoryorder="array",
            categoryarray=EMOTIONS[::-1],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            title_text="",
        ),
        margin=dict(l=30, r=60, t=70, b=40),
        bargap=0.25,
    )

    show_plotly(fig)

    axis_max = min(1.0, max_score * 1.1)

    render_print_bars(
        EMOTIONS,
        [
            {
                "name": "Baseline",
                "values": [float(v) for v in baseline["probabilities"]],
                "color": BASELINE_COLOR,
            },
            {
                "name": "Adaptive",
                "values": [float(v) for v in adaptive["probabilities"]],
                "color": ADAPTIVE_COLOR,
            },
        ],
        max_value=axis_max,
        value_format=lambda value: f"{value:.1%}",
        scale_note=f"Bar scale: 0–{axis_max:.0%}",
    )

    with st.expander("View exact model scores"):

        probability_table = pd.DataFrame({
            "Expression": EMOTIONS,
            "Baseline": [
                f"{value:.2%}" for value in baseline["probabilities"]
            ],
            "Adaptive": [
                f"{value:.2%}" for value in adaptive["probabilities"]
            ],
        })

        render_table(
            probability_table,
            numeric_columns=["Baseline", "Adaptive"],
            widths=[40, 30, 30],
        )

    # ---------------- PDF 2: dedicated report download ----------------
    sub_header("Download analysis report")

    try:
        pdf_data = create_pdf_report(
            original,
            face,
            quality,
            strategies,
            baseline,
            adaptive,
            face_status=face_status,
        )
        pdf_error = None
    except Exception as exc:
        pdf_data = None
        pdf_error = exc

    if pdf_data is not None:

        st.download_button(
            label="↓ Download image analysis report (PDF)",
            data=pdf_data,
            file_name="condition_aware_fer_report.pdf",
            mime="application/pdf",
            key="download_analysis_pdf",
        )

    elif pdf_error is not None:

        st.warning(f"The PDF report could not be generated: {pdf_error}")

    else:

        st.info(
            "The reportlab package could not be imported in the active "
            "environment, so the PDF report is unavailable. Make sure the "
            "app is started from the project's .venv."
        )


# ============================================================
# 16. RESEARCH NOTES (section 07)
# ============================================================

def render_research_notes():

    section_header(
        "research-notes",
        "07 — Research notes",
        "Understanding the predictions",
    )

    render_html(
        '<div class="fer-card">'
        '<div class="fer-section">Research note</div>'
        '<p class="fer-body">This is a research prototype. Predictions '
        'may be inaccurate, particularly for underrepresented expression '
        'classes. Condition assessment is heuristic-based, and face '
        'detection may fail for some poses. Experimental results use '
        'known test-condition labels to select preprocessing strategies. '
        'The system classifies visible facial expressions; it does not '
        'establish a person\'s actual emotional state.</p>'
        '<ul class="fer-list">'
        '<li>The live condition assessment is heuristic-based, not a '
        'trained condition classifier.</li>'
        '<li>Partial occlusion is not reliably detected by the current '
        'quality assessment.</li>'
        '<li>Offline condition experiments use known condition labels, '
        'unlike the live assessment.</li>'
        '<li>Softmax scores are not calibrated probabilities.</li>'
        '<li>The system classifies visible facial expression categories, '
        'not internal emotional states.</li>'
        '</ul>'
        '</div>'
    )


# ============================================================
# 17. MAIN APPLICATION FLOW
# ============================================================

def main():

    inject_styles()
    render_sidebar()
    render_page_header()

    # ---------------- 01 Image analysis ----------------
    section_header(
        "image-analysis",
        "01 — Image analysis",
        "Analyse a facial image",
        lead=(
            "Upload an image to detect a face, assess its visual "
            "condition and compare baseline and adaptive expression "
            "predictions."
        ),
        first=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a facial image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a clear image containing a visible face.",
    )

    if uploaded_file is None:
        render_empty_state()
        render_pending_sections()
    else:
        try:
            render_analysis(uploaded_file)
        except Exception as exc:
            st.error("The application encountered an error.")
            st.exception(exc)

    # ---------------- 06 Research evaluation ----------------
    section_header(
        "research-evaluation",
        "06 — Research evaluation",
        "Experimental results",
        lead=(
            "Baseline and adaptive results are compared using accuracy "
            "and macro-F1."
        ),
    )

    try:
        render_experimental_results()
    except Exception as exc:
        st.error("The experimental results could not be displayed.")
        st.exception(exc)

    # ---------------- 07 Research notes ----------------
    render_research_notes()


main()