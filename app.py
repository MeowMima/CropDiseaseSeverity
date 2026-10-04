import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from cv_pipeline import (
    analyze_disease,
    classify_severity,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Krishi Sahyog",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(48, 117, 76, 0.08),
                transparent 28%
            ),
            #f4f7f3;
        color: #183326;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #183326;
    }

    p {
        color: #52645a;
    }

    /* =====================================================
       TOP BAR
       ===================================================== */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.2rem 0 1.4rem 0;
        border-bottom: 1px solid #dce5de;
        margin-bottom: 1.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: #1f6f4a;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.25rem;
    }

    .brand-name {
        font-size: 1.15rem;
        font-weight: 750;
        color: #173b2a;
        letter-spacing: -0.02em;
    }

    .brand-sub {
        font-size: 0.76rem;
        color: #748078;
        margin-top: 1px;
    }

    .system-status {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        background: #ffffff;
        border: 1px solid #dce5de;
        border-radius: 999px;
        padding: 0.42rem 0.75rem;
        font-size: 0.75rem;
        color: #52645a;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #3f8b5d;
        border-radius: 50%;
        display: inline-block;
    }

    /* =====================================================
       INSPECTION HERO
       ===================================================== */

    .inspection-hero {
        background:
            linear-gradient(
                135deg,
                #183f2d 0%,
                #1f6f4a 65%,
                #2c8058 100%
            );
        border-radius: 22px;
        padding: 1.7rem 1.8rem;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(27, 74, 49, 0.13);
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }

    .hero-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        opacity: 0.72;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 760;
        letter-spacing: -0.04em;
        margin: 0;
        color: white;
    }

    .hero-description {
        color: rgba(255,255,255,0.78);
        font-size: 0.92rem;
        margin-top: 0.45rem;
        max-width: 670px;
    }

    .inspection-id {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 13px;
        padding: 0.7rem 0.85rem;
        min-width: 150px;
    }

    .inspection-id-label {
        font-size: 0.63rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.62;
    }

    .inspection-id-value {
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* =====================================================
       SECTION LABELS
       ===================================================== */

    .section-kicker {
        color: #718078;
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 750;
        margin-top: 1.5rem;
        margin-bottom: 0.3rem;
    }

    .section-title {
        font-size: 1.28rem;
        font-weight: 750;
        letter-spacing: -0.025em;
        margin-bottom: 0.8rem;
        color: #183326;
    }

    /* =====================================================
       SNAPSHOT
       ===================================================== */

    .snapshot {
        background: #ffffff;
        border: 1px solid #dce5de;
        border-radius: 20px;
        padding: 1.15rem;
        height: 100%;
    }

    .snapshot-label {
        color: #758078;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .snapshot-value {
        color: #19382a;
        font-size: 1.08rem;
        font-weight: 750;
        line-height: 1.25;
    }

    .snapshot-small {
        color: #78857d;
        font-size: 0.72rem;
        margin-top: 0.32rem;
    }

    .severity-card {
        background: #f3ead7;
        border-color: #e7d8b9;
    }

    .severity-value {
        color: #805e20;
        font-size: 1.35rem;
        font-weight: 800;
    }

    /* =====================================================
       METRIC STRIP
       ===================================================== */

    .metric-strip {
        display: flex;
        gap: 0.8rem;
        margin-top: 0.9rem;
    }

    .metric {
        flex: 1;
        background: #f7faf7;
        border: 1px solid #e1e9e3;
        border-radius: 14px;
        padding: 0.85rem;
    }

    .metric-label {
        font-size: 0.66rem;
        color: #7a857f;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .metric-value {
        margin-top: 0.22rem;
        font-size: 1.15rem;
        font-weight: 800;
        color: #234c37;
    }

    /* =====================================================
       FIELD REGION
       ===================================================== */

    .field-region {
        background: #eef4ee;
        border: 1px solid #d4e2d6;
        border-radius: 20px;
        padding: 1.15rem;
        margin-top: 1rem;
    }

    .field-region-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.9rem;
    }

    .field-region-title {
        font-size: 1rem;
        font-weight: 750;
        color: #214d35;
    }

    .region-badge {
        border-radius: 999px;
        padding: 0.35rem 0.65rem;
        background: #ffffff;
        border: 1px solid #d5e1d6;
        color: #57705f;
        font-size: 0.68rem;
        font-weight: 700;
    }

    .field-grid {
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        gap: 5px;
        padding: 0.7rem;
        background: #dfeadf;
        border-radius: 14px;
    }

    .field-cell {
        height: 30px;
        border-radius: 5px;
        background: #a9c5a7;
    }

    .field-cell-soft {
        background: #bfd2ba;
    }

    .field-cell-focus {
        background: #b66b4e;
        box-shadow: 0 0 0 2px rgba(182,107,78,0.18);
    }

    .field-cell-focus-light {
        background: #cf9474;
    }

    .region-note {
        color: #627067;
        font-size: 0.74rem;
        margin-top: 0.65rem;
        line-height: 1.5;
    }

    /* =====================================================
       INSPECTION WORKSPACE
       ===================================================== */

    .workspace {
        background: #ffffff;
        border: 1px solid #dce5de;
        border-radius: 20px;
        padding: 1.15rem;
    }

    .workspace-title {
        font-size: 0.95rem;
        font-weight: 750;
        color: #244735;
        margin-bottom: 0.75rem;
    }

    .image-caption {
        color: #78847d;
        font-size: 0.7rem;
        margin-top: 0.45rem;
    }

    /* =====================================================
       ACTION PANEL
       ===================================================== */

    .action-panel {
        background: #183f2d;
        border-radius: 20px;
        padding: 1.25rem;
        color: white;
        margin-top: 1rem;
    }

    .action-kicker {
        color: rgba(255,255,255,0.62);
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
    }

    .action-title {
        color: white;
        font-size: 1.12rem;
        font-weight: 750;
        margin-top: 0.3rem;
    }

    .action-text {
        color: rgba(255,255,255,0.75);
        font-size: 0.78rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }

    /* =====================================================
       INSPECTION STEPS
       ===================================================== */

    .process-box {
        background: #ffffff;
        border: 1px solid #dce5de;
        border-radius: 18px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .process-step {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.52rem 0;
        border-bottom: 1px solid #edf1ee;
    }

    .process-step:last-child {
        border-bottom: none;
    }

    .step-icon {
        width: 27px;
        height: 27px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 800;
        flex-shrink: 0;
    }

    .step-done {
        background: #dceee2;
        color: #287047;
    }

    .step-active {
        background: #e8dcbf;
        color: #806122;
    }

    .step-waiting {
        background: #edf1ee;
        color: #9aa39d;
    }

    .step-text {
        font-size: 0.78rem;
        color: #4f6057;
    }

    .step-active-text {
        color: #244735;
        font-weight: 700;
    }

    /* =====================================================
       NOTICE
       ===================================================== */

    .prototype-note {
        background: #fffaf0;
        border: 1px solid #eadfc7;
        border-radius: 15px;
        padding: 0.9rem 1rem;
        color: #756442;
        font-size: 0.73rem;
        line-height: 1.55;
        margin-top: 1rem;
    }

    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #87928b;
        font-size: 0.68rem;
        margin-top: 2.4rem;
        padding-top: 1rem;
        border-top: 1px solid #dce5de;
    }

    /* =====================================================
       STREAMLIT BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #1f6f4a;
        background: #1f6f4a;
        color: white;
        font-weight: 700;
        min-height: 2.8rem;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #155239;
        border-color: #155239;
        color: white;
    }

    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #bfd0c4;
        border-radius: 17px;
        padding: 0.35rem;
    }

    /* =====================================================
       HIDE STREAMLIT BRANDING
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_disease_name(name):
    """Make model class names readable."""

    if not name:
        return "Unknown"

    name = str(name)

    name = name.replace("___", " — ")
    name = name.replace("__", " — ")
    name = name.replace("_", " ")

    return name.title()


def extract_crop_name(disease_name):
    """
    Extract a human-readable crop name from the YOLO class.

    Examples:
        Tomato___Late_blight -> Tomato
        Potato leaf late blight -> Potato
    """

    if not disease_name:
        return "Unknown"

    name = str(disease_name)

    if "___" in name:
        return name.split("___")[0].replace("_", " ").title()

    if " — " in name:
        return name.split(" — ")[0].strip().title()

    known_crops = [
        "Potato",
        "Tomato",
        "Apple",
        "Grape",
        "Corn",
        "Maize",
        "Peach",
        "Pepper",
        "Bell Pepper",
        "Strawberry",
        "Cherry",
        "Orange",
        "Soybean",
        "Rice",
        "Wheat",
    ]

    lower_name = name.lower()

    for crop in known_crops:
        if lower_name.startswith(crop.lower()):
            return crop

    return name.split()[0].title()


def create_disease_overlay(image_bgr, disease_mask):
    """
    Overlay the estimated disease region on the original image.
    """

    image = image_bgr.copy()

    if disease_mask is None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.asarray(disease_mask).astype(bool)

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    overlay = image.copy()

    # Warm earthy red/brown for affected region.
    overlay[mask] = [50, 80, 185]

    blended = cv2.addWeighted(
        image,
        0.60,
        overlay,
        0.40,
        0,
    )

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def create_leaf_overlay(image_bgr, leaf_mask):
    """
    Show the detected leaf region while dimming the surroundings.
    """

    image = image_bgr.copy()

    if leaf_mask is None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.asarray(leaf_mask).astype(bool)

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    result = (image * 0.28).astype(np.uint8)
    result[mask] = image[mask]

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def severity_display(severity):
    """
    Make severity label readable.
    """

    if not severity:
        return "Unknown"

    return str(severity).replace("_", " ").title()


def create_field_grid():
    """
    Visual field-region concept.

    IMPORTANT:
    This is NOT a GPS map and does NOT represent actual
    field coordinates. It is a visual representation of
    an inspection region for the prototype.
    """

    cells = []

    focus_positions = {
        19,
        20,
        27,
        28,
        29,
        35,
        36,
        37,
        44,
        45,
    }

    light_positions = {
        18,
        21,
        26,
        30,
        34,
        38,
        43,
        46,
    }

    for i in range(48):

        if i in focus_positions:
            cls = "field-cell field-cell-focus"

        elif i in light_positions:
            cls = "field-cell field-cell-focus-light"

        elif i % 5 == 0:
            cls = "field-cell field-cell-soft"

        else:
            cls = "field-cell"

        cells.append(f'<div class="{cls}"></div>')

    return "".join(cells)


def show_inspection_progress():

    progress_placeholder = st.empty()

    steps = [
        ("✓", "Image received", "done"),
        ("✓", "Crop region detected", "done"),
        ("✓", "Disease localized", "done"),
        ("●", "Estimating affected region", "active"),
        ("○", "Preparing field inspection report", "waiting"),
    ]

    def render(active_index):

        html = '<div class="process-box">'

        for index, (icon, text_value, state) in enumerate(steps):

            if index < active_index:
                icon = "✓"
                icon_class = "step-done"
                text_class = "step-text"

            elif index == active_index:
                icon = "●"
                icon_class = "step-active"
                text_class = "step-text step-active-text"

            else:
                icon = "○"
                icon_class = "step-waiting"
                text_class = "step-text"

            html += f"""
                <div class="process-step">
                    <div class="step-icon {icon_class}">
                        {icon}
                    </div>
                    <div class="{text_class}">
                        {text_value}
                    </div>
                </div>
            """

        html += "</div>"

        progress_placeholder.markdown(
            html,
            unsafe_allow_html=True,
        )

    render(0)
    time.sleep(0.35)

    render(1)
    time.sleep(0.35)

    render(2)
    time.sleep(0.35)

    render(3)

    return progress_placeholder


# =========================================================
# TOP NAVIGATION
# =========================================================

st.markdown(
    """
    <div class="topbar">

        <div class="brand">

            <div class="brand-mark">
                🌱
            </div>

            <div>
                <div class="brand-name">
                    Krishi Sahyog
                </div>

                <div class="brand-sub">
                    Crop field inspection
                </div>
            </div>

        </div>

        <div class="system-status">
            <span class="status-dot"></span>
            Inspection system ready
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="inspection-hero">

        <div class="hero-top">

            <div>

                <div class="hero-kicker">
                    Field Inspection
                </div>

                <div class="hero-title">
                    Check the health of your crop
                </div>

                <div class="hero-description">
                    Upload a crop image to inspect the detected disease,
                    estimated affected leaf area, and the region requiring attention.
                </div>

            </div>

            <div class="inspection-id">

                <div class="inspection-id-label">
                    Inspection mode
                </div>

                <div class="inspection-id-value">
                    Image-based scan
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# IMAGE INPUT
# =========================================================

st.markdown(
    '<div class="section-kicker">01 · Field sample</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Capture or upload the crop sample</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Crop sample",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# =========================================================
# WAITING STATE
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="workspace">

            <div class="workspace-title">
                No inspection sample yet
            </div>

            <div style="
                padding: 2.2rem 1rem;
                text-align: center;
                color: #748078;
            ">

                <div style="font-size: 2.4rem; margin-bottom: 0.7rem;">
                    🌾
                </div>

                <div style="
                    font-size: 1rem;
                    font-weight: 700;
                    color: #31503e;
                    margin-bottom: 0.35rem;
                ">
                    Ready for the next field sample
                </div>

                <div style="
                    font-size: 0.76rem;
                    max-width: 520px;
                    margin: auto;
                    line-height: 1.55;
                ">
                    Upload a clear image of a crop leaf.
                    Krishi Sahyog will inspect the visible leaf,
                    identify a disease class, and estimate the affected region.
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="prototype-note">
            <strong>Prototype note:</strong>
            The current inspection uses an experimental computer-vision pipeline.
            Severity is an estimated prototype indication and should not be treated
            as an agronomically validated diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# IMAGE AVAILABLE
# =========================================================

else:

    image = Image.open(uploaded_file).convert("RGB")

    image_rgb = np.array(image)

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # -----------------------------------------------------
    # SAMPLE PREVIEW
    # -----------------------------------------------------

    preview_col1, preview_col2 = st.columns(
        [1.6, 1],
        gap="large",
    )

    with preview_col1:

        st.markdown(
            '<div class="section-kicker">Field sample</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Sample ready for inspection</div>',
            unsafe_allow_html=True,
        )

        st.image(
            image_rgb,
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="image-caption">
                Source image · {image.width} × {image.height}px
            </div>
            """,
            unsafe_allow_html=True,
        )

    with preview_col2:

        st.markdown(
            '<div class="section-kicker">Inspection request</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="workspace">

                <div class="workspace-title">
                    Start field inspection
                </div>

                <div style="
                    font-size:0.77rem;
                    color:#65736a;
                    line-height:1.55;
                    margin-bottom:1rem;
                ">
                    The sample will be processed through the current
                    crop disease detection and affected-area estimation pipeline.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        analyse_button = st.button(
            "🌱  Start Field Inspection",
            use_container_width=True,
        )

        if not analyse_button:

            st.markdown(
                """
                <div style="
                    margin-top:0.9rem;
                    padding:0.75rem;
                    border-radius:12px;
                    background:#f4f7f4;
                    border:1px solid #e0e8e1;
                    color:#718078;
                    font-size:0.7rem;
                    line-height:1.5;
                ">
                    Use a clear image where the crop leaf is visible.
                    For this prototype, the field location and drone position
                    are not inferred from the image.
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None and analyse_button:

    st.markdown(
        '<div class="section-kicker">02 · Live inspection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Inspecting field sample</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    progress_placeholder = show_inspection_progress()

    try:

        with st.spinner("Processing crop sample..."):

            result = analyze_disease(image_bgr)

    except Exception as exc:

        st.error(
            "The field inspection could not be completed."
        )

        st.exception(exc)

        st.stop()

    # -----------------------------------------------------
    # COMPLETE PROGRESS
    # -----------------------------------------------------

    progress_placeholder.markdown(
        """
        <div class="process-box">

            <div class="process-step">
                <div class="step-icon step-done">✓</div>
                <div class="step-text">Image received</div>
            </div>

            <div class="process-step">
                <div class="step-icon step-done">✓</div>
                <div class="step-text">Crop region detected</div>
            </div>

            <div class="process-step">
                <div class="step-icon step-done">✓</div>
                <div class="step-text">Disease localized</div>
            </div>

            <div class="process-step">
                <div class="step-icon step-done">✓</div>
                <div class="step-text">Affected region estimated</div>
            </div>

            <div class="process-step">
                <div class="step-icon step-done">✓</div>
                <div class="step-text">
                    Field inspection report ready
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # VALIDATE RESULT
    # -----------------------------------------------------

    if not result or not result.get("success", False):

        st.error(
            "No usable inspection result was returned for this sample."
        )

        st.stop()

    # -----------------------------------------------------
    # EXTRACT RESULT
    # -----------------------------------------------------

    detection = result.get("detection", {})

    disease_name_raw = detection.get(
        "class_name",
        "Unknown",
    )

    disease_name = clean_disease_name(
        disease_name_raw
    )

    crop_name = extract_crop_name(
        disease_name_raw
    )

    confidence = float(
        detection.get(
            "confidence",
            0,
        )
    )

    affected_area = float(
        result.get(
            "affected_area_percent",
            0,
        )
    )

    disease_mask = result.get(
        "disease_mask"
    )

    leaf_mask = result.get(
        "leaf_mask"
    )

    # -----------------------------------------------------
    # SEVERITY
    # -----------------------------------------------------

    try:

        severity = classify_severity(
            affected_area
        )

    except Exception:

        if affected_area < 10:
            severity = "Low"

        elif affected_area < 25:
            severity = "Moderate"

        elif affected_area < 50:
            severity = "High"

        else:
            severity = "Severe"

    severity = severity_display(
        severity
    )

    # -----------------------------------------------------
    # INSPECTION ID
    # -----------------------------------------------------

    inspection_id = (
        "KS-"
        + time.strftime("%d%m")
        + "-"
        + time.strftime("%H%M%S")
    )

    # =====================================================
    # INSPECTION RESULT HERO
    # =====================================================

    st.markdown(
        '<div class="section-kicker">03 · Inspection result</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">
            Field health snapshot
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="inspection-hero">

            <div class="hero-top">

                <div>

                    <div class="hero-kicker">
                        Inspection complete
                    </div>

                    <div class="hero-title">
                        Attention required in inspected crop region
                    </div>

                    <div class="hero-description">
                        A disease signal was detected in the submitted
                        crop sample. The affected area shown below is an
                        experimental estimate from the current prototype pipeline.
                    </div>

                </div>

                <div class="inspection-id">

                    <div class="inspection-id-label">
                        Inspection ID
                    </div>

                    <div class="inspection-id-value">
                        {inspection_id}
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # SNAPSHOT CARDS
    # =====================================================

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:

        st.markdown(
            f"""
            <div class="snapshot">

                <div class="snapshot-label">
                    Crop
                </div>

                <div class="snapshot-value">
                    {crop_name}
                </div>

                <div class="snapshot-small">
                    Detected crop region
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="snapshot">

                <div class="snapshot-label">
                    Disease
                </div>

                <div class="snapshot-value">
                    {disease_name}
                </div>

                <div class="snapshot-small">
                    Model detection
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="snapshot severity-card">

                <div class="snapshot-label">
                    Severity indication
                </div>

                <div class="severity-value">
                    {severity}
                </div>

                <div class="snapshot-small">
                    Prototype estimate
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # METRICS
    # =====================================================

    st.markdown(
        f"""
        <div class="metric-strip">

            <div class="metric">

                <div class="metric-label">
                    Area requiring attention
                </div>

                <div class="metric-value">
                    {affected_area:.1f}%
                </div>

            </div>

            <div class="metric">

                <div class="metric-label">
                    Detection confidence
                </div>

                <div class="metric-value">
                    {confidence * 100:.1f}%
                </div>

            </div>

            <div class="metric">

                <div class="metric-label">
                    Inspection status
                </div>

                <div class="metric-value">
                    Complete
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # FIELD REGION CONCEPT
    # =====================================================

    st.markdown(
        '<div class="section-kicker">04 · Field response</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Region requiring attention</div>',
        unsafe_allow_html=True,
    )

    field_col1, field_col2 = st.columns(
        [1.25, 1],
        gap="large",
    )

    with field_col1:

        st.markdown(
            f"""
            <div class="field-region">

                <div class="field-region-head">

                    <div class="field-region-title">
                        Inspection zone
                    </div>

                    <div class="region-badge">
                        Visual prototype region
                    </div>

                </div>

                <div class="field-grid">
                    {create_field_grid()}
                </div>

                <div class="region-note">
                    The highlighted region represents the concept of a
                    field area that could receive attention from a future
                    drone-assisted workflow. It is a visual prototype only;
                    no GPS coordinates or real field position are inferred
                    from this image.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with field_col2:

        st.markdown(
            f"""
            <div class="action-panel">

                <div class="action-kicker">
                    Recommended inspection response
                </div>

                <div class="action-title">
                    Review the affected crop region
                </div>

                <div class="action-text">
                    The submitted sample shows approximately
                    <strong>{affected_area:.1f}%</strong>
                    affected leaf area according to the current prototype.
                    Review nearby plants before deciding on any treatment.
                </div>

                <div style="
                    margin-top:0.8rem;
                    padding-top:0.7rem;
                    border-top:1px solid rgba(255,255,255,0.14);
                    font-size:0.69rem;
                    color:rgba(255,255,255,0.60);
                ">
                    Future extension · connect this inspection zone
                    to drone waypoint planning and field-level records.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # VISUAL INSPECTION WORKSPACE
    # =====================================================

    st.markdown(
        '<div class="section-kicker">05 · Visual evidence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">What the inspection found</div>',
        unsafe_allow_html=True,
    )

    image_col1, image_col2 = st.columns(
        2,
        gap="large",
    )

    with image_col1:

        st.markdown(
            """
            <div class="workspace">

                <div class="workspace-title">
                    Captured crop sample
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            image_rgb,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="image-caption">
                Original field sample submitted for inspection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with image_col2:

        overlay_rgb = create_disease_overlay(
            image_bgr,
            disease_mask,
        )

        st.markdown(
            """
            <div class="workspace">

                <div class="workspace-title">
                    Estimated affected region
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            overlay_rgb,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="image-caption">
                Highlighted area represents the prototype's estimated
                affected region.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # DETECTED LEAF
    # =====================================================

    with st.expander(
        "View detected leaf region",
        expanded=False,
    ):

        leaf_rgb = create_leaf_overlay(
            image_bgr,
            leaf_mask,
        )

        st.image(
            leaf_rgb,
            use_container_width=True,
        )

        st.caption(
            "The highlighted region shows the leaf area identified "
            "by the current segmentation stage."
        )

    # =====================================================
    # INSPECTION SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-kicker">06 · Field report</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Inspection summary</div>',
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2 = st.columns(
        [1, 1],
        gap="large",
    )

    with summary_col1:

        st.markdown(
            f"""
            <div class="workspace">

                <div class="workspace-title">
                    Current finding
                </div>

                <div style="
                    color:#53635a;
                    font-size:0.78rem;
                    line-height:1.65;
                ">

                    <strong style="color:#284a36;">
                        {disease_name}
                    </strong>
                    was detected in the inspected crop sample.

                    <br><br>

                    The prototype estimates that approximately
                    <strong style="color:#284a36;">
                        {affected_area:.1f}%
                    </strong>
                    of the detected leaf area is affected.

                    <br><br>

                    The corresponding prototype severity indication is
                    <strong style="color:#284a36;">
                        {severity}
                    </strong>.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_col2:

        st.markdown(
            """
            <div class="workspace">

                <div class="workspace-title">
                    Before field action
                </div>

                <div style="
                    color:#53635a;
                    font-size:0.78rem;
                    line-height:1.65;
                ">

                    <div style="margin-bottom:0.55rem;">
                        • Inspect surrounding leaves and nearby plants.
                    </div>

                    <div style="margin-bottom:0.55rem;">
                        • Compare symptoms across the affected crop area.
                    </div>

                    <div style="margin-bottom:0.55rem;">
                        • Use local agricultural guidance before applying treatment.
                    </div>

                    <div>
                        • Record repeated detections for future field monitoring.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # PROTOTYPE LIMITATION
    # =====================================================

    st.markdown(
        """
        <div class="prototype-note">

            <strong>Prototype limitation:</strong>
            Krishi Sahyog currently provides an experimental disease detection
            and affected-area estimation workflow. The severity indication is
            not scientifically validated and should not be used as a standalone
            basis for pesticide application or crop-treatment decisions.

            The highlighted field region is also a UI concept for the future
            drone-assisted workflow, not a real GPS-mapped field location.

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Krishi Sahyog · Crop field inspection prototype · v0.1
    </div>
    """,
    unsafe_allow_html=True,
)