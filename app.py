import glob
import os

# ── Patch basicsr before importing gfpgan (works on Windows, Linux, Mac) ──────
def _patch_basicsr():
    patterns = [
        r"C:\Users\**\site-packages\basicsr\data\degradations.py",
        "/usr/local/lib/python*/dist-packages/basicsr/data/degradations.py",
        "/opt/conda/lib/python*/site-packages/basicsr/data/degradations.py",
    ]
    import site
    for sp in site.getsitepackages():
        patterns.append(os.path.join(sp, "basicsr", "data", "degradations.py"))

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if not os.path.exists(path):
                continue
            txt = open(path, encoding="utf-8").read()
            if "functional_tensor" in txt:
                open(path, "w", encoding="utf-8").write(txt.replace(
                    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                    "from torchvision.transforms.functional import rgb_to_grayscale",
                ))

_patch_basicsr()

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from gfpgan import GFPGANer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import io
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceRestore · GFPGAN",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e4dc;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(255,180,60,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(255,100,60,0.05) 0%, transparent 60%),
        #0a0a0f;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* Remove top padding */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #f5a623;
    border: 1px solid rgba(245,166,35,0.35);
    padding: 0.35rem 1.1rem;
    border-radius: 2rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 7vw, 5rem);
    font-weight: 800;
    line-height: 1.02;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #fff8f0 30%, #f5a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 0.85rem;
    color: rgba(232,228,220,0.45);
    letter-spacing: 0.06em;
    max-width: 460px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}

/* ── Upload zone ── */
.upload-wrapper {
    border: 1.5px dashed rgba(245,166,35,0.30);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    background: rgba(255,255,255,0.02);
    transition: border-color 0.2s;
    margin-bottom: 1.5rem;
}
.upload-wrapper:hover { border-color: rgba(245,166,35,0.65); }

/* Override Streamlit uploader */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(245,166,35,0.3) !important;
    border-radius: 14px !important;
    transition: all 0.25s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(245,166,35,0.7) !important;
    background: rgba(245,166,35,0.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: rgba(232,228,220,0.5) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    color: rgba(232,228,220,0.3) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stBaseButton-secondary"] {
    background: rgba(245,166,35,0.12) !important;
    border: 1px solid rgba(245,166,35,0.4) !important;
    color: #f5a623 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(245,166,35,0.22) !important;
    border-color: #f5a623 !important;
}

/* ── Image panels ── */
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(232,228,220,0.4);
    margin-bottom: 0.6rem;
}
.panel-label span {
    color: #f5a623;
    margin-right: 0.4rem;
}

[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stImage"] img {
    border-radius: 12px;
    width: 100%;
}

/* ── Metrics row ── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin: 2rem 0 1.5rem;
}
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f5a623, transparent);
}
.metric-label {
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(232,228,220,0.35);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff8f0;
    line-height: 1;
}
.metric-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #f5a623;
    margin-left: 0.3rem;
}
.metric-delta {
    font-size: 0.72rem;
    color: rgba(232,228,220,0.4);
    margin-top: 0.3rem;
}
.delta-good { color: #6fcf97; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #f5a623 0%, #e8621a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    margin-top: 1rem !important;
}
[data-testid="stDownloadButton"] button:hover { opacity: 0.88 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #f5a623 !important; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245,166,35,0.2), transparent);
    margin: 2rem 0;
}

/* ── Info strip ── */
.info-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1rem 0 0.5rem;
    flex-wrap: wrap;
}
.info-item {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(232,228,220,0.3);
}
.info-item strong { color: rgba(232,228,220,0.6); }

/* ── Status tag ── */
.status-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem;
    border-radius: 2rem;
    margin-bottom: 1.5rem;
}
.status-tag.gpu {
    background: rgba(111,207,151,0.1);
    border: 1px solid rgba(111,207,151,0.25);
    color: #6fcf97;
}
.status-tag.cpu {
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.25);
    color: #f5a623;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
.gpu .dot { background: #6fcf97; }
.cpu .dot { background: #f5a623; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 3rem 0 1rem;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: rgba(232,228,220,0.2);
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def restore_image(restorer, pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, _, restored_bgr = restorer.enhance(
        bgr, has_aligned=False, only_center_face=False, paste_back=True
    )
    return Image.fromarray(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ Powered by GFPGAN v1.3</div>
    <div class="hero-title">Face Restoration</div>
    <div class="hero-sub">
        Blind face restoration using generative adversarial networks.<br>
        Upload a degraded photo — get a sharp, high-quality result.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Device status ──────────────────────────────────────────────────────────────
is_gpu = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if is_gpu else "CPU"
tag_class = "gpu" if is_gpu else "cpu"
tag_icon = "◉" if is_gpu else "○"
tag_text = f"GPU · {device_name}" if is_gpu else "CPU · No GPU detected"

st.markdown(f"""
<div style="text-align:center">
    <div class="status-tag {tag_class}">
        <span class="dot"></span>
        {tag_icon} &nbsp;{tag_text}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model weights…"):
    restorer = load_model()

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ── Main flow ──────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="divider"></div>
    <div class="info-strip">
        <div class="info-item">Supports <strong>JPG · PNG · JPEG</strong></div>
        <div class="info-item">Model <strong>GFPGAN v1.3</strong></div>
        <div class="info-item">Dataset <strong>CelebA-HQ 256×256</strong></div>
        <div class="info-item">Auto face <strong>detect & align</strong></div>
    </div>
    """, unsafe_allow_html=True)

else:
    input_img = Image.open(uploaded).convert("RGB")
    w, h = input_img.size

    # ── Two-column layout ──────────────────────────────────────────────────────
    col_in, col_out = st.columns(2, gap="large")

    with col_in:
        st.markdown('<div class="panel-label"><span>01</span>Input Image</div>', unsafe_allow_html=True)
        st.image(input_img, use_column_width=True)
        st.markdown(f'<div class="metric-delta">{w} × {h} px &nbsp;·&nbsp; {uploaded.size // 1024} KB</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="panel-label"><span>02</span>Restored Output</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        result_placeholder.markdown(
            '<div style="aspect-ratio:1;background:rgba(255,255,255,0.03);'
            'border-radius:12px;border:1px dashed rgba(255,255,255,0.08);'
            'display:flex;align-items:center;justify-content:center;'
            'color:rgba(232,228,220,0.2);font-size:0.75rem;letter-spacing:0.1em;">'
            'Awaiting restoration…</div>',
            unsafe_allow_html=True
        )

    # ── Run restoration ────────────────────────────────────────────────────────
    with st.spinner("Restoring face…"):
        t0 = time.time()
        restored_img = restore_image(restorer, input_img)
        elapsed = time.time() - t0

    result_placeholder.image(restored_img, use_column_width=True)

    rw, rh = restored_img.size

    # ── Metrics ────────────────────────────────────────────────────────────────
    gt_arr = np.array(input_img.resize(restored_img.size))
    rs_arr = np.array(restored_img)
    p_val  = psnr(gt_arr, rs_arr, data_range=255)
    s_val  = ssim(gt_arr, rs_arr, data_range=255, channel_axis=2)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">PSNR Score</div>
            <div class="metric-value">{p_val:.1f}<span class="metric-unit">dB</span></div>
            <div class="metric-delta">Peak Signal-to-Noise Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">SSIM Score</div>
            <div class="metric-value">{s_val:.3f}</div>
            <div class="metric-delta">Structural Similarity Index</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Output Size</div>
            <div class="metric-value">{rw}<span class="metric-unit">px</span></div>
            <div class="metric-delta">{rw} × {rh} resolution</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Process Time</div>
            <div class="metric-value">{elapsed:.1f}<span class="metric-unit">s</span></div>
            <div class="metric-delta delta-good">✓ Completed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────────
    fname = uploaded.name.rsplit(".", 1)[0] + "_restored.png"
    st.download_button(
        label="↓  Download Restored Image",
        data=pil_to_bytes(restored_img),
        file_name=fname,
        mime="image/png",
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    GFPGAN v1.3 · CelebA-HQ Dataset · Built with Streamlit
</div>
""", unsafe_allow_html=True)