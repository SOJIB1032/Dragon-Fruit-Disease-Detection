import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Dragon Fruit AI · Disease Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# PREMIUM CSS DESIGN
# =============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f0a !important;
    color: #e8f0e9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(34,197,94,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(16,185,129,0.10) 0%, transparent 55%),
        #0a0f0a !important;
}

[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stSidebar"] {
    background: rgba(15, 25, 15, 0.95) !important;
    border-right: 1px solid rgba(34,197,94,0.12) !important;
}

.hero-wrapper {
    position: relative;
    text-align: center;
    padding: 60px 20px 40px;
}

.hero-eyebrow {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #4ade80;
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    padding: 6px 18px;
    border-radius: 100px;
    margin-bottom: 28px;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(42px, 6vw, 76px);
    font-weight: 800;
    line-height: 1.02;
    letter-spacing: -0.03em;
    color: #f0fdf4;
    margin: 0 0 20px 0;
}

.hero-title span {
    background: linear-gradient(135deg, #4ade80 0%, #34d399 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 17px;
    font-weight: 300;
    color: rgba(232,240,233,0.55);
    width: 100%;
    max-width: 520px;
    margin: 0 auto 48px;
    line-height: 1.65;
    letter-spacing: 0.01em;
    text-align: center;
    display: block;
}

.hero-divider {
    width: 1px;
    height: 60px;
    background: linear-gradient(to bottom, rgba(74,222,128,0.4), transparent);
    margin: 0 auto 48px;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 52px;
}

.stat-pill {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    min-width: 110px;
}

.stat-pill-num {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #4ade80;
    display: block;
}

.stat-pill-label {
    font-size: 11px;
    color: rgba(232,240,233,0.45);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
    display: block;
}

/* ── UPLOAD ZONE ── */
.upload-section-label {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(74,222,128,0.7);
    margin-bottom: 14px;
}

/* FIX: label লুকানো হয়েছে — "upload" দুইবার দেখানো বন্ধ */
[data-testid="stFileUploader"] label {
    display: none !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(74,222,128,0.25) !important;
    border-radius: 20px !important;
    padding: 32px !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,222,128,0.55) !important;
    background: rgba(74,222,128,0.03) !important;
}

/* ── RESULT CARD ── */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 36px;
    position: relative;
    overflow: hidden;
    height: 100%;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,222,128,0.4), transparent);
}

.result-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(232,240,233,0.4);
    margin-bottom: 8px;
}

.result-disease-name {
    font-family: 'Syne', sans-serif;
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 6px;
}

.result-healthy { color: #4ade80; }
.result-disease { color: #fb923c; }
.result-severe  { color: #f87171; }

.confidence-bar-wrap { margin: 20px 0 28px; }

.confidence-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: rgba(232,240,233,0.45);
    margin-bottom: 7px;
    letter-spacing: 0.04em;
}

.confidence-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 6px;
    border-radius: 100px;
    background: linear-gradient(90deg, #4ade80, #34d399);
}

.confidence-bar-fill.orange {
    background: linear-gradient(90deg, #fb923c, #f97316);
}

.confidence-bar-fill.red {
    background: linear-gradient(90deg, #f87171, #ef4444);
}

.info-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(232,240,233,0.35);
    margin: 28px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.symptom-chip {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    color: rgba(232,240,233,0.75);
    margin: 4px 4px 4px 0;
}

.cause-box {
    background: rgba(251,146,60,0.06);
    border: 1px solid rgba(251,146,60,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13.5px;
    color: rgba(232,240,233,0.65);
    line-height: 1.6;
    margin: 8px 0;
}

.treatment-box {
    background: rgba(74,222,128,0.05);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13.5px;
    color: rgba(232,240,233,0.65);
    line-height: 1.6;
    margin: 8px 0;
}

.healthy-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 13px;
    color: #4ade80;
    margin: 4px 0;
    font-weight: 500;
}

.chart-section-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin: 44px 0 20px;
}

.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f0fdf4;
    letter-spacing: -0.01em;
}

.chart-subtitle {
    font-size: 13px;
    color: rgba(232,240,233,0.35);
}

[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

.section-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(232,240,233,0.35);
    margin-bottom: 10px;
}

.stButton > button {
    background: rgba(74,222,128,0.12) !important;
    border: 1px solid rgba(74,222,128,0.3) !important;
    color: #4ade80 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: rgba(74,222,128,0.2) !important;
    border-color: rgba(74,222,128,0.5) !important;
    transform: translateY(-1px) !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif !important;
    color: #f0fdf4 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: rgba(232,240,233,0.65) !important;
    font-size: 15px;
    line-height: 1.7;
}

div[data-testid="column"] { padding: 0 10px !important; }

.footer {
    text-align: center;
    padding: 48px 0 32px;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 64px;
}

.footer-logo {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: rgba(74,222,128,0.6);
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.footer-text {
    font-size: 12px;
    color: rgba(232,240,233,0.25);
    letter-spacing: 0.05em;
}

[data-testid="stSpinner"] { color: #4ade80 !important; }

</style>
""", unsafe_allow_html=True)


# =============================
# HERO
# =============================
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">🔬 AI-Powered Plant Pathology</div>
    <h1 class="hero-title">Dragon Fruit<br><span>Disease Scanner</span></h1>
    <p class="hero-sub">
        Upload a leaf or stem image. Our deep learning model identifies diseases
        in seconds with clinical-grade precision.
    </p>
    <div class="hero-divider"></div>
    <div class="stats-row">
        <div class="stat-pill">
            <span class="stat-pill-num">4</span>
            <span class="stat-pill-label">Classes</span>
        </div>
        <div class="stat-pill">
            <span class="stat-pill-num">224px</span>
            <span class="stat-pill-label">Input Size</span>
        </div>
        <div class="stat-pill">
            <span class="stat-pill-num">MNv2</span>
            <span class="stat-pill-label">Architecture</span>
        </div>
        <div class="stat-pill">
            <span class="stat-pill-num">Real-time</span>
            <span class="stat-pill-label">Inference</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_disease_model():
    # FIX: model file না থাকলে clear error দেখাবে
    import os
    if not os.path.exists("models/best_model.keras"):
        st.error("⚠️ Model file পাওয়া যায়নি! 'models/best_model.keras' path এ রাখুন।")
        st.stop()
    return load_model("models/best_model.keras",   compile=False,
    safe_mode=False)

model = load_disease_model()

CLASS_NAMES = ["Brown stem spot", "Healthy", "Soft Rot", "Stem Cenker"]

DISEASE_DATA = {
    "Brown stem spot": {
        "color_class": "result-severe",
        "bar_class": "red",
        "emoji": "🔴",
        "symptoms": ["Dark sunken lesions", "Brown/black patches", "Stem damage", "Fruit quality loss"],
        "cause": "Caused by Colletotrichum gloeosporioides fungus. Thrives in high humidity (>80%) and warm temperatures.",
        "treatment": "Apply copper-based fungicide (e.g., Bordeaux mixture). Remove and destroy infected plant parts immediately. Improve air circulation.",
    },
    "Healthy": {
        "color_class": "result-healthy",
        "bar_class": "",
        "emoji": "🟢",
        "symptoms": [],
        "cause": "",
        "treatment": "",
    },
    "Soft Rot": {
        "color_class": "result-disease",
        "bar_class": "orange",
        "emoji": "🟠",
        "symptoms": ["Soft mushy stem", "Dark discoloration", "Foul odor", "Structural collapse"],
        "cause": "Overwatering combined with poor soil drainage. Often triggered by Phytophthora or Pythium species.",
        "treatment": "Reduce irrigation immediately. Apply systemic fungicide. Ensure raised beds or well-draining substrate.",
    },
    "Stem Cenker": {
        "color_class": "result-disease",
        "bar_class": "orange",
        "emoji": "🟤",
        "symptoms": ["Circular brown lesions", "Yellow halos", "Leaf drying", "Premature drop"],
        "cause": "Fungal infection (Dothiorella spp.) accelerated by humid, wet conditions and poor nutrition.",
        "treatment": "Spray mancozeb or chlorothalonil fungicide. Remove infected leaves. Avoid overhead irrigation.",
    }
}


# =============================
# PREPROCESS
# =============================
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    arr = np.array(img.convert("RGB"))  # FIX: RGBA/grayscale সব handle করবে
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr.astype(np.float32))  # FIX: float32 নিশ্চিত করা


# =============================
# UPLOAD AREA
# =============================
st.markdown('<div class="upload-section-label">Upload Sample</div>', unsafe_allow_html=True)

# FIX: label_visibility="hidden" — label লুকানো, "upload" একবারই দেখাবে
uploaded = st.file_uploader(
    "Drop your image here — JPG, PNG, JPEG supported",
    type=["jpg", "png", "jpeg"],
    label_visibility="hidden"  # ← এটাই ছিল মূল সমস্যা
)

# =============================
# RESULTS
# =============================
if uploaded:
    # FIX: try-except দিয়ে image open error handle করা
    try:
        img = Image.open(uploaded)
    except Exception as e:
        st.error(f"Image open করতে সমস্যা হয়েছে: {e}")
        st.stop()

    img_array = preprocess(img)

    with st.spinner("Running inference…"):
        try:
            pred = model.predict(img_array)
        except Exception as e:
            st.error(f"Model prediction এ সমস্যা: {e}")
            st.stop()

    idx = int(np.argmax(pred))
    class_name = CLASS_NAMES[idx]
    confidence = float(np.max(pred)) * 100
    data = DISEASE_DATA[class_name]

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    # ── Confidence 90% এর নিচে হলে "Not Predicted" দেখাও ──
    if confidence < 95.0:
        col_img, col_warn = st.columns([1, 1], gap="large")
        with col_img:
            st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
            st.image(Image.open(uploaded), use_container_width=True)
        with col_warn:
            st.markdown(f"""
            <div class="result-card" style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; min-height:300px;">
                <div style="font-size:64px; margin-bottom:16px;">⚠️</div>
                <div class="result-label">Diagnosis Result</div>
                <div class="result-disease-name" style="color:#facc15; font-size:28px; margin:12px 0;">
                    This image is not Predictable
                </div>
                <div style="color:rgba(232,240,233,0.45); font-size:14px; line-height:1.6; margin-top:8px;">
                    Confidence score is too low<br>
                    <span style="color:#facc15; font-weight:600;">{confidence:.1f}%</span>
                    &nbsp;(minimum required: 90%)
                </div>
                <div style="margin-top:24px; background:rgba(250,204,21,0.07); border:1px solid rgba(250,204,21,0.2); border-radius:12px; padding:14px 18px; font-size:13px; color:rgba(232,240,233,0.6);">
                    💡 Please upload a clearer image of the dragon fruit stem or leaf for accurate detection.
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.stop()

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

    with col_res:
        color_cls = data["color_class"]
        bar_cls   = data["bar_class"]

        symptom_chips = "".join(
            [f'<span class="symptom-chip">{s}</span>' for s in data["symptoms"]]
        )
        cause_html = (
            f'<div class="cause-box">⚡ <strong>Cause:</strong> {data["cause"]}</div>'
            if data["cause"] else ""
        )
        treat_html = (
            f'<div class="treatment-box">💊 <strong>Treatment:</strong> {data["treatment"]}</div>'
            if data["treatment"] else ""
        )

        # ── Card header + confidence bar ──
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Diagnosis Result</div>
            <div class="result-disease-name {color_cls}">{data['emoji']} {class_name}</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label-row">
                    <span>Confidence Score</span>
                    <span>{confidence:.1f}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill {bar_cls}" style="width:{confidence:.1f}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Disease info — আলাদা করে render করলে Streamlit sanitize করবে না ──
        if class_name == "Healthy":
            st.markdown("""
            <div class="healthy-badge">✅ No disease detected</div><br>
            <div class="healthy-badge">🌿 Leaf structure normal</div><br>
            <div class="healthy-badge">💧 No fungal or bacterial markers</div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="info-section-title">Recommendation</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="treatment-box">
                Maintain regular irrigation schedule. Ensure adequate sunlight (6–8 hrs/day).
                Monitor for early signs of stress every 2 weeks.
            </div>
            """, unsafe_allow_html=True)

        else:
            # Symptoms
            st.markdown('<div class="info-section-title">Symptoms</div>', unsafe_allow_html=True)
            st.markdown(symptom_chips, unsafe_allow_html=True)

            # Cause
            st.markdown('<div class="info-section-title">Cause</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="cause-box">
                ⚡ <strong>Cause:</strong> {data["cause"]}
            </div>
            """, unsafe_allow_html=True)

            # Treatment
            st.markdown('<div class="info-section-title">Treatment</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="treatment-box">
                💊 <strong>Treatment:</strong> {data["treatment"]}
            </div>
            """, unsafe_allow_html=True)

    # ── PROBABILITY CHART ──
    st.markdown("""
    <div class="chart-section-header">
        <span class="chart-title">Prediction Distribution</span>
        <span class="chart-subtitle">Softmax probability across all classes</span>
    </div>
    """, unsafe_allow_html=True)

    probs = pred[0]
    colors_map = {
        "Brown stem spot": "#f87171",
        "Healthy":         "#4ade80",
        "Soft Rot":        "#fb923c",
        "Stem Cenker":     "#fbbf24",
    }
    bar_colors = [colors_map[c] for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor('#111a11')
    ax.set_facecolor('#111a11')

    bars = ax.barh(CLASS_NAMES, probs, color=bar_colors, height=0.52, edgecolor='none')

    for bar, prob in zip(bars, probs):
        ax.text(
            prob + 0.008, bar.get_y() + bar.get_height() / 2,
            f'{prob*100:.1f}%', va='center', ha='left',
            color='#a0b8a2', fontsize=11, fontfamily='monospace'
        )

    ax.set_xlim(0, 1.18)
    ax.tick_params(colors='#8a9e8a', labelsize=11)
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#2a3a2a')
    ax.xaxis.set_visible(False)

    for label in ax.get_yticklabels():
        label.set_fontfamily('monospace')
        label.set_color('#b0c4b0')

    ax.invert_yaxis()
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)


# =============================
# FOOTER
# =============================
st.markdown("""
<div class="footer">
    <div class="footer-logo">DRAGON FRUIT · AI</div>
    <div class="footer-text">
        Powered by MobileNetV2 · TensorFlow · Streamlit &nbsp;·&nbsp;
        Deep Learning Plant Pathology System
    </div>
</div>
""", unsafe_allow_html=True)