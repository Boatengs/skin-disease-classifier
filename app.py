
import gradio as gr
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import urllib.request

# ── Class Names ────────────────────────────────────────────────
class_names = {
    0: "Melanoma (MEL)",
    1: "Melanocytic Nevi (NV)",
    2: "Basal Cell Carcinoma (BCC)",
    3: "Actinic Keratosis (AKIEC)",
    4: "Benign Keratosis (BKL)",
    5: "Dermatofibroma (DF)",
    6: "Vascular Lesion (VASC)"
}

risk_levels = {
    0: " HIGH RISK — Consult a dermatologist immediately",
    1: " LOW RISK — Benign mole, monitor for changes",
    2: " HIGH RISK — Consult a dermatologist immediately",
    3: " MEDIUM RISK — Precancerous, evaluation recommended",
    4: " LOW RISK — Benign lesion",
    5: " LOW RISK — Benign lesion",
    6: " LOW RISK — Benign lesion"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 7)
)
# Version 2 model: trained on a lesion-level split (see the GitHub README).
MODEL_PATH = "lesion_split_efficientnet_b0.pth"
MODEL_URL = "https://github.com/Boatengs/skin-disease-classifier/raw/main/models/lesion_split_efficientnet_b0.pth"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Both values were fitted on the validation set (results/summary.json in the repo).
TEMPERATURE = 1.3306          # temperature scaling for calibrated probabilities
MELANOMA_THRESHOLD = 0.0946   # flags about 90% of melanomas on validation
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return "Please upload an image", {}
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs / TEMPERATURE, dim=1)[0]
    results = {class_names[i]: float(probs[i]) for i in range(7)}
    top_idx = probs.argmax().item()
    mel_prob = float(probs[0])
    mel_flag = (
        f"**Melanoma screen:** flagged for review (melanoma probability {mel_prob*100:.1f}%, "
        f"threshold {MELANOMA_THRESHOLD*100:.1f}%)"
        if mel_prob >= MELANOMA_THRESHOLD
        else f"**Melanoma screen:** below the review threshold ({mel_prob*100:.1f}%)"
    )
    diagnosis = f"""
### {class_names[top_idx]}
**Confidence:** {float(probs[top_idx])*100:.1f}%
**Risk Level:** {risk_levels[top_idx]}

{mel_flag}

---
 For educational purposes only. Always consult a qualified dermatologist.
    """
    return diagnosis, results

# ── Custom CSS ─────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-card: #ffffff;
    --bg-elevated: #f1f5f9;
    --border: rgba(0,0,0,0.08);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-tertiary: #94a3b8;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --accent-dim: rgba(99,102,241,0.1);
    --danger: #ef4444;
    --warning: #f59e0b;
    --success: #10b981;
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
    --radius: 16px;
}

[data-theme="dark"] {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --bg-elevated: #334155;
    --border: rgba(255,255,255,0.08);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-tertiary: #475569;
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -1px rgba(0,0,0,0.2);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4), 0 4px 6px -2px rgba(0,0,0,0.3);
}

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease;
}

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 24px !important; }

/* Header */
.app-header {
    text-align: center;
    padding: 48px 24px 32px;
    background: linear-gradient(135deg, var(--accent-dim), transparent);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin-bottom: 28px;
}
.app-header h1 {
    font-size: 2.4em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 12px !important;
}
.app-header p {
    color: var(--text-secondary) !important;
    font-size: 1em !important;
    max-width: 600px;
    margin: 0 auto !important;
    line-height: 1.7 !important;
}

/* Badges */
.badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.badge {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--bg-elevated);
    color: var(--text-secondary);
}
.badge.accent { background: var(--accent-dim); color: var(--accent); border-color: rgba(99,102,241,0.25); }

/* Cards */
.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.2s ease !important;
}
.card:hover { box-shadow: var(--shadow-lg) !important; }

/* Theme toggle */
.theme-toggle {
    position: fixed !important;
    top: 20px !important;
    right: 20px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    cursor: pointer !important;
    font-size: 1.2em !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.2s !important;
    z-index: 1000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.theme-toggle:hover { transform: scale(1.1) !important; }

/* Upload area */
.upload-area {
    border: 2px dashed var(--accent) !important;
    border-radius: var(--radius) !important;
    background: var(--accent-dim) !important;
    transition: all 0.2s !important;
}
.upload-area:hover { border-color: var(--accent-hover) !important; }

/* Button */
button.primary {
    background: linear-gradient(135deg, var(--accent), #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95em !important;
    padding: 14px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    letter-spacing: 0.3px !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* Labels */
label { color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 0.82em !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }

/* Stats row */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    box-shadow: var(--shadow);
}
.stat-value { font-size: 1.8em; font-weight: 800; color: var(--accent); }
.stat-label { font-size: 0.75em; color: var(--text-tertiary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: var(--text-tertiary);
    font-size: 0.78em;
    border-top: 1px solid var(--border);
    margin-top: 32px;
}
"""

js = """
function toggleTheme() {
    const root = document.documentElement;
    const btn  = document.getElementById('theme-btn');
    const dark = root.getAttribute('data-theme') === 'dark';
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
    btn.textContent = dark ? '🌙' : '☀️';
    localStorage.setItem('skin-theme', dark ? 'light' : 'dark');
}
document.addEventListener('DOMContentLoaded', () => {
    const saved = localStorage.getItem('skin-theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = saved === 'dark' ? '☀️' : '🌙';
});
"""

# Gradio 6 takes css in launch(); earlier versions take it in Blocks().
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# The custom styles are built for a light page; the app's own toggle handles dark mode.
# Gradio follows the browser's dark preference, so drop its dark class on load.
FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"

with gr.Blocks(title=" Skin Disease Classifier",
               **({} if GRADIO_6 else {"css": css, "js": FORCE_LIGHT_JS})) as demo:

    gr.HTML(f"""
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">🌙</button>
    <script>{js}</script>

    <div class="app-header">
        <h1> Skin Disease Classifier</h1>
        <p>
            AI-powered dermoscopy image analysis across 7 skin lesion categories.
            Upload an image and get an instant classification with confidence scores.
        </p>
        <div class="badges">
            <span class="badge accent">EfficientNetB0</span>
            <span class="badge accent">76.5% Accuracy on Unseen Lesions</span>
            <span class="badge">HAM10000 Dataset</span>
            <span class="badge">7 Classes</span>
            <span class="badge">Lesion-Level Evaluation</span>
        </div>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">76.5%</div>
            <div class="stat-label">Accuracy on Unseen Lesions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">0.879</div>
            <div class="stat-label">Melanoma AUROC</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">7</div>
            <div class="stat-label">Disease Classes</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("###  Upload Image")
            input_image = gr.Image(
                label="Dermoscopy Image",
                height=280,
                elem_classes=["upload-area"]
            )
            classify_btn = gr.Button(
                " Classify Lesion",
                variant="primary"
            )
            gr.Markdown("""
            <div style="margin-top:12px; padding:12px; background:var(--bg-elevated); border-radius:10px; font-size:0.8em; color:var(--text-secondary);">
             <b>Tips:</b> Use clear, well-lit dermoscopy images for best results. 
            Supported formats: JPG, PNG, WEBP
            </div>
            """)

        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("###  Analysis Results")
            diagnosis_output = gr.Markdown(
                value="*Upload an image and click Classify to see results*"
            )
            confidence_output = gr.Label(
                label="Confidence Scores",
                num_top_classes=7
            )

    gr.HTML("""
    <div style="margin-top:28px; background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:24px; box-shadow:var(--shadow);">
        <h3 style="color:var(--text-primary); margin-bottom:16px; font-size:1em; font-weight:700;">📋 Disease Classes & Risk Levels</h3>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:10px;">
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #ef4444;">
                <b style="color:var(--text-primary);">Melanoma (MEL)</b>
                <span style="float:right; font-size:0.75em; color:#ef4444; font-weight:600;">🔴 HIGH RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Malignant skin cancer requiring immediate attention</p>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #10b981;">
                <b style="color:var(--text-primary);">Melanocytic Nevi (NV)</b>
                <span style="float:right; font-size:0.75em; color:#10b981; font-weight:600;">🟢 LOW RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Common benign mole, monitor for changes</p>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #ef4444;">
                <b style="color:var(--text-primary);">Basal Cell Carcinoma (BCC)</b>
                <span style="float:right; font-size:0.75em; color:#ef4444; font-weight:600;">🔴 HIGH RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Most common skin cancer, needs treatment</p>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #f59e0b;">
                <b style="color:var(--text-primary);">Actinic Keratosis (AKIEC)</b>
                <span style="float:right; font-size:0.75em; color:#f59e0b; font-weight:600;">🟡 MEDIUM RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Precancerous lesion, medical evaluation needed</p>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #10b981;">
                <b style="color:var(--text-primary);">Benign Keratosis (BKL)</b>
                <span style="float:right; font-size:0.75em; color:#10b981; font-weight:600;">🟢 LOW RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Non-cancerous skin growth</p>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #10b981;">
                <b style="color:var(--text-primary);">Dermatofibroma (DF)</b>
                <span style="float:right; font-size:0.75em; color:#10b981; font-weight:600;">🟢 LOW RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Benign fibrous nodule</p>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; border-left:3px solid #10b981;">
                <b style="color:var(--text-primary);">Vascular Lesion (VASC)</b>
                <span style="float:right; font-size:0.75em; color:#10b981; font-weight:600;">🟢 LOW RISK</span>
                <p style="color:var(--text-secondary); font-size:0.78em; margin-top:4px;">Benign vascular anomaly</p>
            </div>
        </div>
    </div>

    <div class="footer">
         This tool is for <b>educational purposes only</b> and is not a substitute for professional medical advice.
        Always consult a qualified dermatologist for diagnosis. •
        Built with EfficientNetB0 • Trained on HAM10000 •
        <a href="https://github.com/Boatengs/skin-disease-classifier" target="_blank" style="color:var(--accent);">GitHub</a>
    </div>
    """)

    classify_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[diagnosis_output, confidence_output]
    )

demo.launch(**({"css": css, "js": FORCE_LIGHT_JS} if GRADIO_6 else {}))
