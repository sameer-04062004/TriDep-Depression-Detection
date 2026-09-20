import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import re
import gc
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
BUNDLE_DIR = MODEL_DIR / "demo_bundle"

FYP_MODEL_PATH = BUNDLE_DIR / "fusion_model.keras"
DATA_NPZ_PATH = BUNDLE_DIR / "demo_data.npz"
MERGED_CSV_PATH = MODEL_DIR / "merged_predictions.csv"

# ── Custom Focal Loss for Keras Model ─────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * ce)
    return loss_fn

print("Loading TriDep Multimodal Model (Keras)...")
fyp_model = None
if FYP_MODEL_PATH.exists():
    try:
        fyp_model = tf.keras.models.load_model(
            str(FYP_MODEL_PATH),
            custom_objects={"loss_fn": focal_loss()},
            compile=False
        )
        print(f"✅ TriDep Multimodal Model loaded ({fyp_model.count_params():,} parameters)")
    except Exception as e:
        print(f"⚠️ Failed to load TriDep model: {e}")
else:
    print(f"⚠️ Model file not found at {FYP_MODEL_PATH}")

print("Loading Precomputed Features (demo_data.npz)...")
A, F, T, Y, PIDS, pid_list = None, None, None, None, None, []
if DATA_NPZ_PATH.exists():
    try:
        data = np.load(str(DATA_NPZ_PATH), allow_pickle=True)
        A, F, T, Y, PIDS = data["A"], data["F"], data["T"], data["Y"], data["PIDS"]
        pid_list = [int(p) for p in PIDS]
        print(f"✅ Demo features loaded for {len(pid_list)} subjects (Audio: {A.shape}, Video: {F.shape}, Text: {T.shape})")
    except Exception as e:
        print(f"⚠️ Failed to load demo data: {e}")
else:
    print(f"⚠️ Demo data not found at {DATA_NPZ_PATH}")

print("Loading Ensemble Predictions (merged_predictions.csv)...")
merged = None
dev_pids = []
if MERGED_CSV_PATH.exists():
    try:
        merged = pd.read_csv(str(MERGED_CSV_PATH))
        dev_pids = [int(p) for p in merged["participant_id"].tolist()]
        print(f"✅ Dev set ensemble predictions loaded for {len(dev_pids)} subjects")
    except Exception as e:
        print(f"⚠️ Failed to load merged predictions: {e}")
else:
    print(f"⚠️ Merged predictions CSV not found at {MERGED_CSV_PATH}")

# ── Live Feature Extractors (Lazy Loaded) ──────────────────────────────────────
_sbert_model = None
_w2v_fe = None
_w2v_mdl = None

def get_sbert():
    global _sbert_model
    if _sbert_model is None:
        print("Initializing Sentence-BERT (all-mpnet-base-v2)...")
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _sbert_model

def get_wav2vec2():
    global _w2v_fe, _w2v_mdl
    if _w2v_fe is None or _w2v_mdl is None:
        print("Initializing Wav2Vec2 (facebook/wav2vec2-base-960h)...")
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        _w2v_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        _w2v_mdl = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()
    return _w2v_fe, _w2v_mdl

def text_to_vec(text: str) -> np.ndarray:
    sbert = get_sbert()
    clean_text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in clean_text.replace("?", ".").split(".") if s.strip()]
    if not sents:
        sents = [clean_text or "neutral utterance"]
    emb = sbert.encode(sents, show_progress_bar=False, convert_to_numpy=True)
    return emb.mean(0).astype("float32")

def audio_to_vec(audio_path: str) -> np.ndarray:
    import librosa
    fe, mdl = get_wav2vec2()
    SR, WIN, MAX_W = 16000, 16000 * 8, 20
    sp, sr = librosa.load(audio_path, sr=None, mono=True, duration=MAX_W * 8 + 30)
    if sr != SR:
        sp = librosa.resample(sp.astype("float32"), orig_sr=sr, target_sr=SR)
    wins = [sp[i:i + WIN] for i in range(0, len(sp) - WIN + 1, WIN)][:MAX_W]
    if not wins:
        return np.zeros(1536, dtype="float32")
    with torch.no_grad():
        inp = fe(wins, sampling_rate=SR, return_tensors="pt", padding=True)
        hid = mdl(inp.input_values).last_hidden_state
        emb = hid.mean(dim=1).cpu().numpy()
    return np.concatenate([emb.mean(0), emb.std(0)]).astype("float32")

def video_to_vec(video_path: str):
    # Py-feat AU extraction or zero vector fallback
    try:
        from feat import Detector
        det = Detector(device="cuda" if torch.cuda.is_available() else "cpu")
        fea = det.detect_video(video_path)
        au = fea.aus.dropna(how="all").reset_index(drop=True)
        if len(au) == 0:
            return np.zeros(20, "float32"), "⚠️ No face detected in video"
        COLS = [
            'AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r',
            'AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r',
            'AU04_c','AU12_c','AU15_c','AU23_c','AU28_c','AU45_c'
        ]
        MAP = {
            'AU01_r':'AU01','AU02_r':'AU02','AU04_r':'AU04','AU05_r':'AU05',
            'AU06_r':'AU06','AU09_r':'AU09','AU10_r':'AU10','AU12_r':'AU12',
            'AU14_r':'AU14','AU15_r':'AU15','AU17_r':'AU17','AU20_r':'AU20',
            'AU25_r':'AU25','AU26_r':'AU26','AU04_c':'AU04','AU12_c':'AU12',
            'AU15_c':'AU15','AU23_c':'AU23','AU28_c':'AU28','AU45_c':'AU43'
        }
        arr = np.zeros((len(au), 20), "float32")
        for i, tc in enumerate(COLS):
            pc = MAP.get(tc)
            if pc and pc in au.columns:
                arr[:, i] = au[pc].fillna(0).values.astype("float32")
        if arr.shape[0] > 1:
            arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-8)
        return arr.mean(0).astype("float32"), f"✅ Video: {len(au)} frames processed"
    except Exception as e:
        return np.zeros(20, "float32"), f"⚠️ Video processing skipped: {e}"

def parse_transcript(filepath: str, orig_name: str):
    if str(orig_name).lower().endswith(".csv"):
        try:
            df = pd.read_csv(filepath, header="infer", sep=None, engine="python")
            df.columns = [str(c).strip().lower() for c in df.columns]
            cols = list(df.columns)
            sc = next((c for c in cols if "speaker" in c), None)
            vc = next((c for c in cols if c in ("value", "text", "utterance", "transcript")), None)
            if sc and vc:
                parts = df[df[sc].astype(str).str.lower().str.contains("participant")][vc].dropna().tolist()
                text = " ".join(str(p).strip() for p in parts if str(p).strip())
                return text, f"✅ Parsed DAIC CSV transcript ({len(parts)} participant utterances)"
        except Exception:
            pass
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text, "✅ Plain text transcript loaded"

# ── UI Card Helpers ───────────────────────────────────────────────────────────
def make_card(label: str, prob: float, true_label=None):
    pred = prob >= 0.5
    color = "#e11d48" if pred else "#059669"
    bg = "#fff1f2" if pred else "#ecfdf5"
    border = "#fb7185" if pred else "#34d399"
    emoji = "🔴" if pred else "🟢"
    txt = "Depressed" if pred else "Not Depressed"
    conf = prob * 100 if pred else (1 - prob) * 100

    h = f"""
    <div style="
        background:{bg};
        border:2px solid {border};
        border-radius:16px;
        padding:1.2rem;
        box-shadow:0 6px 18px rgba(0,0,0,.08);
        text-align:center;
        height:100%;
        box-sizing:border-box;">

        <div style="font-size:.8rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">
            {label}
        </div>
        <div style="font-size:2.8rem; margin:0.2rem 0;">{emoji}</div>
        <div style="font-weight:800; color:{color}; font-size:1.4rem; margin-bottom:0.4rem;">
            {txt}
        </div>
        <div style="color:#334155; font-size:.9rem;">
            Probability: <b>{prob*100:.1f}%</b><br>
            Confidence: <b>{conf:.1f}%</b>
        </div>
    """
    if true_label is not None:
        actual_txt = "Depressed" if true_label == 1 else "Healthy"
        is_correct = (pred == (true_label == 1))
        badge_bg = "#dcfce7" if is_correct else "#fee2e2"
        badge_color = "#15803d" if is_correct else "#b91c1c"
        badge_icon = "✓" if is_correct else "✗"
        h += f"""
        <div style="margin-top:0.8rem; padding:4px 8px; border-radius:8px; background:{badge_bg}; color:{badge_color}; font-weight:700; font-size:0.8rem;">
            Ground Truth: {actual_txt} ({badge_icon})
        </div>
        """
    h += "</div>"
    return h

def make_final(or_pred: bool, true_label=None):
    color = "#dc2626" if or_pred else "#16a34a"
    border = "#f87171" if or_pred else "#4ade80"
    bg = "linear-gradient(135deg,#fef2f2,#fee2e2)" if or_pred else "linear-gradient(135deg,#f0fdf4,#dcfce7)"
    emoji = "🔴" if or_pred else "🟢"
    txt = "DEPRESSION RISK DETECTED" if or_pred else "LOW RISK / HEALTHY"

    h = f"""
    <div style="
        background:{bg};
        border:3px solid {border};
        border-radius:18px;
        padding:1.5rem;
        text-align:center;
        box-shadow:0 10px 25px rgba(0,0,0,.1);
        margin-top:1rem;">

        <div style="font-size:3.2rem; margin-bottom:0.3rem;">{emoji}</div>
        <div style="font-size:1.8rem; font-weight:900; color:{color}; letter-spacing:0.02em;">
            {txt}
        </div>
        <div style="color:#475569; font-weight:600; margin-top:0.4rem; font-size:0.95rem;">
            Final Screening Assessment — Multimodal Ensemble Decision
        </div>
    """
    if true_label is not None:
        correct = (or_pred == (true_label == 1))
        msg = "✅ Correct Diagnosis" if correct else "⚠️ Mismatch with Clinical Ground Truth"
        msg_color = "#15803d" if correct else "#b91c1c"
        h += f"""
        <div style="margin-top:0.8rem; font-weight:700; color:{msg_color}; font-size:1rem;">
            {msg} (Actual: {"Depressed" if true_label == 1 else "Healthy"})
        </div>
        """
    h += "</div>"
    return h

# ── Callback Functions ────────────────────────────────────────────────────────
def cb_known(chosen_id):
    if not chosen_id:
        return "", "", "", ""
    chosen_id = int(chosen_id)
    if chosen_id not in pid_list:
        return "<p>Subject not found</p>", "", "", ""

    idx = pid_list.index(chosen_id)
    tl = int(Y[idx])

    # Run inference on TriDep model
    if fyp_model is not None:
        fp = float(fyp_model.predict([A[idx:idx+1], F[idx:idx+1], T[idx:idx+1]], verbose=0).ravel()[0])
    else:
        fp = 0.5

    c1 = make_card("🎵 TriDep (Audio + Video + Text)", fp, tl)

    # Check dev set ensemble predictions
    row = merged[merged["participant_id"] == chosen_id] if merged is not None and chosen_id in dev_pids else None
    if row is not None and not row.empty:
        tp = float(row["text_prob"].values[0])
        c2 = make_card("📝 InducT-GCN (Text Graph)", tp, tl)
        or_r = (fp >= 0.5) or (tp >= 0.5)
        c3 = make_card("🔗 Combined OR Decision", 1.0 if or_r else 0.0, tl)
        fin = make_final(or_r, tl)
    else:
        c2 = """
        <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:16px; padding:1.5rem; text-align:center; color:#64748b; height:100%;">
            <b>📝 InducT-GCN</b><br><br>InducT-GCN evaluation was computed on the 35 Dev Set subjects.<br>TriDep Multimodal inference is active.
        </div>
        """
        c3 = make_card("🔗 Single-Model Decision", fp, tl)
        fin = make_final(fp >= 0.5, tl)

    return c1, c2, c3, fin

def cb_live(audio_file, video_file, text_file):
    try:
        if audio_file is None:
            return "❌ Error: Please upload an audio file (.wav or .mp3)", "", "", "", ""
        if text_file is None:
            return "❌ Error: Please upload a transcript file (.txt or .csv)", "", "", "", ""

        log = "🚀 Starting TriDep Multi-Modal Inference Pipeline...\n"
        audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
        text_path = text_file if isinstance(text_file, str) else text_file.name
        orig_name = Path(text_path).name

        raw_text, info = parse_transcript(text_path, orig_name)
        log += f"{info}\n"
        log += f"Preview: \"{raw_text[:120]}...\"\n\n"

        log += "• Extracting Text features (Sentence-BERT all-mpnet-base-v2)...\n"
        t_vec = text_to_vec(raw_text)
        log += f"  Text embedding shape: {t_vec.shape}\n"

        log += "• Extracting Audio features (Wav2Vec2 960h)...\n"
        a_vec = audio_to_vec(audio_path)
        log += f"  Audio embedding shape: {a_vec.shape}\n"

        if video_file:
            video_path = video_file if isinstance(video_file, str) else video_file.name
            log += "• Extracting Video Facial Action Units...\n"
            f_vec, vmsg = video_to_vec(video_path)
            log += f"  {vmsg}\n"
        else:
            f_vec = np.zeros(20, "float32")
            log += "• Video not provided — using zero vector for video branch.\n"

        log += "\n• Running TriDep Fusion Model Forward Pass...\n"
        fp = float(fyp_model.predict(
            [a_vec.reshape(1, -1), f_vec.reshape(1, -1), t_vec.reshape(1, -1)],
            verbose=0
        ).ravel()[0])
        log += f"✅ TriDep Multimodal Depression Probability: {fp * 100:.1f}%\n"

        pred_bool = (fp >= 0.5)
        c1 = make_card("🎵 TriDep (Audio + Video + Text)", fp)
        c2 = """
        <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:16px; padding:1.5rem; text-align:center; color:#64748b; height:100%;">
            <b>📝 InducT-GCN</b><br><br>GCN requires global vocabulary graph.<br>TriDep Multimodal model evaluated.
        </div>
        """
        c3 = make_card("🔗 TriDep Multimodal Result", fp)
        fin = make_final(pred_bool)

        return log, c1, c2, c3, fin

    except Exception as e:
        err = traceback.format_exc()
        return f"❌ Error occurred:\n{err}", "", "", "", ""

# ── Benchmark Performance Data ────────────────────────────────────────────────
perf_df = pd.DataFrame({
    'Model Configuration': [
        'FYP Audio Only (Wav2Vec2)',
        'FYP Video Only (FAUs)',
        'FYP Text Only (SBERT)',
        'TriDep Fused (Audio + Video + Text)',
        'InducT-GCN Text Graph Model',
        'Combined Ensemble (OR-Decision Vote)'
    ],
    'Accuracy': [0.513, 0.481, 0.693, 0.714, 0.800, 0.800],
    'Precision (Dep)': [0.300, 0.281, 0.482, 0.600, 0.667, 0.647],
    'Recall (Dep)': [0.482, 0.482, 0.482, 0.500, 0.833, 0.917],
    'F1-Score (Dep)': [0.370, 0.355, 0.482, 0.545, 0.741, 0.759],
    'F1-Score (Weighted)': [0.534, 0.504, 0.693, 0.707, 0.804, 0.805],
})

# ── Gradio Application Layout ─────────────────────────────────────────────────
choices_list = [str(p) for p in (dev_pids if dev_pids else pid_list)]
default_val = choices_list[0] if choices_list else None

with gr.Blocks(title="TriDep — Multimodal Depression Screening") as demo:
    gr.HTML("""
    <div style="
        background:linear-gradient(135deg,#1e3a8a,#3b82f6,#6366f1);
        padding:26px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 12px 30px rgba(30,58,138,0.25);
        margin-bottom:18px;">

        <div style="font-size:2.2rem; font-weight:900; color:white; letter-spacing:0.02em;">
            🧠 TriDep — Multi-Modal Depression Detection Framework
        </div>
        <div style="color:#e0e7ff; font-size:1.05rem; font-weight:600; margin-top:6px; letter-spacing:0.05em; text-transform:uppercase;">
            Audio (Wav2Vec2) · Video (Facial Action Units) · Text (SBERT & InducT-GCN) · Ensemble Decision
        </div>
    </div>
    """)

    with gr.Tab("🔬 Known Participant Screening"):
        gr.Markdown("Select a DAIC-WOZ participant ID to run screening across the multi-modal fusion network and graph model:")
        with gr.Row():
            sub_dd = gr.Dropdown(choices=choices_list, value=default_val, label="Participant ID", scale=3)
            btn1 = gr.Button("🔬 Run Screening", variant="primary", scale=1)

        with gr.Row():
            o1a = gr.HTML()
            o1b = gr.HTML()
            o1c = gr.HTML()
        o1f = gr.HTML()

        btn1.click(cb_known, inputs=[sub_dd], outputs=[o1a, o1b, o1c, o1f])
        demo.load(cb_known, inputs=[sub_dd], outputs=[o1a, o1b, o1c, o1f])

    with gr.Tab("📤 Live Upload & Testing"):
        gr.Markdown("Upload audio and interview transcript for real-time feature extraction and classification:")
        with gr.Row():
            in_au = gr.File(label="🎵 Audio Recording (.wav, .mp3)", file_types=[".wav", ".mp3", ".ogg", ".m4a"])
            in_tx = gr.File(label="📝 Interview Transcript (.txt, .csv)", file_types=[".txt", ".csv"])
            in_vi = gr.File(label="🎥 Video (Optional, .mp4)", file_types=[".mp4", ".avi", ".mov"])

        btn2 = gr.Button("🔬 Run Live Screening", variant="primary")
        o_log = gr.Textbox(label="Inference & Feature Extraction Log", lines=8, interactive=False)

        with gr.Row():
            o2a = gr.HTML()
            o2b = gr.HTML()
            o2c = gr.HTML()
        o2f = gr.HTML()

        btn2.click(cb_live, inputs=[in_au, in_vi, in_tx], outputs=[o_log, o2a, o2b, o2c, o2f])

    with gr.Tab("📊 Model Performance & Benchmarks"):
        gr.Markdown("### Clinical Benchmark Comparison — DAIC-WOZ Development Set")
        gr.Dataframe(value=perf_df, interactive=False)
        gr.Markdown("""
> **Key Finding:** The **Combined OR-Decision Ensemble** achieves **91.7% Recall** on depressed subjects, ensuring near-zero false negatives in initial psychiatric screening scenarios.
        """)

    gr.HTML("""
    <div style="
        text-align:center;
        background:#f8fafc;
        border:1px solid #e2e8f0;
        padding:12px;
        border-radius:12px;
        font-size:0.85rem;
        color:#64748b;
        margin-top:16px;">
        ⚠️ <b>Research Prototype Disclaimer:</b> This system is designed for research and automated screening assistance only, not for definitive clinical diagnosis.
    </div>
    """)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TriDep Multimodal Depression Detection — Local Server")
    print("=" * 60 + "\n")
    demo.launch(inbrowser=True)

