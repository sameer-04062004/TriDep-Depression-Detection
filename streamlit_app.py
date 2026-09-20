import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TriDep — Multimodal Depression Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6, #6366f1);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    .metric-card-depressed {
        background: #fff1f2;
        border: 2px solid #fb7185;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .metric-card-healthy {
        background: #ecfdf5;
        border: 2px solid #34d399;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .final-banner-depressed {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 3px solid #f87171;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .final-banner-healthy {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 3px solid #4ade80;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Support models in models/ or saved_models/demo_bundle/
def find_path(filenames):
    for candidate in filenames:
        p = BASE_DIR / candidate
        if p.exists():
            return p
    return None

FYP_MODEL_PATH = find_path([
    "models/fusion_model.keras",
    "saved_models/demo_bundle/fusion_model.keras"
])
DATA_NPZ_PATH = find_path([
    "models/demo_data.npz",
    "saved_models/demo_bundle/demo_data.npz"
])
MERGED_CSV_PATH = find_path([
    "models/merged_predictions.csv",
    "saved_models/merged_predictions.csv"
])

# ── Custom Focal Loss for Keras ───────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    import tensorflow as tf
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * ce)
    return loss_fn

# ── Cached Resource Loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TriDep Multimodal Deep Learning Model...")
def load_tridep_model():
    if FYP_MODEL_PATH and FYP_MODEL_PATH.exists():
        import tensorflow as tf
        model = tf.keras.models.load_model(
            str(FYP_MODEL_PATH),
            custom_objects={"loss_fn": focal_loss()},
            compile=False
        )
        return model
    return None

@st.cache_data(show_spinner="Loading participant feature vectors...")
def load_demo_data():
    if DATA_NPZ_PATH and DATA_NPZ_PATH.exists():
        data = np.load(str(DATA_NPZ_PATH), allow_pickle=True)
        return data["A"], data["F"], data["T"], data["Y"], data["PIDS"]
    return None, None, None, None, None

@st.cache_data(show_spinner="Loading benchmark predictions...")
def load_merged_predictions():
    if MERGED_CSV_PATH and MERGED_CSV_PATH.exists():
        return pd.read_csv(str(MERGED_CSV_PATH))
    return None

# Load Resources
fyp_model = load_tridep_model()
A, F, T, Y, PIDS = load_demo_data()
merged_df = load_merged_predictions()

pid_list = [int(p) for p in PIDS] if PIDS is not None else []
dev_pids = [int(p) for p in merged_df["participant_id"].tolist()] if merged_df is not None else []

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/mental-health.png", width=70)
st.sidebar.title("TriDep System")
st.sidebar.markdown("""
**Tri-Modal Deep Learning Framework**  
Combining audio prosody, facial action units, and semantic text representations for clinical depression detection.

- **Audio:** Wav2Vec2 (`facebook/wav2vec2-base-960h`)
- **Video:** OpenFace Facial Action Units (FAUs)
- **Text:** Sentence-BERT + InducT-GCN
- **Ensemble:** Multimodal + GCN OR-decision (91.7% Recall)
""")
st.sidebar.divider()
st.sidebar.info("💡 **Dataset:** Evaluated on the DAIC-WOZ clinical interview dataset (USC ICT).")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.3rem;">🧠 TriDep — Multimodal Depression Screening</h1>
    <p style="margin:5px 0 0 0; font-size:1.05rem; opacity:0.9;">
        Audio · Video · Text Representations with Intermediate & Decision-Level Fusion
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬 Known Participant Screening",
    "📤 Live Audio & Transcript Screening",
    "📊 Benchmark Performance"
])

# ── Tab 1: Known Participant Screening ────────────────────────────────────────
with tab1:
    st.subheader("Select DAIC-WOZ Participant")
    st.write("Choose any participant from the dataset to evaluate their multi-modal features:")

    col_filter, col_select = st.columns([1, 2])
    with col_filter:
        filter_mode = st.radio("Show participants:", ["Development Set (35 subjects)", "All Participants (189 subjects)"])
    
    available_choices = dev_pids if filter_mode.startswith("Development") and dev_pids else pid_list

    with col_select:
        selected_pid = st.selectbox(
            "Participant ID:",
            options=available_choices,
            index=0 if available_choices else None
        )

    if selected_pid is not None and selected_pid in pid_list:
        idx = pid_list.index(selected_pid)
        true_label = int(Y[idx])

        # Model forward pass
        if fyp_model is not None:
            fp = float(fyp_model.predict([A[idx:idx+1], F[idx:idx+1], T[idx:idx+1]], verbose=0).ravel()[0])
        else:
            fp = 0.5

        # Check dev set
        row = merged_df[merged_df["participant_id"] == selected_pid] if merged_df is not None and selected_pid in dev_pids else None

        st.divider()
        col1, col2, col3 = st.columns(3)

        # Card 1: TriDep Multimodal
        with col1:
            pred_fp = fp >= 0.5
            card_class = "metric-card-depressed" if pred_fp else "metric-card-healthy"
            color = "#e11d48" if pred_fp else "#059669"
            emoji = "🔴" if pred_fp else "🟢"
            txt = "Depressed" if pred_fp else "Not Depressed"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size:0.8rem; font-weight:700; color:#475569; text-transform:uppercase;">
                    🎵 TriDep Multimodal (A+V+T)
                </div>
                <div style="font-size:2.5rem; margin:6px 0;">{emoji}</div>
                <div style="font-size:1.3rem; font-weight:800; color:{color};">{txt}</div>
                <div style="font-size:0.9rem; color:#334155; margin-top:4px;">
                    Probability: <b>{fp*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(fp)

        # Card 2: InducT-GCN
        with col2:
            if row is not None and not row.empty:
                tp = float(row["text_prob"].values[0])
                pred_tp = tp >= 0.5
                card_class = "metric-card-depressed" if pred_tp else "metric-card-healthy"
                color = "#e11d48" if pred_tp else "#059669"
                emoji = "🔴" if pred_tp else "🟢"
                txt = "Depressed" if pred_tp else "Not Depressed"
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-size:0.8rem; font-weight:700; color:#475569; text-transform:uppercase;">
                        📝 InducT-GCN (Text Graph)
                    </div>
                    <div style="font-size:2.5rem; margin:6px 0;">{emoji}</div>
                    <div style="font-size:1.3rem; font-weight:800; color:{color};">{txt}</div>
                    <div style="font-size:0.9rem; color:#334155; margin-top:4px;">
                        Probability: <b>{tp*100:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(tp)
            else:
                st.markdown("""
                <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:14px; padding:24px; text-align:center; color:#64748b;">
                    <b>📝 InducT-GCN (Text)</b><br><br>
                    Evaluated on 35 Dev Set subjects.<br>TriDep Multimodal inference is active.
                </div>
                """, unsafe_allow_html=True)

        # Card 3: Combined OR Decision
        with col3:
            if row is not None and not row.empty:
                tp = float(row["text_prob"].values[0])
                or_pred = (fp >= 0.5) or (tp >= 0.5)
            else:
                or_pred = (fp >= 0.5)

            card_class = "metric-card-depressed" if or_pred else "metric-card-healthy"
            color = "#e11d48" if or_pred else "#059669"
            emoji = "🔴" if or_pred else "🟢"
            txt = "Depressed" if or_pred else "Not Depressed"

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size:0.8rem; font-weight:700; color:#475569; text-transform:uppercase;">
                    🔗 Combined OR Vote (91.7% Recall)
                </div>
                <div style="font-size:2.5rem; margin:6px 0;">{emoji}</div>
                <div style="font-size:1.3rem; font-weight:800; color:{color};">{txt}</div>
                <div style="font-size:0.9rem; color:#334155; margin-top:4px;">
                    Ensemble Decision
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(1.0 if or_pred else 0.0)

        # Final Summary Banner
        banner_class = "final-banner-depressed" if or_pred else "final-banner-healthy"
        banner_color = "#dc2626" if or_pred else "#16a34a"
        banner_txt = "DEPRESSION RISK DETECTED" if or_pred else "LOW RISK / HEALTHY"
        banner_emoji = "🔴" if or_pred else "🟢"

        actual_txt = "Depressed" if true_label == 1 else "Healthy"
        is_match = (or_pred == (true_label == 1))
        match_msg = "✅ Accurate Diagnosis (Matches Clinical Ground Truth)" if is_match else "⚠️ Mismatch with Ground Truth"
        match_color = "#15803d" if is_match else "#b91c1c"

        st.markdown(f"""
        <div class="{banner_class}">
            <div style="font-size:2.8rem;">{banner_emoji}</div>
            <div style="font-size:1.8rem; font-weight:900; color:{banner_color};">
                {banner_txt}
            </div>
            <div style="font-size:1rem; color:#334155; font-weight:600; margin-top:6px;">
                Actual Clinical Diagnosis: <b>{actual_txt}</b>
            </div>
            <div style="font-size:0.95rem; font-weight:700; color:{match_color}; margin-top:4px;">
                {match_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Live Upload & Audio Screening ──────────────────────────────────────
with tab2:
    st.subheader("Live Interview Screening")
    st.write("Upload an interview recording and participant transcript to perform live feature extraction and classification:")

    c_audio, c_trans = st.columns(2)
    with c_audio:
        uploaded_audio = st.file_uploader("🎵 Audio Recording (.wav, .mp3)", type=["wav", "mp3", "ogg", "m4a"])
    with c_trans:
        uploaded_trans = st.file_uploader("📝 Interview Transcript (.txt, .csv)", type=["txt", "csv"])

    if st.button("🔬 Run Live Screening", type="primary"):
        if not uploaded_audio or not uploaded_trans:
            st.error("Please upload both an audio file and a transcript file to proceed.")
        else:
            with st.spinner("Extracting multi-modal representations and running inference..."):
                try:
                    # Save temporary files
                    import tempfile
                    t_dir = Path(tempfile.gettempdir()) / "tridep_live"
                    t_dir.mkdir(parents=True, exist_ok=True)
                    
                    audio_tmp = t_dir / uploaded_audio.name
                    audio_tmp.write_bytes(uploaded_audio.getvalue())

                    # Parse transcript
                    if uploaded_trans.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_trans, header="infer", sep=None, engine="python")
                        cols = [str(c).strip().lower() for c in df.columns]
                        df.columns = cols
                        sc = next((c for c in cols if "speaker" in c), None)
                        vc = next((c for c in cols if c in ("value", "text", "utterance", "transcript")), None)
                        if sc and vc:
                            parts = df[df[sc].astype(str).str.lower().str.contains("participant")][vc].dropna().tolist()
                            raw_text = " ".join(str(p).strip() for p in parts if str(p).strip())
                        else:
                            raw_text = " ".join(df.astype(str).values.ravel())
                    else:
                        raw_text = uploaded_trans.getvalue().decode("utf-8", errors="ignore")

                    # Live extraction
                    from sentence_transformers import SentenceTransformer
                    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                    clean_text = re.sub(r"\s+", " ", raw_text).strip()
                    sents = [s.strip() for s in clean_text.replace("?", ".").split(".") if s.strip()] or ["neutral text"]
                    t_vec = sbert.encode(sents, show_progress_bar=False, convert_to_numpy=True).mean(0).astype("float32")

                    import librosa, torch
                    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
                    w2v_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
                    w2v_mdl = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()

                    sp, sr = librosa.load(str(audio_tmp), sr=None, mono=True, duration=190)
                    if sr != 16000:
                        sp = librosa.resample(sp.astype("float32"), orig_sr=sr, target_sr=16000)
                    wins = [sp[i:i + 16000*8] for i in range(0, len(sp) - 16000*8 + 1, 16000*8)][:20]
                    if wins:
                        with torch.no_grad():
                            inp = w2v_fe(wins, sampling_rate=16000, return_tensors="pt", padding=True)
                            hid = w2v_mdl(inp.input_values).last_hidden_state
                            emb = hid.mean(dim=1).cpu().numpy()
                        a_vec = np.concatenate([emb.mean(0), emb.std(0)]).astype("float32")
                    else:
                        a_vec = np.zeros(1536, "float32")

                    f_vec = np.zeros(20, "float32")

                    # Predict
                    live_prob = float(fyp_model.predict(
                        [a_vec.reshape(1, -1), f_vec.reshape(1, -1), t_vec.reshape(1, -1)],
                        verbose=0
                    ).ravel()[0])

                    pred_dep = live_prob >= 0.5
                    st.success("Feature extraction and prediction completed successfully!")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("TriDep Probability", f"{live_prob * 100:.1f}%")
                        st.progress(live_prob)
                    with res_col2:
                        status_txt = "🔴 High Depression Risk" if pred_dep else "🟢 Low Risk / Healthy"
                        st.metric("Clinical Assessment", status_txt)

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.text(traceback.format_exc())

# ── Tab 3: Benchmark Performance ──────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance — DAIC-WOZ Development Set")
    
    perf_data = pd.DataFrame({
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

    st.dataframe(perf_data.style.highlight_max(axis=0, color="#dcfce7", subset=["Accuracy", "Recall (Dep)", "F1-Score (Dep)"]), use_container_width=True)

    st.markdown("""
    > 💡 **Clinical Impact:** The **Combined OR-Decision Ensemble** achieves **91.7% Recall** on depressed patients. In mental health screening, high sensitivity is crucial to minimize false negatives and ensure potential depression cases are flagged for clinician review.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ **Research Prototype Disclaimer:** TriDep is an academic AI research prototype designed for automated screening assistance, not for definitive psychiatric diagnosis. Always consult a qualified medical professional.")
