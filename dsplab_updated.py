
import streamlit as st
import numpy as np
import librosa
from streamlit_webrtc import webrtc_streamer
import av
import joblib
model = joblib.load("alarm_model.pkl")

# 🎨 Page Configuration
st.set_page_config(page_title="🔔 Alarm Classifier", layout="centered", page_icon="🎧")

# 🌗 Toggle Dark Mode
theme = st.toggle("🌗 Toggle Dark Mode", value=False)
bg_color = "#1E1E1E" if theme else "#FFFFFF"
text_color = "#FFFFFF" if theme else "#000000"

st.markdown(
    f"""
    <style>
    .dark-container {{
        background-color: {bg_color};
        color: {text_color};
        padding: 30px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #0072B2;
        color: white;
    }}
    </style>
    <div class="dark-container">
    """,
    unsafe_allow_html=True
)

# 🔊 Header
st.markdown("<h1 style='text-align:center;'>🔊 Real Alarm Classifier</h1>", unsafe_allow_html=True)

# 🧠 How it works
with st.expander("🧠 How does this work?"):
    st.markdown("""
    - Upload a `.wav` file or use your microphone.
    - Detects types of alarms or background sounds.
    - Trained with real-world audio samples.
    """)

# 🎭 Sound Classes (must match labels used during training)
ALL_CLASSES = {
    "fire_alarm": "🔥",
    "buzzer": "🛎️",
    "smoke_detector": "🚨",
    "timer_alarm": "⏰"
    # Add more classes if you train on additional categories
}
CLASSES = list(ALL_CLASSES.keys())

# 🔍 Load the trained model
MODEL_PATH = "alarm_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    # If the model has a `classes_` attribute, ensure it's defined
    if not hasattr(model, "classes_"):
        st.error("Loaded model does not have `classes_`. Did you train & save correctly?")
        model = None
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please run `train_model.py` first.")
    model = None

# 🧪 Feature Extraction (must match training)
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    return np.hstack([mfccs_mean, chroma_mean, contrast_mean, zcr_mean])

# 📂 Tabs
tab1, tab2 = st.tabs(["📂 Upload File", "🎤 Microphone"])

# 📂 Upload Tab
with tab1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file and model:
        with st.spinner("Analyzing audio..."):
            y, sr = librosa.load(uploaded_file, sr=None, duration=5.0)
            features = extract_features(y, sr).reshape(1, -1)
            if features.shape[1] == model.n_features_in_:
                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                confidence = np.max(probs)
                st.success(f"{ALL_CLASSES[pred]} *{pred.replace('_', ' ').title()}* detected!")
                st.progress(confidence)
                st.toast(f"✅ {int(confidence*100)}% confidence", icon="📊")
            else:
                st.error("⚠️ Feature size mismatch.")

# 🎤 Live Mic Tab
with tab2:
    st.markdown("### 🎙️ Real-time Microphone Input")
    st.caption("Allow mic permissions in your browser.")

    if "live_prediction" not in st.session_state:
        st.session_state["live_prediction"] = "Waiting..."

    def audio_callback(frame: av.AudioFrame):
        audio = frame.to_ndarray(format="flt32")
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        sr = frame.sample_rate
        features = extract_features(audio, sr).reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            pred = model.predict(features)[0]
            st.session_state["live_prediction"] = f"{ALL_CLASSES.get(pred)} {pred.replace('_', ' ').title()}"
        else:
            st.session_state["live_prediction"] = "⚠️ Feature error"
        return frame

    if model:
        webrtc_streamer(
            key="live-audio",
            audio_frame_callback=audio_callback,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        st.info(f"🔔 Real-time: *{st.session_state['live_prediction']}*")

# 📘 Legend
with st.expander("📘 Sound Labels"):
    for name, emoji in ALL_CLASSES.items():
        st.markdown(f"- {emoji} **{name.replace('_', ' ').title()}**")

# 📍 Footer
st.markdown("---")
st.caption("🔧 Built with Streamlit · Uses real ML model for alarm classification")

st.markdown("</div>", unsafe_allow_html=True)
