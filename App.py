import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.attack-badge  {
    background:#ff4b4b;
    color:white;
    padding:8px 20px;
    border-radius:20px;
    font-size:18px;
    font-weight:bold;
}

.normal-badge  {
    background:#00c853;
    color:white;
    padding:8px 20px;
    border-radius:20px;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'duration','protocol_type','service','flag',
    'src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
    'num_failed_logins','logged_in','num_compromised','root_shell',
    'su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

# ─────────────────────────────────────────────
# LOAD MODELS (FIXED)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "ids_model_final.pkl")
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
        encoder_path = os.path.join(BASE_DIR, "label_encoders.pkl")

        st.write("📂 Current Directory:", BASE_DIR)
        st.write("📄 Files Found:", os.listdir(BASE_DIR))

        # Check files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Missing: {model_path}")

        if not os.path.exists(scaler_path):
            st.error(f"❌ Missing: {scaler_path}")

        if not os.path.exists(encoder_path):
            st.error(f"❌ Missing: {encoder_path}")

        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoder_path)

        st.success("✅ Models loaded successfully!")

        return model, scaler, encoders, True

    except Exception as e:
        st.error(f"🚨 MODEL LOADING ERROR: {e}")
        return None, None, None, False


model, scaler, label_encoders, models_loaded = load_models()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def traffic_to_image(features_scaled: np.ndarray) -> np.ndarray:
    vec = np.array(features_scaled, dtype=float).flatten()

    if vec.size < 42:
        vec = np.pad(vec, (0, 42 - vec.size))

    vec = vec[:42]

    mn, mx = vec.min(), vec.max()

    if mx - mn > 0:
        vec = (vec - mn) / (mx - mn)

    return vec.reshape(6, 7)


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ IDS Dashboard")
    st.markdown("---")

    if models_loaded:
        st.success("✅ Models loaded")
    else:
        st.warning("⚠️ Model loading failed")

    st.markdown("### Detection Engines")
    st.markdown("- 🌲 Random Forest")
    st.markdown("- 🧠 CNN")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛡️ Network Intrusion Detection System")
st.markdown("### Machine Learning + CNN Image Processing based IDS")
st.markdown("---")

# ─────────────────────────────────────────────
# SIMPLE DETECTION UI
# ─────────────────────────────────────────────
st.header("🔍 ML-Based Intrusion Detection")

col1, col2, col3 = st.columns(3)

with col1:
    duration = st.number_input("Duration", 0, 100000, 0)
    src_bytes = st.number_input("Source Bytes", 0, 1000000, 0)

with col2:
    protocol_type = st.selectbox("Protocol", ['tcp', 'udp', 'icmp'])
    service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'ssh'])

with col3:
    flag = st.selectbox("Flag", ['SF', 'S0', 'REJ'])
    land = st.selectbox("Land", [0, 1])

if st.button("🚀 Detect"):

    input_data = {
        'duration': duration,
        'protocol_type': protocol_type,
        'service': service,
        'flag': flag,
        'src_bytes': src_bytes,
        'dst_bytes': 0,
        'land': land,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 1,
        'srv_count': 1,
        'serror_rate': 0,
        'srv_serror_rate': 0,
        'rerror_rate': 0,
        'srv_rerror_rate': 0,
        'same_srv_rate': 1,
        'diff_srv_rate': 0,
        'srv_diff_host_rate': 0,
        'dst_host_count': 255,
        'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1,
        'dst_host_diff_srv_rate': 0,
        'dst_host_same_src_port_rate': 0,
        'dst_host_srv_diff_host_rate': 0,
        'dst_host_serror_rate': 0,
        'dst_host_srv_serror_rate': 0,
        'dst_host_rerror_rate': 0,
        'dst_host_srv_rerror_rate': 0,
    }

    try:
        input_df = pd.DataFrame([input_data])

        cat_cols = ['protocol_type', 'service', 'flag']

        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                val = input_df[col].iloc[0]

                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0

        X_scaled = scaler.transform(input_df[FEATURE_COLS])

        prediction = model.predict(X_scaled)[0]

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                '<span class="attack-badge">🚨 ATTACK DETECTED</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="normal-badge">✅ NORMAL TRAFFIC</span>',
                unsafe_allow_html=True
            )

        img = traffic_to_image(X_scaled[0])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(img, cmap='hot', interpolation='nearest')
        ax.set_title("Traffic Heatmap")

        st.image(fig_to_pil(fig))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
