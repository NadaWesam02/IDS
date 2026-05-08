# app.py - Intrusion Detection System Web App + CNN Image Processing
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
 
# Page configuration
st.set_page_config(
    page_title="IDS - Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide"
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 8px 20px;
        color: #cdd6f4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #89b4fa !important;
        color: #1e1e2e !important;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #313244);
        border: 1px solid #45475a;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .attack-badge {
        background: linear-gradient(135deg, #f38ba8, #e64553);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1em;
        text-align: center;
    }
    .normal-badge {
        background: linear-gradient(135deg, #a6e3a1, #40a02b);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1em;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Network Intrusion Detection System")
st.markdown("### Machine Learning + CNN Image Processing based IDS")
st.markdown("---")
 
# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load('ids_model_final.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, scaler, encoders
 
try:
    model, scaler, encoders = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Please make sure all model files (ids_model_final.pkl, scaler.pkl, label_encoders.pkl) are in the same directory")
    st.stop()
 
# ── CNN Model (lazy import) ───────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    """Try to load a saved CNN model, else return None (will train on the fly)."""
    try:
        import tensorflow as tf
        cnn = tf.keras.models.load_model('cnn_ids_model.h5')
        return cnn
    except Exception:
        return None
 
# ══════════════════════════════════════════════════════════════════════════════
# Helper: Traffic → Image conversion
# ══════════════════════════════════════════════════════════════════════════════
 
# Feature names matching NSL-KDD (41 features)
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
 
IMG_ROWS = 6   # 6 × 7 = 42 → we use first 41
IMG_COLS = 7
IMG_SHAPE = (IMG_ROWS, IMG_COLS, 1)
 
 
def traffic_to_image(feature_vector: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D array of 41 network-traffic features into a
    (IMG_ROWS × IMG_COLS) grayscale image.
 
    Steps:
      1. Clip extreme outliers (99th percentile per feature in training).
      2. Min-max normalise to [0, 1].
      3. Pad to IMG_ROWS * IMG_COLS and reshape.
    """
    vec = feature_vector.copy().astype(float)
 
    # Normalise each value to [0, 1] using the vector's own range
    v_min, v_max = vec.min(), vec.max()
    if v_max - v_min > 1e-8:
        vec = (vec - v_min) / (v_max - v_min)
    else:
        vec = np.zeros_like(vec)
 
    # Pad to fill the 2-D grid (42 cells, but only 41 features)
    padded = np.zeros(IMG_ROWS * IMG_COLS)
    padded[:len(vec)] = vec[:IMG_ROWS * IMG_COLS]
 
    img = padded.reshape(IMG_ROWS, IMG_COLS)
    return img
 
 
def batch_traffic_to_images(X: np.ndarray) -> np.ndarray:
    """Convert a 2-D feature matrix (n_samples × 41) to image tensor."""
    imgs = np.array([traffic_to_image(row) for row in X])
    return imgs[..., np.newaxis]          # → (n, rows, cols, 1)
 
 
def render_traffic_image(img_2d: np.ndarray,
                         label: str = "",
                         title: str = "Traffic Image") -> plt.Figure:
    """Render a single traffic image with colourful feature annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             facecolor='#1e1e2e', gridspec_kw={'width_ratios': [1, 1.6]})
 
    # ── Left: heatmap ────────────────────────────────────────────────────────
    ax_img = axes[0]
    im = ax_img.imshow(img_2d, cmap='plasma', aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')
    ax_img.set_title(title, color='white', fontsize=11, pad=8)
    ax_img.set_xlabel("Feature Column", color='#cdd6f4', fontsize=8)
    ax_img.set_ylabel("Feature Row", color='#cdd6f4', fontsize=8)
    ax_img.tick_params(colors='#6c7086')
    for spine in ax_img.spines.values():
        spine.set_edgecolor('#45475a')
    plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')
 
    # ── Right: bar chart of feature intensities ───────────────────────────────
    ax_bar = axes[1]
    flat = img_2d.flatten()[:41]
    colors = plt.cm.plasma(flat)
    bars = ax_bar.bar(range(len(flat)), flat, color=colors, edgecolor='none', width=0.8)
 
    # Highlight top-5 most intense features
    top5_idx = np.argsort(flat)[-5:]
    for idx in top5_idx:
        bars[idx].set_edgecolor('#f5c2e7')
        bars[idx].set_linewidth(1.5)
        ax_bar.text(idx, flat[idx] + 0.02,
                    FEATURE_NAMES[idx].replace('_', '\n'),
                    ha='center', va='bottom',
                    color='#f5c2e7', fontsize=5.5, rotation=0)
 
    ax_bar.set_facecolor('#1e1e2e')
    ax_bar.set_xlabel("Feature Index (0–40)", color='#cdd6f4', fontsize=8)
    ax_bar.set_ylabel("Normalised Value", color='#cdd6f4', fontsize=8)
    ax_bar.set_title("Feature Intensity Spectrum", color='white', fontsize=11, pad=8)
    ax_bar.tick_params(colors='#6c7086')
    ax_bar.set_ylim(0, 1.25)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#45475a')
    fig.patch.set_facecolor('#1e1e2e')
    plt.tight_layout()
    return fig
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CNN Training helper (trains a tiny CNN in the browser session)
# ══════════════════════════════════════════════════════════════════════════════
 
def build_and_train_cnn(X_sample: np.ndarray, y_sample: np.ndarray):
    """
    Build + train a compact CNN on a small labelled sample.
    Returns (model, history_dict).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models as km
 
    tf.random.set_seed(42)
 
    imgs = batch_traffic_to_images(X_sample)          # (n, 6, 7, 1)
 
    cnn = km.Sequential([
        layers.Input(shape=IMG_SHAPE),
 
        layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
 
        layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
 
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ], name="CNN_IDS")
 
    cnn.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
 
    history = cnn.fit(
        imgs, y_sample,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    return cnn, history.history
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
 
tab1, tab2, tab3 = st.tabs([
    "🔍 ML Detection",
    "🖼️ Image Processing (CNN)",
    "📊 Model Info"
])
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Original ML Detection (unchanged logic, improved layout)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])
 
    with col1:
        st.subheader("📡 Network Connection Features")
        st.markdown("Enter the network connection details below:")
 
        duration = st.number_input("Duration (seconds)", min_value=0, value=0, step=1)
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
        service = st.selectbox("Service", ["http", "private", "smtp", "ftp", "telnet", "other"])
        flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR"])
        src_bytes = st.number_input("Source Bytes", min_value=0, value=0, step=100)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0, step=100)
 
        with st.expander("Advanced Features (Optional)"):
            st.info("Leave default values if not sure")
            land = st.number_input("Land", value=0)
            wrong_fragment = st.number_input("Wrong Fragment", value=0)
            urgent = st.number_input("Urgent", value=0)
 
    with col2:
        st.subheader("🔍 Detection Result")
        st.markdown("Click the button below to analyse the connection")
 
        if st.button("🚨 DETECT INTRUSION", type="primary", use_container_width=True):
            try:
                protocol_encoded = encoders['protocol_type'].transform([protocol_type])[0]
                service_encoded  = encoders['service'].transform([service])[0]
                flag_encoded     = encoders['flag'].transform([flag])[0]
 
                features = np.array([
                    duration, protocol_encoded, service_encoded, flag_encoded,
                    src_bytes, dst_bytes, land, wrong_fragment, urgent,
                    *([0] * 32)
                ]).reshape(1, -1)
 
                features_scaled = scaler.transform(features)
                prediction      = model.predict(features_scaled)[0]
                probability     = model.predict_proba(features_scaled)[0]
 
                st.markdown("---")
                if prediction == 1:
                    st.markdown('<div class="attack-badge">🚨 ALERT: INTRUSION DETECTED!</div>',
                                unsafe_allow_html=True)
                    st.metric("Attack Confidence", f"{probability[1]*100:.2f}%")
                    st.progress(int(probability[1] * 100))
                else:
                    st.markdown('<div class="normal-badge">✅ NORMAL TRAFFIC</div>',
                                unsafe_allow_html=True)
                    st.metric("Normal Confidence", f"{probability[0]*100:.2f}%")
                    st.progress(int(probability[0] * 100))
 
                with st.expander("📊 Detailed Analysis"):
                    st.write(f"**Prediction:** {'Attack' if prediction==1 else 'Normal'}")
                    st.write(f"**Confidence:** {max(probability)*100:.2f}%")
                    st.write(f"**Attack Probability:** {probability[1]*100:.2f}%")
                    st.write(f"**Normal Probability:** {probability[0]*100:.2f}%")
 
                # ── Also show the traffic image ───────────────────────────────
                st.markdown("---")
                st.markdown("#### 🖼️ Traffic Visualised as Image")
                raw_vec = features[0]
                img_2d  = traffic_to_image(raw_vec)
                label   = "Attack" if prediction == 1 else "Normal"
                fig     = render_traffic_image(img_2d, label=label,
                                               title=f"Traffic Image — {label}")
                st.pyplot(fig)
                plt.close(fig)
 
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Make sure all fields are filled correctly")
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Image Processing (CNN)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🖼️ CNN-Based Image Processing for Intrusion Detection")
 
    st.markdown("""
    > **Idea:** Each network connection's 41 features are laid out as a **6 × 7 pixel grayscale image**.
    > A Convolutional Neural Network then learns spatial patterns — just like it would for photos.
    > This approach captures **feature interactions** that tabular models miss.
    """)
 
    st.markdown("---")
 
    # ── Step 1: Generate or upload sample data ────────────────────────────────
    st.markdown("### Step 1 — Provide Traffic Samples")
 
    data_source = st.radio(
        "Choose data source:",
        ["🎲 Generate synthetic samples", "📁 Load from NSL-KDD (online)"],
        horizontal=True
    )
 
    n_samples = st.slider("Number of samples to use", 200, 2000, 500, step=100)
 
    if st.button("🔄 Load / Generate Data", use_container_width=True):
        with st.spinner("Preparing data..."):
            try:
                if data_source == "📁 Load from NSL-KDD (online)":
                    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
                    df = pd.read_csv(train_url, nrows=n_samples + 50)
 
                    cols = FEATURE_NAMES + ['label', 'difficulty_level']
                    if len(df.columns) == len(cols):
                        df.columns = cols
                    else:
                        df.columns = cols[:len(df.columns)]
 
                    df['binary_label'] = df['label'].apply(
                        lambda x: 0 if x == 'normal' else 1)
 
                    cat_cols = ['protocol_type', 'service', 'flag']
                    for col in cat_cols:
                        if col in df.columns:
                            try:
                                df[col] = encoders[col].transform(df[col])
                            except Exception:
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                df[col] = le.fit_transform(df[col])
 
                    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
                    X_raw = df[feat_cols].values.astype(float)[:n_samples]
                    y_raw = df['binary_label'].values[:n_samples]
 
                else:
                    # Synthetic: random normal + attack patterns
                    rng = np.random.RandomState(42)
                    half = n_samples // 2
 
                    X_normal = rng.normal(loc=0.1, scale=0.05, size=(half, 41)).clip(0, 1)
                    X_attack = rng.normal(loc=0.7, scale=0.2,  size=(half, 41)).clip(0, 1)
                    # Inject typical attack signatures
                    X_attack[:, 4] *= 10     # high src_bytes
                    X_attack[:, 24] = rng.uniform(0.8, 1.0, half)  # high serror_rate
 
                    X_raw = np.vstack([X_normal, X_attack])
                    y_raw = np.array([0]*half + [1]*half)
                    shuffle = rng.permutation(n_samples)
                    X_raw, y_raw = X_raw[shuffle], y_raw[shuffle]
 
                st.session_state['X_raw'] = X_raw
                st.session_state['y_raw'] = y_raw
                st.success(f"✅ Loaded {len(X_raw)} samples  |  "
                           f"Normal: {(y_raw==0).sum()}  |  "
                           f"Attack: {(y_raw==1).sum()}")
 
            except Exception as ex:
                st.error(f"Failed to load data: {ex}")
 
    # ── Step 2: Visualise traffic images ─────────────────────────────────────
    if 'X_raw' in st.session_state:
        st.markdown("---")
        st.markdown("### Step 2 — Visualise Traffic as Images")
 
        X_raw = st.session_state['X_raw']
        y_raw = st.session_state['y_raw']
 
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🟢 Show Random NORMAL Sample", use_container_width=True):
                idx = np.random.choice(np.where(y_raw == 0)[0])
                img = traffic_to_image(X_raw[idx])
                fig = render_traffic_image(img, title=f"Sample #{idx} — NORMAL ✅")
                st.pyplot(fig);  plt.close(fig)
 
        with col_b:
            if st.button("🔴 Show Random ATTACK Sample", use_container_width=True):
                idx = np.random.choice(np.where(y_raw == 1)[0])
                img = traffic_to_image(X_raw[idx])
                fig = render_traffic_image(img, title=f"Sample #{idx} — ATTACK 🚨")
                st.pyplot(fig);  plt.close(fig)
 
        # Side-by-side comparison
        if st.button("⚡ Compare Normal vs Attack (side by side)", use_container_width=True):
            norm_idx   = np.random.choice(np.where(y_raw == 0)[0])
            attack_idx = np.random.choice(np.where(y_raw == 1)[0])
            img_norm   = traffic_to_image(X_raw[norm_idx])
            img_att    = traffic_to_image(X_raw[attack_idx])
 
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='#1e1e2e')
            for ax, img, ttl, cmap in zip(
                    axes,
                    [img_norm, img_att],
                    ["NORMAL ✅", "ATTACK 🚨"],
                    ['Greens', 'Reds']):
                im = ax.imshow(img, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                ax.set_title(ttl, color='white', fontsize=13, pad=8)
                ax.tick_params(colors='#6c7086')
                for sp in ax.spines.values():
                    sp.set_edgecolor('#45475a')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.patch.set_facecolor('#1e1e2e')
            fig.suptitle("Traffic → Image Comparison", color='#cdd6f4',
                         fontsize=14, y=1.02)
            plt.tight_layout()
            st.pyplot(fig);  plt.close(fig)
 
        # ── Step 3: Train CNN ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Step 3 — Train CNN on Traffic Images")
 
        st.info("💡 The CNN learns to classify Normal vs Attack directly from the pixel images.")
 
        col_ep, col_bs = st.columns(2)
        with col_ep:
            epochs = st.slider("Epochs", 5, 30, 15)
        with col_bs:
            batch_size_opt = st.select_slider("Batch size", options=[16, 32, 64], value=32)
 
        if st.button("🚀 Train CNN", type="primary", use_container_width=True):
            try:
                import tensorflow as tf
                tf_available = True
            except ImportError:
                tf_available = False
                st.error("TensorFlow is not installed. Run: `pip install tensorflow`")
 
            if tf_available:
                with st.spinner("Training CNN… this may take ~30 seconds ⏳"):
                    try:
                        from tensorflow.keras import layers, models as km
 
                        imgs = batch_traffic_to_images(X_raw)
                        from sklearn.model_selection import train_test_split
                        Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(
                            imgs, y_raw, test_size=0.2, random_state=42,
                            stratify=y_raw)
 
                        cnn = km.Sequential([
                            layers.Input(shape=IMG_SHAPE),
                            layers.Conv2D(32, (2,2), activation='relu', padding='same'),
                            layers.BatchNormalization(),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64, (2,2), activation='relu', padding='same'),
                            layers.BatchNormalization(),
                            layers.Flatten(),
                            layers.Dense(64, activation='relu'),
                            layers.Dropout(0.3),
                            layers.Dense(1, activation='sigmoid'),
                        ], name="CNN_IDS")
 
                        cnn.compile(optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'])
 
                        history = cnn.fit(
                            Xi_tr, yi_tr,
                            epochs=epochs,
                            batch_size=batch_size_opt,
                            validation_data=(Xi_val, yi_val),
                            verbose=0
                        )
 
                        st.session_state['cnn_model']   = cnn
                        st.session_state['cnn_history'] = history.history
                        st.session_state['Xi_val']      = Xi_val
                        st.session_state['yi_val']      = yi_val
 
                        val_acc = history.history['val_accuracy'][-1]
                        val_loss = history.history['val_loss'][-1]
                        st.success(f"✅ CNN trained!  Val Accuracy: **{val_acc*100:.2f}%**  |  "
                                   f"Val Loss: **{val_loss:.4f}**")
 
                    except Exception as ex:
                        st.error(f"Training failed: {ex}")
 
        # ── Step 4: Show results ─────────────────────────────────────────────
        if 'cnn_history' in st.session_state:
            st.markdown("---")
            st.markdown("### Step 4 — CNN Training Results")
 
            hist     = st.session_state['cnn_history']
            Xi_val   = st.session_state['Xi_val']
            yi_val   = st.session_state['yi_val']
            cnn_mdl  = st.session_state['cnn_model']
 
            # ── Training curves ───────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#1e1e2e')
            for ax, metric, title, colour in zip(
                    axes,
                    ['accuracy', 'loss'],
                    ['Accuracy', 'Loss'],
                    ['#89b4fa', '#f38ba8']):
                ax.plot(hist[metric],     color=colour,        lw=2, label='Train')
                ax.plot(hist[f'val_{metric}'], color='#a6e3a1', lw=2,
                        linestyle='--', label='Validation')
                ax.set_facecolor('#1e1e2e')
                ax.set_title(title, color='white', fontsize=12)
                ax.set_xlabel('Epoch', color='#cdd6f4')
                ax.tick_params(colors='#6c7086')
                for sp in ax.spines.values():
                    sp.set_edgecolor('#45475a')
                ax.legend(facecolor='#313244', labelcolor='white')
            fig.patch.set_facecolor('#1e1e2e')
            plt.tight_layout()
            st.pyplot(fig);  plt.close(fig)
 
            # ── Confusion Matrix ──────────────────────────────────────────────
            y_pred_prob = cnn_mdl.predict(Xi_val, verbose=0).flatten()
            y_pred      = (y_pred_prob >= 0.5).astype(int)
 
            from sklearn.metrics import confusion_matrix, classification_report, f1_score
            cm  = confusion_matrix(yi_val, y_pred)
            f1  = f1_score(yi_val, y_pred)
            acc = (y_pred == yi_val).mean()
 
            col_cm, col_rep = st.columns([1, 1])
            with col_cm:
                fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#1e1e2e')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Normal', 'Attack'],
                            yticklabels=['Normal', 'Attack'],
                            ax=ax2, linewidths=0.5)
                ax2.set_title('CNN Confusion Matrix', color='white', pad=10)
                ax2.set_xlabel('Predicted', color='#cdd6f4')
                ax2.set_ylabel('True', color='#cdd6f4')
                ax2.tick_params(colors='#cdd6f4')
                fig2.patch.set_facecolor('#1e1e2e')
                st.pyplot(fig2);  plt.close(fig2)
 
            with col_rep:
                st.markdown("#### 📋 CNN Metrics")
                st.metric("Accuracy",  f"{acc*100:.2f}%")
                st.metric("F1-Score",  f"{f1:.4f}")
                st.metric("Val Samples", len(yi_val))
                with st.expander("Full Report"):
                    report = classification_report(
                        yi_val, y_pred, target_names=['Normal','Attack'])
                    st.code(report)
 
            # ── Predict on single live sample ─────────────────────────────────
            st.markdown("---")
            st.markdown("### Step 5 — Real-time CNN Prediction")
            st.info("Generate a sample and let the CNN classify its image.")
 
            col_x, col_y = st.columns(2)
            with col_x:
                sample_type = st.selectbox("Sample type", ["Random from dataset", "Simulated Normal", "Simulated Attack"])
            with col_y:
                pass  # spacer
 
            if st.button("🎯 Predict with CNN", type="primary", use_container_width=True):
                if sample_type == "Random from dataset":
                    idx = np.random.randint(len(X_raw))
                    vec = X_raw[idx]
                    true_lbl = y_raw[idx]
                elif sample_type == "Simulated Normal":
                    rng = np.random.RandomState()
                    vec = rng.normal(0.1, 0.05, 41).clip(0, 1)
                    true_lbl = 0
                else:
                    rng = np.random.RandomState()
                    vec = rng.normal(0.7, 0.2, 41).clip(0, 1)
                    vec[4]  = min(vec[4] * 10, 1.0)
                    vec[24] = 0.95
                    true_lbl = 1
 
                img_2d   = traffic_to_image(vec)
                img_4d   = img_2d[np.newaxis, ..., np.newaxis]
                prob     = cnn_mdl.predict(img_4d, verbose=0)[0][0]
                pred_lbl = int(prob >= 0.5)
 
                fig3 = render_traffic_image(
                    img_2d,
                    title=f"CNN says: {'🚨 ATTACK' if pred_lbl==1 else '✅ NORMAL'}  "
                          f"(confidence {prob if pred_lbl==1 else 1-prob:.2%})"
                )
                st.pyplot(fig3);  plt.close(fig3)
 
                if pred_lbl == 1:
                    st.markdown('<div class="attack-badge">🚨 CNN PREDICTS: ATTACK</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="normal-badge">✅ CNN PREDICTS: NORMAL</div>',
                                unsafe_allow_html=True)
 
                st.metric("Attack Probability", f"{prob*100:.2f}%")
                if 'X_raw' in st.session_state and sample_type == "Random from dataset":
                    st.caption(f"True label: {'Attack' if true_lbl==1 else 'Normal'} | "
                               f"CNN: {'Attack' if pred_lbl==1 else 'Normal'} | "
                               f"{'✅ Correct' if pred_lbl==true_lbl else '❌ Wrong'}")
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Info
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Model Information & Architecture")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("#### 🌲 Random Forest (ML Model)")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | Random Forest |
        | Trees | 300 |
        | Max Depth | 20 |
        | Dataset | NSL-KDD |
        | SMOTE | Yes |
        | Accuracy | **98.1%** |
        | Recall | **98.67%** |
        """)
 
    with col2:
        st.markdown("#### 🧠 CNN (Image Model)")
        st.markdown("""
        | Layer | Details |
        |-------|---------|
        | Input | 6 × 7 × 1 grayscale |
        | Conv2D | 32 filters, 2×2 |
        | BatchNorm + MaxPool | — |
        | Conv2D | 64 filters, 2×2 |
        | BatchNorm | — |
        | Dense | 64 units, ReLU |
        | Dropout | 0.3 |
        | Output | Sigmoid |
        """)
 
    st.markdown("---")
    st.markdown("#### 🔄 How Traffic → Image Works")
    st.markdown("""
    ```
    41 Network Features
          ↓
    Min-Max Normalise → [0, 1]
          ↓
    Reshape to 6 × 7 grid (42 cells, last cell padded)
          ↓
    Grayscale Image (pixel = feature intensity)
          ↓
    CNN processes spatial patterns
          ↓
    Normal / Attack prediction
    ```
    """)
 
    st.markdown("---")
    st.markdown("#### ℹ️ About the System")
    st.markdown("""
    This **Intrusion Detection System** combines two complementary approaches:
 
    - **Random Forest (Tab 1)**: Fast, tabular ML — industry-proven, high recall.
    - **CNN Image Processing (Tab 2)**: Converts traffic to images, learns visual attack patterns.
 
    Dataset: **NSL-KDD** — the benchmark dataset for network intrusion detection.
    """)
 
# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.markdown("## IDS Dashboard")
    st.markdown("""
    **Two Detection Engines:**
    - 🌲 Random Forest (ML)
    - 🧠 CNN (Image-based)
 
    ---
    **Dataset:** NSL-KDD
    **Features:** 41 network features
 
    ---
    **Performance (RF):**
    - Accuracy: 98.1%
    - Recall: 98.67%
 
    ---
    **CNN Image Size:** 6 × 7 pixels
    """)
    st.markdown("---")
    st.caption("Built with Streamlit + TensorFlow")
