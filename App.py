# app.py - Intrusion Detection System Web App
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('ids_model_final.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="IDS - Intrusion Detection System", page_icon="🛡️")
st.title("🛡️ Network Intrusion Detection System")
st.markdown("Machine Learning based IDS to detect cyber attacks")

# Sidebar for input
st.sidebar.header("Network Connection Features")

# Create input fields (41 features)
def get_user_input():
    features = []
    
    # Basic features
    duration = st.sidebar.number_input("Duration", value=0)
    protocol = st.sidebar.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
    service = st.sidebar.selectbox("Service", ["http", "private", "smtp", "ftp", "telnet"])
    flag = st.sidebar.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR"])
    
    # Convert categorical
    protocol_map = {'tcp':0, 'udp':1, 'icmp':2}
    service_map = {'http':0, 'private':1, 'smtp':2, 'ftp':3, 'telnet':4}
    flag_map = {'SF':0, 'S0':1, 'REJ':2, 'RSTO':3, 'RSTR':4}
    
    features.append(duration)
    features.append(protocol_map[protocol])
    features.append(service_map[service])
    features.append(flag_map[flag])
    
    # Add remaining features with default values
    for i in range(37):  # باقي 37 ميزة
        features.append(st.sidebar.number_input(f"Feature_{i+5}", value=0.0, format="%.1f"))
    
    return np.array(features).reshape(1, -1)

# Predict button
if st.button("🔍 Detect Intrusion", type="primary"):
    input_data = get_user_input()
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Show result
    st.markdown("---")
    st.subheader("🔔 Detection Result")
    
    col1, col2 = st.columns(2)
    
    if prediction == 1:
        col1.error("🚨 ALERT: INTRUSION DETECTED!")
        col2.metric("Confidence", f"{probability[1]*100:.2f}%")
        st.warning("⚠️ This connection appears to be malicious!")
    else:
        col1.success("✅ NORMAL TRAFFIC")
        col2.metric("Confidence", f"{probability[0]*100:.2f}%")
        st.info("This connection appears to be normal.")
    
    # Additional info
    with st.expander("📊 Detailed Analysis"):
        st.write(f"Prediction: {'Attack' if prediction==1 else 'Normal'}")
        st.write(f"Confidence: {max(probability)*100:.2f}%")
