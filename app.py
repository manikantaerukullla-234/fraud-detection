import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it's fraudulent.")

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Exact Error: {e}")
        return None

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.warning("model.pkl not found. Run src/train.py first to generate it.")
    model_loaded = False

st.subheader("Transaction Input")

col1, col2 = st.columns(2)

with col1:
    scaled_time = st.number_input("Scaled Time", value=0.0, format="%.4f")
    scaled_amount = st.number_input("Scaled Amount", value=0.0, format="%.4f")

with col2:
    st.markdown("**PCA Features (V1 – V10)**")

v_values = []
cols = st.columns(5)
feature_names = [f"V{i}" for i in range(1, 29)]

for i, name in enumerate(feature_names):
    col = cols[i % 5]
    v_values.append(col.number_input(name, value=0.0, format="%.4f", key=name))

if st.button("Predict", type="primary", disabled=not model_loaded):
    features = np.array([[scaled_amount, scaled_time] + v_values])
    # reorder to match training column order
    col_order = feature_names + ["scaled_amount", "scaled_time"]
    input_df = pd.DataFrame(features, columns=["scaled_amount", "scaled_time"] + feature_names)
    input_df = input_df[feature_names + ["scaled_amount", "scaled_time"]]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"FRAUD DETECTED — Confidence: {probability*100:.1f}%")
    else:
        st.success(f"Legitimate Transaction — Fraud probability: {probability*100:.1f}%")

    st.progress(float(probability), text=f"Fraud risk: {probability*100:.1f}%")

st.divider()
st.caption("Built with Scikit-learn + Streamlit · Trained on Kaggle Credit Card Fraud Dataset")
