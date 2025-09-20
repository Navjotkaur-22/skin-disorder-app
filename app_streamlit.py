import streamlit as st
from app.model import load_model, predict

st.set_page_config(page_title="Skin Disorder Detector", layout="centered")

st.title("Skin Disorder Detector — Demo")
st.write("Enter numeric features as comma-separated values (same order as model training).")

@st.cache_resource
def load():
    return load_model()

MODEL = load()

st.subheader("Input features")
features_text = st.text_area("Comma-separated features", placeholder="e.g. 0.12, 1.45, 3.2, ...")

if st.button("Predict"):
    if not features_text.strip():
        st.warning("Please enter feature values.")
    else:
        try:
            features = [float(x.strip()) for x in features_text.split(",") if x.strip()!='']
            res = predict(MODEL, features)
            st.success(f"Prediction: {res}")
            st.json({"input": features, "prediction": res})
        except Exception as e:
            st.error(f"Error running prediction: {e}")
