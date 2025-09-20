import streamlit as st
from app.model import load_model, predict

st.set_page_config(page_title="Skin Disorder Detector", layout="wide")

st.title("Skin Disorder Detector — Demo")
st.write("Fill in the 34 features below:")

@st.cache_resource
def load():
    return load_model()

MODEL = load()

# Make input fields
features = []
cols = st.columns(4)  # 4 columns layout for clean UI

for i in range(34):
    with cols[i % 4]:
        val = st.number_input(f"Feature {i+1}", value=0.0, step=0.1, format="%.2f")
        features.append(val)

if st.button("Predict"):
    try:
        res = predict(MODEL, features)
        label = res["prediction"] if isinstance(res, dict) and "prediction" in res      else res
        st.success(f"Predicted Disease: {label}")

    except Exception as e:
        st.error(f"Error: {e}")


