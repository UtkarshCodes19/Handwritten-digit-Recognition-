import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

MODEL_PATH = "digit_model.keras"

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="🔢")
st.title("🔢 Handwritten Digit Recognizer")
st.write("Upload a handwritten digit image (0–9). The CNN model will predict the digit.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.warning("Model file not found. Run `python train.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Uploaded image", width=220)

    # MNIST-style preprocessing: grayscale, invert if needed, resize to 28x28.
    image = ImageOps.fit(image, (28, 28))
    arr = np.array(image).astype("float32") / 255.0

    # Heuristic: MNIST has a dark background and bright digit.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    x = arr.reshape(1, 28, 28, 1)
    probs = model.predict(x, verbose=0)[0]
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted])

    st.success(f"Predicted digit: **{predicted}**")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.subheader("Class probabilities")
    st.bar_chart({str(i): float(probs[i]) for i in range(10)})
