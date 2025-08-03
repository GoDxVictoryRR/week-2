import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tree_species_model")

model = load_model()

st.title("🌳 Tree Species Classifier")

uploaded_file = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess
    image = image.resize((224, 224))  # Change if your model uses different input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    st.markdown(f"### 🌿 Predicted Class: **{predicted_class}**")
