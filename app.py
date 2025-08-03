import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Construct full model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tree_species_model")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🌳 Tree Species Classifier")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))  # ensure RGB
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.write(f"### 🌲 Predicted Class: {predicted_class}")
