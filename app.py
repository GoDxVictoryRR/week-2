import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.title("🌳 Tree Species Classifier")

# Debug output to verify folder structure
st.subheader("Debug Info")
st.write("📁 Files in root directory:", os.listdir())
if "tree_species_model" in os.listdir():
    st.write("📁 Files in `tree_species_model/` directory:", os.listdir("tree_species_model"))
else:
    st.error("❌ `tree_species_model` folder not found in the root directory!")

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tree_species_model")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="🖼 Uploaded Leaf", use_column_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    st.success(f"✅ **Predicted Class:** {predicted_class}")
