import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.title("🌳 Tree Species Classifier")

# Debug output
st.subheader("Debug Info")
st.write("📁 Files in root directory:", os.listdir())
if "tree_species_model" in os.listdir():
    st.write("📁 Files in `tree_species_model/` directory:", os.listdir("tree_species_model"))
else:
    st.error("❌ `tree_species_model` folder not found in the root directory!")

@st.cache_resource
def load_model():
    return tf.keras.layers.TFSMLayer("tree_species_model", call_endpoint="serving_default")

try:
    model_layer = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="🖼 Uploaded Leaf", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Predict
    prediction = model_layer(img_array)
    predicted_class = np.argmax(prediction.numpy(), axis=1)[0]

    st.success(f"✅ **Predicted Class:** {predicted_class}")

