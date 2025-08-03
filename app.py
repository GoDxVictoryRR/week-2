import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.title("🌳 Tree Species Classifier")

# Show debug info
st.subheader("Debug Info")
st.write("📁 Files in root directory:", os.listdir())
if "tree_species_model" in os.listdir():
    st.write("📁 Files in `tree_species_model/` directory:", os.listdir("tree_species_model"))
else:
    st.error("❌ `tree_species_model` folder not found in the root directory!")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.layers.TFSMLayer("tree_species_model", call_endpoint="serving_default")

try:
    model_layer = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="🖼 Uploaded Leaf", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    try:
        output = model_layer(img_array)
        if isinstance(output, dict):
            output_tensor = list(output.values())[0]  # Extract tensor from dict
        else:
            output_tensor = output

        prediction = output_tensor.numpy()
        predicted_class = np.argmax(prediction, axis=1)[0]
        st.success(f"✅ **Predicted Class:** {predicted_class}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
