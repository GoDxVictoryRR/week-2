import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App title and layout
st.set_page_config(page_title="Tree Species Classifier 🌳", layout="centered")
st.title("🌲 Tree Species Prediction App")

# Load model (cached to avoid reloading on each run)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("tree_species_model.h5")
    return model

try:
    model = load_model()
except Exception as e:
    st.error("❌ Failed to load the model. Please check the file `tree_species_model.h5`.")
    st.exception(e)
    st.stop()

# Class labels (adjust if needed)
class_names = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa',
    'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur',
    'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren', 'pipal',
    'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad'
]

# Upload image
uploaded_file = st.file_uploader("📤 Upload a tree image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Preprocess image
        image = image.resize((224, 224))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display result
        st.success(f"✅ Predicted: **{predicted_class}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")
    except Exception as e:
        st.error("⚠️ Error during prediction.")
        st.exception(e)
else:
    st.info("👆 Upload a tree image to get started.")
