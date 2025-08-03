import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Tree Species Classifier 🌳", layout="centered")
st.title("🌿 Tree Species Classifier")

@st.cache_resource
def load_model():
    model_path = "tree_species_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please make sure 'tree_species_model.h5' is in the repo.")
        raise FileNotFoundError("Model file not found.")

    try:
        with tf.keras.utils.custom_object_scope({}):  # Handle possible custom layers
            model = tf.keras.models.load_model(model_path)
        model.build(input_shape=(None, 224, 224, 3))  # Ensure model is built
        return model
    except Exception as e:
        st.error("❌ Failed to load model:")
        st.exception(e)
        raise

model = load_model()

class_names = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa',
    'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur',
    'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren', 'pipal',
    'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad'
]

uploaded_file = st.file_uploader("📤 Upload a tree image (leaf/bark)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        st.success(f"✅ Predicted Species: **{predicted_label}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")
    except Exception as e:
        st.error("🚨 Error during prediction:")
        st.exception(e)
else:
    st.info("📂 Please upload a tree image to get started.")
