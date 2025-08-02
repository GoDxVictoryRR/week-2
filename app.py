import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Streamlit page config
st.set_page_config(page_title="Tree Classifier 🌿", layout="centered")
st.title("🌳 Tree Species Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a tree image (leaf, bark, etc.)", type=["jpg", "jpeg", "png"])

# Load the model safely with caching
@st.cache_resource
def load_model():
    model_path = "tree_species_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: `{model_path}`. Please check the file name or path.")
        raise FileNotFoundError(f"{model_path} not found.")
    return tf.keras.models.load_model(model_path)

# Load model once
try:
    model = load_model()
except Exception as e:
    st.stop()

# Tree species class labels
class_names = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa',
    'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur',
    'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren', 'pipal',
    'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad'
]

# Prediction section
if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = Image.open(uploaded_file).resize((224, 224)).convert('RGB')
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = class_names[np.argmax(prediction)]
        confidence = 100 * np.max(prediction)

        # Show result
        st.success(f"✅ Predicted Species: **{predicted_label}** ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
