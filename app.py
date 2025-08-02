import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Tree Classifier 🌿", layout="centered")
st.title("🌳 Tree Species Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a tree image (leaf, bark, etc.)", type=["jpg", "jpeg", "png"])

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("tree_model.h5")
    return model

model = load_model()

# Class labels
class_names = ['amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa', 'coconut',
               'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur', 'mango',
               'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren', 'pipal', 'saptaparni', 'shirish',
               'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad']

# Predict
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    st.success(f"✅ Predicted Species: **{predicted_label}** ({confidence:.2f}% confidence)")
