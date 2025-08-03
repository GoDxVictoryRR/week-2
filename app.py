import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Tree Species Classifier 🌳", layout="centered")
st.title("🌿 Tree Species Classifier")

@st.cache_resource
def load_model():
    # Preferred: SavedModel format directory
    model_path_dir = "tree_species_model"
    # Fallback: .h5 file
    model_path_h5 = "tree_species_model.h5"

    if os.path.isdir(model_path_dir):
        st.info(f"📂 Loading model from directory: `{model_path_dir}`")
        model = tf.keras.models.load_model(model_path_dir)
    elif os.path.isfile(model_path_h5):
        st.warning(f"⚠️ Model directory not found; loading fallback .h5 model: `{model_path_h5}`")
        model = tf.keras.models.load_model(model_path_h5)
    else:
        st.error("❌ Model file/folder not found! Please upload either `tree_species_model/` folder or `tree_species_model.h5` file to your repo.")
        raise FileNotFoundError("No model file or directory found.")
    
    # Optional: show model architecture summary in logs
    print("✅ Loaded model summary:")
    model.summary(print_fn=lambda x: print(x))
    return model

try:
    model = load_model()
except Exception as e:
    st.error("❌ Failed to load model:")
    st.exception(e)
    st.stop()

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
        st.write("🔄 Processing...")

        # Preprocess image
        image = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Predict
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
