import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model once at startup
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tree_species_model")

model = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Streamlit UI
st.title("🌳 Tree Species Classifier")
st.write("Upload an image of a leaf/tree and get the predicted species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Classifying..."):
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

    # Display result
    st.success(f"✅ Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: `{confidence * 100:.2f}%`")

    # Optional: If you have class names
    # class_names = ["Oak", "Maple", "Pine", ...]
    # st.success(f"✅ Predicted Species: **{class_names[predicted_class]}**")
