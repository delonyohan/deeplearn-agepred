import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2

import app.model_loader as model_loader
from src.utils import AGE_CATEGORIES

st.set_page_config(
    page_title="Deep Learning Age Prediction",
    page_icon="👴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Live Age Prediction")

st.markdown("""
    Upload your own image to see the live age prediction and face detection process.
""")

st.markdown("""
<style>
    .highlighted-image {
        border: 4px solid #f0ad4e;
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Use caching for model loading
@st.cache_resource
def cached_load_keras_model():
    return model_loader.load_keras_model()

@st.cache_resource
def cached_load_face_cascade():
    return model_loader.load_face_cascade()

model = cached_load_keras_model()
face_cascade = cached_load_face_cascade()

st.sidebar.header("Image Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload your own image",
    type=['png', 'jpg'],
    accept_multiple_files=False
)

if uploaded_file is not None:
    if uploaded_file.size > 1 * 1024 * 1024:
        st.sidebar.error("File size exceeds 1MB. Please upload a smaller image.")
        uploaded_file = None

image_to_predict = None

if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file)

if image_to_predict:
    st.subheader("Prediction Process")

    processed_input, img_with_rect, cropped_face, faces = model_loader.load_and_preprocess_image(image_to_predict, face_cascade)

    if processed_input is not None:
        predicted_age = model.predict(processed_input)[0][0]

        tab1, tab2, tab3 = st.tabs(["Pre-process", "Process", "Result"])

        with tab1:
            st.markdown("#### Image Pre-processing Steps")
            st.markdown("The first step is to prepare the image for the model.")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(image_to_predict, caption="1. Original Image", width=150)
            with col2:
                st.image(img_with_rect, caption="2. Face Detection", width=150)
                st.write("A face detection algorithm (Haar Cascade) is used to find the bounding box of the face.")
            with col3:
                st.image(cropped_face, caption="3. Cropped & Resized", width=100)
                st.write("The detected face is cropped and resized to the model's input shape (224x224 pixels).")
            with col4:
                st.markdown("##### 4. Normalization")
                st.write("The image's pixel values are normalized according to the requirements of the ResNet model. This involves mean subtraction to zero-center the data.")

        with tab2:
            st.markdown("#### Model Prediction Process")
            st.markdown("Once the face is prepared, it's passed through the deep learning model.")
            st.write("##### How it works:")
            st.markdown(f"""
                1.  **Input to Model:** The cropped face image is converted into a numerical array (a tensor) of shape `(1, 224, 224, 3)`.
                2.  **ResNet Model:** This tensor is fed into the pre-trained ResNet model, which extracts complex features associated with age.
                3.  **Output:** The model outputs a single numerical value, which is the predicted age.
            """)
            st.info("**Disclaimer:** This model is a demonstration and may not be 100% accurate. Predictions can be influenced by factors like image quality, lighting, and face orientation.")

        with tab3:
            st.markdown("#### Final Prediction Result")
            # For uploaded images, actual_age is unknown, so we pass 0
            final_image = model_loader.draw_prediction(image_to_predict, faces, predicted_age, 0)
            st.image(final_image, caption="Final Image with Prediction", width=400)

            st.subheader("Prediction Metrics")
            st.metric("Predicted Age", f"{predicted_age:.2f} years")

    else:
        st.warning("No face detected in the selected image.")
else:
    st.info("Please upload an image to perform a prediction.")
