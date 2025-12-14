import streamlit as st
from PIL import Image
import numpy as np
import os
import random
import cv2
import sys

# Add the parent directory to the path to allow imports from `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import utils
from app import model_loader

def main():
    st.title("Live Age Prediction")

    st.markdown("""
        Select an age category and an image from the dataset, or upload your own image to see the live age prediction and face detection process.
    """)

    st.markdown("""
    <style>
        .highlighted-image {
            border: 4px solid #f0ad4e;
            border-radius: 5px;
            padding: 5px;
        }
        .gallery-image {
            padding: 5px;
            border: 4px solid transparent;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)


    @st.cache_resource
    def cached_load_keras_model():
        return model_loader.load_keras_model()

    @st.cache_resource
    def cached_load_face_cascade():
        return model_loader.load_face_cascade()

    model = cached_load_keras_model()
    face_cascade = cached_load_face_cascade()
    df_dataset = utils.load_dataset()

    @st.cache_data
    def get_face_detection_results(df_filepaths):
        face_cascade_local = model_loader.load_face_cascade()
        def check_for_face(image_path):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    return False
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade_local.detectMultiScale(gray, 1.1, 4)
                return len(faces) > 0
            except Exception:
                return False
        return df_filepaths.apply(check_for_face)

    if df_dataset.empty and not st.sidebar.file_uploader:
        st.error("Dataset not found or is empty. Please ensure 'dataset/UTKFace' contains images, or upload an image.")
    else:
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
            else:
                st.session_state.selected_image_path = None


        if 'prev_age_category' not in st.session_state:
            st.session_state.prev_age_category = None

        age_category_name = st.sidebar.selectbox(
            "Select Age Category (for gallery)",
            list(utils.AGE_CATEGORIES.keys())
        )

        if st.session_state.prev_age_category != age_category_name:
            if 'random_images' in st.session_state:
                del st.session_state['random_images']
            if 'selected_image_path' in st.session_state:
                st.session_state.selected_image_path = None
            st.session_state.prev_age_category = age_category_name


        filtered_df = utils.filter_dataset(df_dataset, age_category_name)

        if not df_dataset.empty:
            with st.spinner('Preparing images... This may take a moment.'):
                filtered_df['has_face'] = get_face_detection_results(filtered_df['filepath'])

            face_detected_df = filtered_df[filtered_df['has_face']]

            if face_detected_df.empty and uploaded_file is None:
                st.warning(f"No images with detectable faces found for the '{age_category_name}' category.")
            elif not face_detected_df.empty:
                st.subheader(f"Images in '{age_category_name}' category")

                if 'random_images' not in st.session_state:
                    num_images_to_show = min(10, len(face_detected_df))
                    st.session_state.random_images = face_detected_df.sample(num_images_to_show)['filepath'].tolist()

                cols = st.columns(len(st.session_state.random_images))

                if 'selected_image_path' not in st.session_state:
                    st.session_state.selected_image_path = None

                for i, img_path in enumerate(st.session_state.random_images):
                    with cols[i]:
                        try:
                            img = Image.open(img_path)
                            st.image(img, caption=f"Age: {os.path.basename(img_path).split('_')[0]}", width=100)
                            if st.button("Select", key=img_path):
                                st.session_state.selected_image_path = img_path
                                uploaded_file = None
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error loading {os.path.basename(img_path)}")

        image_to_predict = None
        actual_age = None

        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)
            actual_age = "N/A"
        elif st.session_state.selected_image_path:
            image_to_predict = Image.open(st.session_state.selected_image_path)
            actual_age = int(os.path.basename(st.session_state.selected_image_path).split('_')[0])

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
                    final_image = model_loader.draw_prediction(image_to_predict, faces, predicted_age, actual_age if isinstance(actual_age, int) else 0)
                    st.image(final_image, caption="Final Image with Prediction", width=400)

                    st.subheader("Prediction Metrics")
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Age", f"{predicted_age:.2f} years")
                    if isinstance(actual_age, int):
                        col2.metric("Actual Age", f"{actual_age} years")
                        error = abs(predicted_age - actual_age)
                        st.metric("Prediction Error", f"{error:.2f} years")


            else:
                st.warning("No face detected in the selected image.")
        else:
            st.info("Please select an image from the gallery or upload your own to perform a prediction.")