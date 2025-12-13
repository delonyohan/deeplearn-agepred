import streamlit as st
from PIL import Image
import numpy as np
import os
import random
import utils
import cv2

st.title("Live Age Prediction")

st.markdown("""
    Select an age category and an image from the dataset to see the live age prediction and face detection process.
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


model = utils.load_keras_model()
face_cascade = utils.load_face_cascade()
df_dataset = utils.load_dataset()

@st.cache_data
def get_face_detection_results(df_filepaths):
    face_cascade_local = utils.load_face_cascade()
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

if df_dataset.empty:
    st.error("Dataset not found or is empty. Please ensure 'dataset/UTKFace' contains images.")
else:
    st.sidebar.header("Image Selection")
    
    # Store previous age category to detect changes
    if 'prev_age_category' not in st.session_state:
        st.session_state.prev_age_category = None

    age_category_name = st.sidebar.selectbox(
        "Select Age Category",
        list(utils.AGE_CATEGORIES.keys())
    )

    # If age category changes, clear random images and selected image
    if st.session_state.prev_age_category != age_category_name:
        if 'random_images' in st.session_state:
            del st.session_state['random_images']
        if 'selected_image_path' in st.session_state:
            st.session_state.selected_image_path = None
        st.session_state.prev_age_category = age_category_name


    filtered_df = utils.filter_dataset(df_dataset, age_category_name)
    
    # Pre-filter images to ensure they have a detectable face
    with st.spinner('Preparing images... This may take a moment.'):
        filtered_df['has_face'] = get_face_detection_results(filtered_df['filepath'])
    
    face_detected_df = filtered_df[filtered_df['has_face']]

    if face_detected_df.empty:
        st.warning(f"No images with detectable faces found for the '{age_category_name}' category.")
    else:
        st.subheader(f"Images in '{age_category_name}' category")

        # Display a few random images from the filtered dataset
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
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading {os.path.basename(img_path)}")

        if st.session_state.selected_image_path:
            st.subheader("Prediction Process")
            
            original_image = Image.open(st.session_state.selected_image_path)
            actual_age = int(os.path.basename(st.session_state.selected_image_path).split('_')[0])

            # Use the utility function to load and preprocess the image
            processed_input, img_with_rect, cropped_face, faces = utils.load_and_preprocess_image(original_image, face_cascade)

            if processed_input is not None:
                # Make prediction
                predicted_age = model.predict(processed_input)[0][0]

                tab1, tab2, tab3 = st.tabs(["Pre-process", "Process", "Result"])

                with tab1:
                    st.markdown("#### Image Pre-processing Steps")
                    st.markdown("The first step is to prepare the image for the model.")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(original_image, caption="1. Original Image", width=200)
                    with col2:
                        st.image(img_with_rect, caption="2. Face Detection", width=200)
                        st.write("A face detection algorithm (Haar Cascade) is used to find the bounding box of the face.")
                    with col3:
                        st.image(cropped_face, caption="3. Cropped & Resized", width=100)
                        st.write("The detected face is cropped and resized to the model's input shape (224x224 pixels).")

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
                    final_image = utils.draw_prediction(original_image, faces, predicted_age, actual_age)
                    st.image(final_image, caption="Final Image with Prediction", width=400)
                    
                    st.subheader("Prediction Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Actual Age", f"{actual_age} years")
                    col2.metric("Predicted Age", f"{predicted_age:.2f} years")
                    error = abs(predicted_age - actual_age)
                    col3.metric("Prediction Error", f"{error:.2f} years")

            else:
                st.warning("No face detected in the selected image.")
        else:
            st.info("Please select an image from above to perform a prediction.")
