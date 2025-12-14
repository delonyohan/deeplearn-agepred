import streamlit as st

def main():
    st.set_page_config(
        page_title="Deep Learning Age Prediction",
        page_icon="👴",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    </style>
    """, unsafe_allow_html=True)

    st.title("Welcome to the Deep Learning Age Prediction App! 👴👶")

    st.markdown("""
        This application demonstrates an age prediction model built using deep learning techniques.
        This app utilizes a ResNet model for age prediction.
        Explore the different pages in the sidebar to understand the model's performance,
        and try out live predictions using images from the available dataset.

        **Project Overview:**

        The primary goal of this project is to develop a deep learning model capable of accurately predicting the age of a person from
     a facial image. This involves several key stages:

        1.  **Data Preprocessing:** Utilizing the UTKFace dataset, which contains over 20,000 images, each labeled with age, gender, and ethnicity. The images are preprocessed to be suitable for model training.
        2.  **Model Development:** Two different architectures are explored for this task:
            - A custom **Convolutional Neural Network (CNN)** for age prediction built from scratch.
            - A more complex **Residual Network (ResNet)**, leveraging transfer learning principles.
        3.  **Model Evaluation:** The performance of both models is rigorously evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
        4.  **Live Prediction:** The best-performing model is deployed in this interactive web application, allowing users to select an image and see the age prediction in real-time.

        This application serves as a comprehensive demonstration of the end-to-end process of building and deploying a deep learning model for a real-world computer vision task.

        **Key Features:**
        - **Evaluation Metrics:** See a detailed breakdown of the model's performance (MAE, MSE, RMSE) and a comparison between different architectures.
        - **Live Prediction:** Select an image from the dataset, visualize the face detection, and get an age prediction.

        Navigate through the pages using the sidebar on the left!
    """)

    st.markdown("---")
    st.markdown("<h5 style='text-align: center; color: grey;'>Developed by Delon Yohan & Filip Nathan from LB01</h5>", unsafe_allow_html=True)
