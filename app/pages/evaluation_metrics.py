import streamlit as st
from PIL import Image

def main():
    st.set_page_config(
        page_title="Evaluation Metrics",
        page_icon="📊",
        layout="wide"
    )

    st.title("Model Evaluation Metrics")

    st.markdown("""
        This page displays various plots and metrics related to the model's performance.
        These plots were generated during the model training and evaluation phase.
    """)

    st.subheader("CNN Model Performance")

    st.image("outputs/plots/cnn_training_plot.png", caption="CNN Training Metrics (Loss and Accuracy)")
    st.image("outputs/plots/cnn_prediction_plot.png", caption="CNN Prediction Performance")

    st.subheader("ResNet Model Performance")

    st.image("outputs/plots/resnet_training_plot.png", caption="ResNet Training Metrics (Loss and Accuracy)")
    st.image("outputs/plots/resnet_prediction_plot.png", caption="ResNet Prediction Performance")

    st.markdown("""
        **Note:** For more detailed analysis, refer to the `notebooks/notebook.ipynb` file.
    """)