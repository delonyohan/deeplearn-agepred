import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Model Evaluation Metrics")

st.markdown("""
    This page presents the evaluation metrics for the two age prediction models developed: 
    a Convolutional Neural Network (CNN) from scratch and a ResNet50-based model using transfer learning.
    We evaluate their performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
""")

# Evaluation Metrics Data (from notebook.ipynb)
metrics_data = {
    'Model': ['CNN', 'ResNet50'],
    'MAE': [8.58, 6.72],
    'MSE': [136.07, 93.78],
    'RMSE': [11.67, 9.68]
}
df_metrics = pd.DataFrame(metrics_data)

st.header("1. Model Performance Overview")
st.dataframe(df_metrics.set_index('Model'))

st.markdown("""
    ### Understanding the Metrics:
    - **Mean Absolute Error (MAE):** This is the average of the absolute differences between predictions and actual values. 
        It measures the average magnitude of the errors in a set of predictions, without considering their direction.
        A lower MAE indicates a more accurate model.
    - **Mean Squared Error (MSE):** This is the average of the squares of the errors. 
        It's a common metric for regression tasks. Since the errors are squared before they are averaged, 
        MSE gives a higher weight to larger errors. This means it's useful when large errors are particularly undesirable.
        A lower MSE indicates better performance.
    - **Root Mean Squared Error (RMSE):** This is the square root of the MSE. 
        It is often preferred over MSE because it is in the same units as the target variable (age in this case),
        making it more interpretable. A lower RMSE indicates better performance.
""")

st.header("2. Comparison and Analysis")
st.markdown("""
    From the table above, we can observe that the **ResNet50 model significantly outperforms the CNN model** 
    across all three evaluation metrics (MAE, MSE, and RMSE). This was expected, as the ResNet50 model
    leverages transfer learning from a pre-trained model (ImageNet), allowing it to benefit from features
    learned from a vast dataset.
""")

st.subheader("Visualizing the Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="CNN MAE", value=f"{df_metrics.loc[0, 'MAE']:.2f} years")
    st.metric(label="ResNet50 MAE", value=f"{df_metrics.loc[1, 'MAE']:.2f} years")
    st.caption("Lower MAE indicates better accuracy.")

with col2:
    st.metric(label="CNN MSE", value=f"{df_metrics.loc[0, 'MSE']:.2f}")
    st.metric(label="ResNet50 MSE", value=f"{df_metrics.loc[1, 'MSE']:.2f}")
    st.caption("Lower MSE indicates better performance, penalizing larger errors more.")

with col3:
    st.metric(label="CNN RMSE", value=f"{df_metrics.loc[0, 'RMSE']:.2f} years")
    st.metric(label="ResNet50 RMSE", value=f"{df_metrics.loc[1, 'RMSE']:.2f} years")
    st.caption("Lower RMSE (in years) implies predictions are closer to actual ages.")

st.subheader("Key Findings:")
st.markdown("""
    - **Accuracy:** ResNet50 shows a lower MAE (6.72 vs 8.58), meaning on average its predictions are closer to the true age.
    - **Error Sensitivity:** ResNet50's lower MSE (93.78 vs 136.07) indicates it makes fewer large errors compared to the CNN.
    - **Interpretability:** With an RMSE of 9.68 years, the ResNet50 model's predictions are, on average, within approximately 9.68 years of the actual age.
        The CNN model's predictions are, on average, within approximately 11.67 years.
    
    The use of a pre-trained ResNet50 as a feature extractor, combined with a custom regression head, 
    has clearly provided a more robust and accurate age prediction model.
""")

st.header("3. Model Training Insights")
st.markdown("""
    The plots below provide further insights into the training process for both models.
    **Note:** Please ensure these image files are present in the `assets/` directory for them to display correctly.
""")

st.subheader("CNN Training Plot")
st.markdown("""
    The plot below shows the training and validation loss for the CNN model. 
    The divergence between the two lines after the third epoch indicates overfitting.
""")
st.image("assets/cnn_training_plot.png", caption="CNN Training and Validation Loss")

st.subheader("ResNet Training Plot")
st.markdown("""
    This plot shows the training and validation loss for the ResNet model. 
    While more stable than the CNN, there is still a divergence, suggesting some overfitting.
""")
st.image("assets/resnet_training_plot.png", caption="ResNet Training and Validation Loss")

st.subheader("CNN Prediction Plot")
st.markdown("""
    This scatter plot shows the relationship between the actual ages and the ages predicted by the CNN model.
    The plot reveals the model's tendency to underestimate the age of older individuals.
""")
st.image("assets/cnn_prediction_plot.png", caption="CNN Predictions vs. Actual Age")

st.subheader("ResNet Prediction Plot")
st.markdown("""
    This scatter plot shows the relationship between the actual ages and the ages predicted by the ResNet model.
    Similar to the CNN, it shows a tendency to underestimate older ages, but with a tighter concentration of errors around zero.
""")
st.image("assets/resnet_prediction_plot.png", caption="ResNet Predictions vs. Actual Age")
