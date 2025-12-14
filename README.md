# Deep Learning Age Prediction

This application demonstrates an age prediction model built using deep learning techniques. This app utilizes a ResNet model for age prediction.

## Project Overview

The primary goal of this project is to develop a deep learning model capable of accurately predicting the age of a person from a facial image. This involves several key stages:

1.  **Data Preprocessing:** Utilizing the UTKFace dataset, which contains over 20,000 images, each labeled with age, gender, and ethnicity.
2.  **Model Development:** A deep learning model using a ResNet architecture is trained for age prediction.
3.  **Live Prediction:** The trained model is deployed in an interactive web application.

This project is structured to follow best practices in machine learning engineering, with separate directories for data, notebooks, source code, configuration, and application.

## How to Use

### Prerequisites
- Python 3.8 or higher
- Pip for package management

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/delonyohan/deeplearn-agepred.git
    cd deeplearn-agepred
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
This application is structured as a multi-page Streamlit app, with a central dispatcher managing navigation between different sections: Home, Live Prediction, and Evaluation Metrics.

To start the Streamlit application locally, navigate to the project's root directory and run the following command:
```bash
streamlit run app/streamlit_app.py
```
The application will open in your web browser. You can then use the sidebar navigation to explore the Home page, perform live age predictions, or view model evaluation metrics.

#### Deployment to Streamlit Cloud
For deployment on Streamlit Cloud, ensure your repository is linked and configure the "Main module path" in your app's advanced settings to:
```
app/streamlit_app.py
```

## Built With

*   **Streamlit:** For the web application interface.
*   **TensorFlow / Keras:** For building and training the deep learning model.
*   **OpenCV:** For image processing and face detection.
*   **Pandas & NumPy:** For data manipulation.
*   **Scikit-learn:** For splitting the dataset.
*   **Matplotlib:** For plotting and visualization in the notebooks.
