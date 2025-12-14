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
To start the Streamlit application, run the following command from the root directory:
```bash
streamlit run app/app.py
```
The application will open in your web browser. You can then upload an image or select one from the gallery to see the age prediction.

## Built With

*   **Streamlit:** For the web application interface.
*   **TensorFlow / Keras:** For building and training the deep learning model.
*   **OpenCV:** For image processing and face detection.
*   **Pandas & NumPy:** For data manipulation.
*   **Scikit-learn:** For splitting the dataset.
*   **Matplotlib:** For plotting and visualization in the notebooks.
