import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import cv2
import os

AGE_CATEGORIES = {
    "All": (0, 116),
    "Toddlerhood (1-3 years)": (1, 3),
    "Early/Preschool (2-6 years)": (2, 6),
    "Childhood (6-12 years)": (6, 12),
    "Adolescence (12-18 years)": (12, 18),
    "Early Adulthood (18-40 years)": (18, 40),
    "Middle Adulthood (40-65 years)": (40, 65),
    "Late Adulthood (65+ years)": (65, 116),
}

@st.cache_resource
def load_keras_model():
    model = load_model('model-process/resnet_age_prediction_model.h5', compile=False)
    return model

def load_face_cascade():
    # Construct path relative to the current script
    cascade_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        st.error(f"Haar cascade file not found at: {cascade_path}")
        return None
    return cv2.CascadeClassifier(cascade_path)

@st.cache_data
def load_dataset():
    DATASET_PATH = "dataset/UTKFace"
    data = []
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame(data, columns=["filepath", "age", "gender", "race"])
        
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".jpg"):
            try:
                parts = file.split("_")
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                filepath = os.path.join(DATASET_PATH, file)
                data.append({"filepath": filepath, "age": age, "gender": gender, "race": race})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(data)

def filter_dataset(df, age_category):
    if age_category == "All":
        return df
    
    min_age, max_age = AGE_CATEGORIES[age_category]
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]

def load_and_preprocess_image(image, face_cascade):
    # Convert PIL image to numpy array
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, None, None, None

    # Assume the first detected face is the one we want
    (x, y, w, h) = faces[0]
    
    # Draw rectangle around the face
    img_with_rect = img_np.copy()
    cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Crop the face
    cropped_face = img_np[y:y+h, x:x+w]
    
    # Preprocess for ResNet
    img_resized = cv2.resize(cropped_face, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    return img_expanded, Image.fromarray(img_with_rect), Image.fromarray(cropped_face), faces

def draw_prediction(image, faces, predicted_age, actual_age):
    img_np = np.array(image.convert('RGB'))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        # Draw rectangle around the face
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare text for predicted and actual age
        pred_text = f"Predicted: {predicted_age:.0f}"
        actual_text = f"Actual: {actual_age}"
        
        # Position the text above the bounding box
        # Add background rectangle for text for better visibility
        (tw, th), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_np, (x, y - 50), (x + tw, y), (0,0,0), -1)
        
        # Draw the text
        cv2.putText(img_np, pred_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_np, actual_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return Image.fromarray(img_np)
