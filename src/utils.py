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

@st.cache_data
def load_dataset():
    DATASET_PATH = "data/dataset/UTKFace"
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

