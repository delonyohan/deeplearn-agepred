import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import cv2
import os

def load_keras_model():
    model = load_model('outputs/resnet_age_prediction_model.h5', compile=False)
    return model

def load_face_cascade():
    cascade_path = os.path.join(os.getcwd(), 'config/haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        return None
    return cv2.CascadeClassifier(cascade_path)

def load_and_preprocess_image(image, face_cascade):
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, None, None, None

    (x, y, w, h) = faces[0]
    
    img_with_rect = img_np.copy()
    cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cropped_face = img_np[y:y+h, x:x+w]
    
    img_resized = cv2.resize(cropped_face, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    return img_expanded, Image.fromarray(img_with_rect), Image.fromarray(cropped_face), faces

def draw_prediction(image, faces, predicted_age, actual_age):
    img_np = np.array(image.convert('RGB'))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        pred_text = f"Predicted: {predicted_age:.0f}"
        
        (tw, th), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        if actual_age > 0:
            actual_text = f"Actual: {actual_age}"
            cv2.rectangle(img_np, (x, y - 50), (x + tw, y), (0,0,0), -1)
            cv2.putText(img_np, pred_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_np, actual_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(img_np, (x, y - 25), (x + tw, y), (0,0,0), -1)
            cv2.putText(img_np, pred_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return Image.fromarray(img_np)
