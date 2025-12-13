#!/usr/bin/env python
# coding: utf-8

# **Deep Learning Final Project Notebook**

# ### 1. Libraries

# In[49]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


# ### 2. Load Data & Preprocess

# In[3]:


# from google.colab import drive
# drive.mount('/content/drive')


# Because the UTKFace dataset contains over 23.000 images and the available resource (RAM) is limited, I cant load all of them into the memory at once. For that, tensor slices are used. Tensor slices only stores the file paths and labels of each image, dividing them into 'slices' which allows the model to process the dataset in small batches, preventing memory overflow. 

# The steps for data processing are:
# 1. Extract target variable (age) from the file name. The format of the image names are "[age]\_[gender]\_[race]_[date&time].jpg".
# 2. Store the file names and age into lists.
# 3. The lists are converted into a numpy array so that they can be passed into tensorflow’s dataset pipeline.
# 4. The images are resized into a fixed size.
# 5. Images are then normalized.
# 6. Split dataset into train, validation, and test set with a ratio of 70 : 15 : 15

# ** CNN and ResNet have some difference in step 4 & 5, as the ResNet one must follow the original ImageNet configuration.
# 

# In[4]:


DATASET_PATH = "dataset/UTKFace" # Updated to local path


filepaths = []
ages = []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".jpg"):
        age = int(file.split("_")[0])
        filepaths.append(os.path.join(DATASET_PATH, file))
        ages.append(age)

filepaths = tf.constant(filepaths)
ages = tf.constant(ages)

filepaths = filepaths.numpy()
ages = ages.numpy()

print("Total images:", len(filepaths))


# In[21]:


def load_and_preprocess_cnn(path, age):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (200, 200))

    img = tf.cast(img, tf.float32)
    img = img / 255.0

    return img, age

def load_and_preprocess_resnet(path, age):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))

    img = preprocess_input(img)

    return img, age


# In[23]:


X_train, X_temp, y_train, y_temp = train_test_split(filepaths, ages, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[24]:


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_ds_cnn = train_ds.map(load_and_preprocess_cnn).batch(32).prefetch(1)
val_ds_cnn   = val_ds.map(load_and_preprocess_cnn).batch(32).prefetch(1)
test_ds_cnn  = test_ds.map(load_and_preprocess_cnn).batch(32).prefetch(1)

train_ds_resnet = train_ds.map(load_and_preprocess_resnet).batch(32).prefetch(1)
val_ds_resnet   = val_ds.map(load_and_preprocess_resnet).batch(32).prefetch(1)
test_ds_resnet  = test_ds.map(load_and_preprocess_resnet).batch(32).prefetch(1)


# ### 3. CNN

# #### Modelling

# This model is a simple convolutional neural network designed to estimate age from images. The first part of the model uses four convolutional layers that apply filters to the image to detect features such as edges, shapes, and eventually facial details related to age. Batch normalization stabilizes the activations, and max-pooling reduces the spatial size, allowing the network to focus on the most important features while reducing computation.
# 
# After feature extraction, the data is flattened and passed into dense layers, which learn how the extracted features relate to the target value (age). Dropout is used to prevent overfitting by randomly disabling neurons during training. The final dense layer outputs a single number representing the predicted age. The model is trained using the Adam optimizer, and the loss function MAE measures how far predicted ages are from the true values.
# 

# In[9]:


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae', 'mse']
)

model.summary()


# #### Training

# Two extra mechanisms are used to help the model learn more efficiently and prevent overfitting. Early stopping monitors the validation loss and stops training automatically if the model stops improving for a number of epochs (10). It will then return the model with the lowest val_loss. Reduce learning rate on plateu is used for adjusting learning rate dynamically during training, which might help the model achieve better convergence. The model is then trained on 50 epochs (max).

# In[10]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    train_ds_cnn,
    validation_data=val_ds_cnn,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# #### Evaluation

# In[40]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(history.history['mae'], label='train_mae')
axes[0].plot(history.history['val_mae'], label='val_mae')
axes[0].set_title('Model Mean Absolute Error (MAE)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MAE')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['mse'], label='train_mse')
axes[1].plot(history.history['val_mse'], label='val_mse')
axes[1].set_title('Model Mean Square Error (MSE)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('assets/cnn_training_plot.png')

# The CNN model is able to learn quite effectively and efficiently, with its third epoch being the one taken (lowest val_loss). After the third epoch, the train and val losses diverged, which indicates that the model has stopped learning and is only memorizing the train data.

# In[ ]:


test_loss, test_mae, test_mse = model.evaluate(test_ds_cnn)

y_pred = model.predict(test_ds_cnn).flatten()

y_true = []
for _, labels in test_ds_cnn.unbatch():
    y_true.append(labels.numpy())
y_true = np.array(y_true)


# In[35]:


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred, alpha=0.2)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Age')
plt.grid(True)

plt.subplot(1, 2, 2)
errors = y_pred - y_true
plt.hist(errors, bins=50, edgecolor='green')
plt.xlabel('Error (years)')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.grid(True)

plt.tight_layout()

print(f"MAE: {test_mae:.2f} years")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {np.sqrt(test_mse):.2f} years")
plt.savefig('assets/cnn_prediction_plot.png')


# The left plot shows a clear positive relationship, meaning the model generally learns how age increases. Most younger and middle-aged predictions fall close to the diagonal line, but the spread grows for older ages. This means that the model becomes less accurate as age increases, an issue that could be caused by fewer samples and higher variability in that age range. The error distribution plot confirms this behavior. Most of the errors cluster around zero, but the center is slightly negative, meaning the model predicts younger ages more often than older ones. The majority of predictions fall within a reasonable error range, while a small number of large negative errors represent significant underestimations. Overall, the model performs consistently but shows a bias toward predicting lower ages, especially for older faces.

# ### 4. ResNet

# #### Modelling

# This model uses transfer learning by starting with a pretrained ResNet50 network instead of training a CNN from scratch. The ResNet50 loaded here comes with weights learned from ImageNet, which means it already knows how to detect general visual features such as edges, textures, shapes, and object structures.
# 
# Since it is originally a classification model, the classification layers are removed (include_top=False), and a custom regression head is added for it to be able to estimate ages. The ResNet base model is also froze to prevent the pretrained layers weights getting updated.
# 
# 
# GlobalAveragePooling2D compresses the spatial feature maps into a single vector, keeping the most important information while reducing the number of parameters. Two dense layers with ReLU activation learn patterns linking the extracted features to age, and dropout helps reduce overfitting by randomly deactivating neurons during training. The final dense layer outputs a single value representing the predicted age.
# 
# The custom head while keeping the powerful ResNet50 backbone fixed is expected to give much better performance and stability then the previous model (CNN).

# In[38]:


base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# regression head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1)(x)

resnet_model = Model(inputs=base_model.input, outputs=output)

resnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae', 'mse']
)

resnet_model.summary()


# #### Training

# In[39]:


early_stopping_resnet = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_resnet = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

history_resnet = resnet_model.fit(
    train_ds_resnet,
    validation_data=val_ds_resnet,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping_resnet, reduce_lr_resnet],
    verbose=1
)


# #### Evaluation

# In[41]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(history_resnet.history['mae'], label='train_mae')
axes[0].plot(history_resnet.history['val_mae'], label='val_mae')
axes[0].set_title('ResNet Model MAE')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MAE')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history_resnet.history['mse'], label='train_mse')
axes[1].plot(history_resnet.history['val_mse'], label='val_mse')
axes[1].set_title('ResNet Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('assets/resnet_training_plot.png')


# Compared to the CNN model, this looks more stable, achieving lower MAE and MSE. The train and val losses diverges early as well, at the ninth epoch.

# In[51]:


test_loss_resnet, test_mae_resnet, test_mse_resnet = resnet_model.evaluate(test_ds_resnet)

y_pred_resnet = resnet_model.predict(test_ds_resnet).flatten()

y_true = []
for _, labels in test_ds_resnet.unbatch():
    y_true.append(labels.numpy())

y_true = np.array(y_true)


# In[52]:


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred_resnet, alpha=0.2)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ResNet: Actual vs Predicted Age")
plt.grid(True)

plt.subplot(1, 2, 2)
errors_resnet = y_pred_resnet - y_true
plt.hist(errors_resnet, bins=50, edgecolor='green')
plt.xlabel("Error (years)")
plt.ylabel("Frequency")
plt.title("ResNet: Distribution of Prediction Errors")
plt.grid(True)

plt.tight_layout()

print(f"MAE: {test_mae_resnet:.2f} years")
print(f"MSE: {test_mse_resnet:.2f}")
print(f"RMSE: {np.sqrt(test_mse_resnet):.2f} years")
plt.savefig('assets/resnet_prediction_plot.png')


# Similarly, this model also have the same issue as the CNN model, often underestimating people's age. However, the errors appear more tightly concentrated around zero, suggesting that ResNet produces more consistent and accurate predictions overall, even though a few large errors still occur.

# #### Save Model

# In[44]:


resnet_model.save('resnet_age_prediction_model.h5')


# ### 5. Comparison

# In[45]:


comparison = pd.DataFrame({
    'Model': ['CNN', 'ResNet50'],
    'MAE': [test_mae, test_mae_resnet],
    'MSE': [test_mse, test_mse_resnet],
    'RMSE': [np.sqrt(test_mse), np.sqrt(test_mse_resnet)]
})

print(comparison)

plt.figure(figsize=(10, 6))

bar_width = 0.2
index = np.arange(len(comparison['Model']))

plt.bar(index, comparison['MAE'], bar_width, label='MAE')
plt.bar(index + bar_width, comparison['MSE'], bar_width, label='MSE')
plt.bar(index + 2 * bar_width, comparison['RMSE'], bar_width, label='RMSE')

plt.xlabel('Model')
plt.ylabel('Value')
plt.title('Model Performance Metrics Comparison')
plt.xticks(index + bar_width, comparison['Model'])
plt.legend()
plt.tight_layout()
plt.savefig('assets/model_metrics_plot.png')
# plt.show() # Commented out to prevent blocking script execution if run interactively


# Overall both models performed fairly well. The ResNet with regression head model performed better as expected, although the CNN architecture and the hyperparameters used are open for changes and the results could've been different. The model’s performance could be improved by tuning hyperparameters, adding regularization, or using techniques such as data augmentation to reduce overfitting. Additionally, experimenting with more advanced architectures or fine-tuning deeper layers of ResNet may further enhance prediction accuracy.
# 
