#!/usr/bin/env python
# coding: utf-8

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

X_train, X_temp, y_train, y_temp = train_test_split(filepaths, ages, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_ds_cnn = train_ds.map(load_and_preprocess_cnn).batch(32).prefetch(1)
val_ds_cnn   = val_ds.map(load_and_preprocess_cnn).batch(32).prefetch(1)
test_ds_cnn  = test_ds.map(load_and_preprocess_cnn).batch(32).prefetch(1)

train_ds_resnet = train_ds.map(load_and_preprocess_resnet).batch(32).prefetch(1)
val_ds_resnet   = val_ds.map(load_and_preprocess_resnet).batch(32).prefetch(1)
test_ds_resnet  = test_ds.map(load_and_preprocess_resnet).batch(32).prefetch(1)

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

test_loss, test_mae, test_mse = model.evaluate(test_ds_cnn)

y_pred = model.predict(test_ds_cnn).flatten()

y_true = []
for _, labels in test_ds_cnn.unbatch():
    y_true.append(labels.numpy())
y_true = np.array(y_true)

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

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

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

test_loss_resnet, test_mae_resnet, test_mse_resnet = resnet_model.evaluate(test_ds_resnet)

y_pred_resnet = resnet_model.predict(test_ds_resnet).flatten()

y_true = []
for _, labels in test_ds_resnet.unbatch():
    y_true.append(labels.numpy())

y_true = np.array(y_true)

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

resnet_model.save('resnet_age_prediction_model.h5')

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
