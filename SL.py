import os
import time
import shutil
import pathlib
import itertools
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

data_path = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train" 

images = []
labels = []

for subfolder in os.listdir(data_path):
    
    subfolder_path = os.path.join(data_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
  
    for image_filename in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_filename)
        images.append(image_path)
    
        labels.append(subfolder)
 
data = pd.DataFrame({'image': images, 'label': labels})

data.head()
strat = data['label']
train_df, dummy_df = train_test_split(data,  train_size= 0.80, shuffle= True, random_state= 123, stratify= strat)

strat = dummy_df['label']
valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 123, stratify= strat)
batch_size = 32
img_size = (150, 150)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='image', y_col='label', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='image', y_col='label', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='image', y_col='label', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def cct_model(input_shape=(150, 150, 3), num_classes=29):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Reshape((-1, 256))(x)

    transformer_block = layers.MultiHeadAttention(num_heads=4, key_dim=64)
    x = transformer_block(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = cct_model(input_shape=(150, 150, 3), num_classes=29)
model.summary()
epochs = 5

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    verbose=1
)

loss, accuracy = model.evaluate(test_gen)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

model_name = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)
feature_extractor = ViTFeatureExtractor()

def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def predict(image_path):
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1)
    return predicted_class

image_path = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A10.jpg'  
predicted_class = predict(image_path)
print(f'Predicted class: {predicted_class.item()}')
model_name1 = 'google/vit-base-patch32-224-in21k'
model1 = ViTForImageClassification.from_pretrained(model_name1, num_labels=10)
feature_extractor1 = ViTFeatureExtractor()

def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor1(images=image, return_tensors="pt")
    return inputs

def predict(image_path):
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model1(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1)
    return predicted_class

image_path = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A10.jpg'  
predicted_class = predict(image_path)
print(f'Predicted class: {predicted_class.item()}')

