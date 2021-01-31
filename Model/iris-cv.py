import pandas as pd
# import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224
PATH = './archive'

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator( rescale = 1.0/255. )

training_ds = train_datagen.flow_from_directory(
    PATH, 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    class_mode='categorical', 
    batch_size=1
)

validation_ds = val_datagen.flow_from_directory(
    PATH, 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    class_mode='categorical', 
    batch_size=1
)

class_names = training_ds.classes
training_ds.samples
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=training_ds.image_shape), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(64, 3, activation='relu'), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(
            training_ds,
            steps_per_epoch=1, 
            validation_data = validation_ds,
            validation_steps=1,
            epochs = 20,
            verbose = 2)