import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
# import matplotlib.pyplot as plt

keras = tf.keras
DEMO_IMAGE = "./archive/iris-versicolour/iris-1b0b5aabd59e4c6ed1ceb54e57534d76f2f3f97e0a81800ff7ed901c35a424ab.jpg"
MODEL = "Model/model-005-0.619048-0.975623.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224

st.title("Iris Classification")

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# st.write(img_file_buffer)


if img_file_buffer is not None:
    st.write("## Results")
    img = Image.open(img_file_buffer).convert('RGB')
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (IMG_HEIGHT, IMG_WIDTH))
    img_arr = np.asarray(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    st.image(img_file_buffer)
    
else:
    st.write("## Image Example")
    demo_image = DEMO_IMAGE
    img = Image.open(demo_image).convert('RGB')
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (IMG_HEIGHT, IMG_WIDTH))
    img_arr = np.asarray(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    st.image(opencvImage)


st.write(img_arr.shape)
stance_name = ["**Setosa**", "**Versicolour**", "**Virginica**"]

def prediction(image): 
    image = (image / 255.0)
    model = keras.models.load_model(MODEL)
    pred = model.predict(image)
    return pred

st.write(str(prediction(img_arr)[0]))
st.write("Prediction: " + stance_name[prediction(img_arr).argmax(axis=1).astype(int)[0]])
st.write("Confidence: " + str(prediction(img_arr).max()))