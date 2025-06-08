import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# Load VGG19 model
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))

x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
dropout = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(dropout)
output = Dense(2, activation='softmax')(class_2)

model_03 = Model(base_model.inputs, output)

# Load trained weights
model_03.load_weights("vgg19_model_01.h5")


# Helper function
def get_class_name(class_no):
    return "Normal" if class_no == 0 else "Pneumonia"

def get_result(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_03.predict(img_array)
    result = np.argmax(prediction, axis=1)
    return get_class_name(int(result[0]))

# Streamlit UI
st.set_page_config(page_title="PNEUMONIA DETECTION", layout="centered")
st.title("🩺 PNEUMONIA DETECTION Using Deep Learning")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    with st.spinner("Predicting..."):
        result = get_result(uploaded_file)
    st.success(f"Prediction: **{result}**")
