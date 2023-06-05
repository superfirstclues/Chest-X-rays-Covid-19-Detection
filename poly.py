import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# Load the pre-trained model
model = keras.models.load_model('C:\\Users\\firstclues\\Desktop\\covid\\covid_model\\mobilenet.h5')
data=[]
labels=[]
# Define text labels for the output classes
cxr_labels = ['Normal','COVID-19','Viral']


# Define a function for making a prediction
def predict(image):
    # Resize and preprocess the image
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)


    # Make the prediction
    prediction = model.predict(image)[0]
    predicted_label = np.argmax(prediction)


    # Return the predicted label and the model accuracy
    accuracy = prediction[predicted_label] * 100
    return cxr_labels[predicted_label], accuracy


# Define the app layout
st.set_page_config(page_title='COVID-19 Prediction App')
st.title('COVID-19 Prediction App')
st.write('Upload a chest X-ray image and we will predict whether it shows signs of COVID-19 or not.')


# Create a file uploader for uploading the X-ray image
uploaded_file = st.file_uploader('Choose a chest X-ray image', type=['jpg', 'jpeg', 'png'])


# If an image has been uploaded, display it and make a prediction on it
if uploaded_file is not None:
    # Read the image and display it
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)


    # Make a prediction on the image and display the result
    result, accuracy = predict(image)
    st.write('Prediction: ', result)
    st.write('Accuracy: ', accuracy, '%')


# If no image has been uploaded yet, display a message prompting the user to upload one
else:
    st.write('Please upload a chest X-ray image to make a prediction.')

import streamlit as st

# Add HTML code to change the background color
st.markdown(
    """
    <style>
    body {
        background-color: cyan;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Continue with your Streamlit app code below

