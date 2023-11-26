import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Load the pre-trained model
model = load_model('mnist.h5')

def preprocess_image(image):
    # Convert the image to grayscale
    gray = image.convert('L')
    
    # Resize the image to (28, 28)
    resized = gray.resize((28, 28))
    
    # Convert the image to a numpy array and normalize
    img_array = np.array(resized) / 255.0
    
    # Reshape to (1, 28, 28, 1) as the model expects a batch of images
    img_array = img_array.reshape((1,28, 28, 1))
    
    return img_array

def predict_digit(image):
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction using the model
    pred = model.predict(preprocessed_image)[0]
    
    # Get the index of the highest predicted value
    final_pred = np.argmax(pred)
    
    return final_pred

picture = st.camera_input("Take a picture")
if "myImage" not in st.session_state.keys():
    st.session_state["myImage"] = None

if picture:
    st.session_state["myImage"] = picture
    st.image(picture)
    # st.subheader(picture)
    fileName = st.session_state["myImage"].name
    save = st.button("Save")

    if save:
        with open(fileName, "wb") as imageFile:
            sa = imageFile.write(st.session_state["myImage"].getbuffer())
            if sa:
                # s_butt = st.button("Predict")
                st.header(fileName)
                
                    # Load the saved image
                saved_image = Image.open(fileName)
                st.header("ok image")
                image = preprocess_image(saved_image)
                pred = model.predict(image)
                final_pred = np.argmax(pred)
                st.header(final_pred)

