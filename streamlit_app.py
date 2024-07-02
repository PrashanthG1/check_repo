import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
MODEL_PATH = 'chest_xray.h5'
model = load_model(MODEL_PATH)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #E67E22 ;
        }
        .title {
            font-family: 'Times New Roman', Gadget, sans-serif;
            color: #17202A ;
            text-align: center;
            font-size:38px;
            text-decoration:underline;
        }
        .subtitle {
            font-family: 'Times New Roman', sans-serif;
            color: #17202A ;
            text-align: center;
            font-size: 22px;
        }
        .result {
            font-family: 'Times New Roman', sans-serif;
            color: #17202A ;
            text-align: center;
            font-size: 22px;
            text-decoration: underline;
        }
        .normal {
            font-family: 'Times New Roman', sans-serif;
            color:#17202A ;
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.markdown('<h1 class="title">Pneumonia Detection from Chest X-rays</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a chest X-ray image to detect if the person has pneumonia or not.</p>', unsafe_allow_html=True)

# Image upload
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpeg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded X-ray image.', use_column_width=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    # Prediction
    classes = model.predict(img_data)
    result = int(classes[0][0])

    # Display the result with styling
    if result == 0:
        st.markdown('<p class="result">Prediction: The person is affected by <span class="normal">PNEUMONIA</span>.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result normal">Prediction: The result is NORMAL.</p>', unsafe_allow_html=True)
