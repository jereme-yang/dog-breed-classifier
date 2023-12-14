import streamlit as  st
import cv2
import numpy as np
from main import load_model, process_image, find_label, SingletonClass
from constants import *

# title and description
st.title("Dog Image Classifier")
st.write("CNN for image classification with respect to dog breeds. For code + methodology, see executive summary: https://docs.google.com/document/d/1tSKfv9cTl6AeA-zaYEW3tdgV9THCdUu6uzYBe3fwFjs/edit?usp=sharing")

# load model
singleton = SingletonClass()
if singleton.model is None:
    with st.spinner("Loading..."):
        singleton.model = load_model()
model = singleton.model



st.title("Upload an image")
# Upload file through Streamlit widget
uploaded_file = st.file_uploader("Choose a file")

# Check if a file was uploaded
if uploaded_file is not None:
    # Display the file info
    file_contents = uploaded_file.read()
    # np_array = np.frombuffer(file_contents, np.uint8)
    np_array = np.frombuffer(file_contents, np.uint8)
    im_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    processed_image = cv2.resize(im_array, (331,331), interpolation=cv2.INTER_AREA)
    st.image(processed_image, channels='RGB', caption="Uploaded Image", use_column_width=True)
if st.button("Go!"):
    with st.spinner("Making prediction..."):
        image_for_model = process_image(im_array)
        prediction_array = model.predict(image_for_model, verbose=1)
        ans_ind = np.argmax(prediction_array, axis=1)
        pred_dog_name = CATEGORIES[ans_ind[0]]
    st.title(f"It is likely that your image is a {pred_dog_name}")
    st.image(find_label(pred_dog_name), caption="predicted image", use_column_width=True)