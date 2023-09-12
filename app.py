import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Digit-recognition Images")

st.write("Predict the sport that is being represented in the image.")

model = load_model("model.h5")
labels = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9'   
    }












uploaded_file = st.file_uploader(
    "Upload an image:", type='jpg'
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open("/content/drive/MyDrive/DL/Kaggle competition Datasets/train.csv")
    image1=image.smart_resize(image1,(28,28))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open("/content/drive/MyDrive/DL/Kaggle competition Datasets/train.csv")
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("/content/drive/MyDrive/DL/Kaggle competition Datasets/train.csv")
    image1=image.smart_resize(image1,(28,28))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]
    image1 = Image.open("/content/drive/MyDrive/DL/Kaggle competition Datasets/train.csv")
    st.image(image1, caption="Uploaded Image", use_column_width=True)    
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
