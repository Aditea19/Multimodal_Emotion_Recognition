import streamlit as st
st.title('Multimodal Emotion Recognition - Demo (TensorFlow)')
st.write('This demo is a scaffold. Add model loading and inference logic to use it.')
st.write('Place your model weights under the `models/` folder and implement prediction.')
st.header('Image input')
st.file_uploader('Upload an image', type=['jpg','png','jpeg'])
st.header('Audio input')
st.file_uploader('Upload a wav file', type=['wav'])
st.info('Notebooks are intentionally blank (section titles only) per your request.')
