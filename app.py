import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Emotion Analyzer", layout="centered")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Advanced Face Emotion Analyzer")
st.write("Let's discover the emotions behind your expression!")

option = st.radio("How would you like to provide your image?", ("Use Camera", "Upload Image"))

def analyze_emotions(image_np):
    try:
        result = DeepFace.analyze(
            img_path=image_np,
            actions=['emotion'],
            detector_backend='retinaface',
            enforce_detection=True,
            align=True
        )
        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']

        st.image(image_np, caption="Here's what I analyzed!", use_column_width=True)

        st.subheader("Dominant Emotion Detected")
        st.success(f"You seem to be feeling **{dominant_emotion.capitalize()}**.")

        st.subheader("Emotion Breakdown")
        emotion_df = pd.DataFrame.from_dict(emotions, orient='index', columns=['Intensity'])
        emotion_df = emotion_df.sort_values(by='Intensity', ascending=False)

        for emo, val in emotion_df['Intensity'].items():
            st.write(f"**{emo.capitalize()}**: {val:.2f}%")

        st.bar_chart(emotion_df)

    except Exception as e:
        st.error(" Hmm, I couldn't quite read the emotions. Here's what went wrong:")
        st.code(str(e), language='text')

if option == "Use Camera":
    img = st.camera_input("Take a picture to analyze your emotion:")
    if img is not None:
        image = Image.open(img)
        image_np = np.array(image)
        analyze_emotions(image_np)
else:
    uploaded_file = st.file_uploader("Upload a photo (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        analyze_emotions(image_np)
