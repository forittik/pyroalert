# fire_detection_yolo_app.py

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Load the YOLO model
yolo_model = YOLO('best.onnx')  # Update the path as needed

# Configure Streamlit app
st.title("🔥 YOLO Fire Detection App")
st.write("Upload images or videos to detect fire using YOLO.")

# File upload option
file_type = st.sidebar.selectbox("Choose the type of file to upload", ["Image", "Video"])

# Define YOLO fire detection function
def detect_fire_yolo(input_path, is_video=False):
    results = yolo_model.predict(source=input_path, save=True)  # Run YOLO detection
    if is_video:
        return results[0].save_path  # Return path to processed video
    return results[0].orig_img  # Return processed image with bounding boxes

# Image processing
if file_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png"])
    if uploaded_image is not None:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_image.read())
            image_path = tmp.name

        # Display original image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        # Run fire detection
        if st.button("Detect Fire"):
            result_image = detect_fire_yolo(image_path)
            st.image(result_image, caption="Detection Result", use_column_width=True)

# Video processing
elif file_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4"])
    if uploaded_video is not None:
        # Save the uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        # Display original video
        st.video(video_path)

        # Run fire detection
        if st.button("Detect Fire in Video"):
            output_video_path = detect_fire_yolo(video_path, is_video=True)
            st.video(output_video_path)
            st.write("Processed video saved at:", output_video_path)
