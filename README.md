# pyroalert
# 🔥 YOLO Fire Detection App

This project is a Streamlit web application that uses a YOLO (You Only Look Once) deep learning model to detect fire in **images** and **videos**. It provides a simple user interface where you can upload media files and get visual results with bounding boxes highlighting detected fire regions.

## 🚀 Features

- 🔍 Real-time fire detection using YOLO model
- 🖼 Upload and process images
- 🎥 Upload and analyze video files
- ✅ Interactive and clean Streamlit UI
- 🧠 Powered by a pre-trained ONNX model (`best.onnx`)

## 📦 Requirements

- Python 3.8+
- `streamlit`
- `opencv-python`
- `ultralytics`
- `onnxruntime`
- `tempfile` (standard library)

Install dependencies using:

```bash
pip install streamlit opencv-python ultralytics
```

## Model
The app uses a custom-trained YOLO model (exported to ONNX format) to detect fire in frames. You can train your own YOLO model or use a public one trained on fire datasets.

streamlit run pyroalert.py

## Usage
1. Select Image or Video from the sidebar.
2. Upload your file.
3. Click Detect Fire.
4. View the output with fire regions highlighted.
