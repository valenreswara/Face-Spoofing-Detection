# Face Spoofing Detection

This project implements a deep learning solution for face spoofing detection using PyTorch and MobileNetV3. It includes model training, evaluation, ONNX export, and a Streamlit web app for real-time prediction.

## Features

- **Model Training:** Binary classification (real vs. spoofed faces) using MobileNetV3 backbone.
- **Evaluation:** Accuracy, confusion matrix, and visualization of wrong predictions.
- **ONNX Export:** Convert trained PyTorch model to ONNX format for fast inference.
- **Streamlit App:** Upload or capture face images and get spoofing predictions instantly.
- **Inference Comparison:** Script to compare inference speed between PyTorch and ONNX models.
