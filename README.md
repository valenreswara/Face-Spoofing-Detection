# Face Spoofing Detection

This project implements a deep learning solution for face spoofing detection using PyTorch and MobileNetV3. It includes model training, evaluation, ONNX export, and a Streamlit web app for real-time prediction.

## Features

- **Model Training:** Binary classification (real vs. spoofed faces) using MobileNetV3 backbone.
- **Evaluation:** Accuracy and confusion matrix.
- **ONNX Export:** Convert trained PyTorch model to ONNX format for fast inference.
- **Streamlit App:** Upload or capture face images and get spoofing predictions instantly.

## Folder Structure

```
Face Spoofing Project/
│
├── app.py                  # Streamlit web app for prediction
├── model_creation.ipynb    # Model training and evaluation notebook
├── requirements.txt        # Python dependencies
├── .gitignore              # Files/folders to ignore in git
├── Face Dataset/           # (Ignored) Dataset folder
│   ├── Data Liveness All/  # Real images
│   └── Fake Filtered/      # Spoofed images
```

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Place real images in `Face Dataset/Data Liveness All/real/`
   - Place spoofed images in `Face Dataset/Fake Filtered/fake/`

3. **Train the model:**
   - Run `model_creation.ipynb` to train and export the model (`face_spoofing_mobilenet.pth` and `image_classifier_model.onnx`).

4. **Run the Streamlit app:**
   ```
   streamlit run app.py
   ```
   - Upload a face image to get a prediction.

## Access the deployed Streamlit link
[Click here to access the deployed link](https://valenreswara-face-spoofing.streamlit.app/)

## Model

- **Architecture:** MobileNetV3 Small (PyTorch)
- **Classes:** Real (0), Spoofed (1)
- **Export:** ONNX for fast deployment
