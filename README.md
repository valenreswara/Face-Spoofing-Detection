# 🕵️‍♂️ Face Spoofing Detection

A robust deep learning solution to distinguish between real and spoofed face images, built with PyTorch, ONNX, and Streamlit. This project enables fast, accurate, and interactive face liveness detection for anti-spoofing applications.

---

## 🚀 Features

- **State-of-the-Art Model:**  
  Utilizes MobileNetV3 as the backbone for lightweight and efficient binary classification (Real vs. Spoofed).

- **End-to-End Pipeline:**  
  Includes data preprocessing, stratified splitting, model training, evaluation, and ONNX export.

- **Interactive Web App:**  
  Deploys a user-friendly Streamlit app for real-time prediction from uploaded or captured images.

- **Fast Inference:**  
  Compare PyTorch and ONNX inference speeds with provided benchmarking scripts.

- **Visualization:**  
  Confusion matrix and wrong prediction visualization for model interpretability.

---

## 📁 Project Structure

```
Face Spoofing Project/
│
├── app.py                  # Streamlit web app for prediction
├── comparison.py           # PyTorch vs ONNX inference time comparison
├── model_creation.ipynb    # Model training and evaluation notebook
├── requirements.txt        # Python dependencies
├── .gitignore              # Files/folders to ignore in git
├── Face Dataset/           # (Ignored) Dataset folder
│   ├── Data Liveness All/real/  # Real images
│   └── Fake Filtered/fake/      # Spoofed images
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

- Place real images in: `Face Dataset/Data Liveness All/real/`
- Place spoofed images in: `Face Dataset/Fake Filtered/fake/`

### 3. Train & Export Model

- Run `model_creation.ipynb` to train the model and export:
  - `face_spoofing_mobilenet.pth` (PyTorch)
  - `image_classifier_model.onnx` (ONNX)

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```
- Upload or capture a face image to get instant predictions.

### 5. Benchmark Inference Speed

```bash
python comparison.py
```
- Compares inference time between PyTorch and ONNX models.

---

## 🧠 Model Details

- **Architecture:** MobileNetV3 Small
- **Input Size:** 224x224 RGB
- **Classes:** Real (0), Spoofed (1)
- **Export:** ONNX for cross-platform, high-speed inference

---

## 📊 Example Results

| Metric         | Value      |
|----------------|-----------|
| Accuracy       | ~95%      |
| Real-time Demo | Supported |

Confusion matrix and wrong prediction visualizations are available in the notebook.

---

## 🌐 Web App

[Click here to access the Streamlit app](http://valenreswara-face-spoofing.streamlit.app/)

---

## 🛠️ Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for full list

---
