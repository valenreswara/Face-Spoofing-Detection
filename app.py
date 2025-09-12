import streamlit as st
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image

# Load ONNX model
@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession("image_classifier_model.onnx", providers=["CPUExecutionProvider"])
    return session

session = load_onnx_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Face Spoofing Detection")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image/"):
        try:
            image = Image.open(uploaded_file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).numpy()  # Convert to numpy

            # ONNX expects float32
            img_tensor = img_tensor.astype(np.float32)

            # Predict
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_tensor})
            logits = outputs[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax

            pred = np.argmax(logits, axis=1)[0]
            label_map = {0: "Real", 1: "Spoofed"}
            st.success(f"Prediction: Real: {probs[0][0]*100:.2f}%, Spoofed: {probs[0][1]*100:.2f}%")
            st.image(image, caption=f"Uploaded {uploaded_file.name}", width='stretch')
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.error("Uploaded file is not a valid image.")