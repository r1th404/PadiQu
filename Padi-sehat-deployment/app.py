import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

# Load metadata
with open("Padi-sehat-deployment/model/model_metadata.json", "r") as f:
    metadata = json.load(f)

label_map = metadata["label_map"]
img_size = metadata["input_shape"][:2]

# Load model
model = load_model("Padi-sehat-deployment/model/best_model_mobilenetv2.keras")

# Page config
st.set_page_config(page_title="PadiSehat AI", layout="centered")

# Custom minimalistic style
st.markdown("""
    <style>
        .title { 
            font-size: 32px; 
            font-weight: bold; 
            color: #2b7a0b;
        }
        .subtitle {
            font-size: 18px;
            color: #333;
        }
        .result {
            padding: 10px;
            background: #e9ffe6;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            color: #2b7a0b;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌾 PadiSehat AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi penyakit daun padi secara cepat.</div><br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload foto daun padi", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(img_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300, caption="Gambar yang diupload")

    img_processed = preprocess_image(img)
    preds = model.predict(img_processed)[0]

    class_id = np.argmax(preds)
    class_name = metadata["class_names"][str(class_id)]
    disease_name = label_map[class_name]

    st.markdown(f"<div class='result'>{disease_name}</div>", unsafe_allow_html=True)
