import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Folder upload
UPLOAD_FOLDER = "Padi-sehat-deployment/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path model dan metadata
MODEL_PATH = "Padi-sehat-deployment/model/best_model_mobilenetv2.keras"
META_PATH = "Padi-sehat-deployment/model/model_metadata.json"

# Load model keras
model = load_model(MODEL_PATH)

# Load metadata
with open(META_PATH, "r") as f:
    metadata = json.load(f)

class_indices = metadata["class_indices"]
class_names = metadata["class_names"]
label_map = metadata["label_map"]

# Input size harus 224x224x3
IMG_SIZE = (224, 224)

def preprocess_image(filepath):
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", prediction="No image uploaded")

        # Simpan gambar
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        image_path = filepath

        # Preprocess
        img = preprocess_image(filepath)

        # Predict
        preds = model.predict(img)
        pred_idx = np.argmax(preds[0])

        # Ambil nama class (English)
        class_name = class_names[str(pred_idx)]

        # Ubah ke bahasa Indonesia
        prediction = label_map[class_name]

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
