from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

app = Flask(__name__)

# =========================
# LOAD MODEL & CLASS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_urine_v2.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

print("Model path :", MODEL_PATH)
print("Class path :", CLASS_PATH)

model = load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# =========================
# FOLDER UPLOAD
# =========================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# HALAMAN UTAMA
# =========================
@app.route("/")
def index():
    return render_template("index.html")

# =========================
# PREDIKSI
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "Tidak ada file yang dipilih"

    file = request.files["file"]

    if file.filename == "":
        return "Nama file kosong"

    # Simpan file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        print("✅ File berhasil disimpan:", filepath)

    except Exception as e:
        print("❌ Gagal menyimpan file:", e)
        return "Gagal menyimpan file"

    # Preprocessing gambar
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    print("Prediksi :", predicted_class)
    print("Confidence :", confidence)

    # Path gambar untuk HTML
    image_path = f"uploads/{filename}"

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image_path=image_path
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )