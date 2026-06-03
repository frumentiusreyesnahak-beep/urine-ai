import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# =========================
# LOAD MODEL
# =========================
model = load_model("model_urine_v2.h5")

# Load nama kelas
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# INPUT GAMBAR
# =========================
img_path = "test_image.jpg"  # ganti dengan gambar kamu

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# =========================
# PREDIKSI
# =========================
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Hasil Prediksi:", predicted_class)
print("Akurasi:", round(float(confidence) * 100, 2), "%")