import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

# =========================
# KONFIGURASI
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "test")

MODEL_PATH = "model_urine_v2.h5"
GRAPH_ACC_PATH = "grafik_accuracy.png"
GRAPH_LOSS_PATH = "grafik_loss.png"
CM_PATH = "confusion_matrix.png"
CLASS_NAMES_PATH = "class_names.json"

# =========================
# CEK FOLDER
# =========================
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Folder tidak ditemukan: {TRAIN_DIR}")
if not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(f"Folder tidak ditemukan: {VAL_DIR}")

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

if train_data.samples == 0:
    raise ValueError("Folder dataset/train masih kosong.")
if val_data.samples == 0:
    raise ValueError("Folder dataset/test masih kosong.")

# =========================
# SIMPAN NAMA KELAS
# =========================
class_names = list(train_data.class_indices.keys())
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print("Class names:", class_names)
print("Class indices:", train_data.class_indices)

# =========================
# MODEL CNN (TRANSFER LEARNING)
# =========================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# =========================
# TRAINING
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Simpan model
model.save(MODEL_PATH)
print(f"Model tersimpan di: {MODEL_PATH}")

# =========================
# GRAFIK ACCURACY
# =========================
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy Training vs Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.savefig(GRAPH_ACC_PATH)
plt.show()

# =========================
# GRAFIK LOSS
# =========================
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss Training vs Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.savefig(GRAPH_LOSS_PATH)
plt.show()

# =========================
# CONFUSION MATRIX
# =========================
val_data.reset()
pred = model.predict(val_data)
y_pred = np.argmax(pred, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(CM_PATH)
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nSelesai!")