import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("model.h5")

print("Reading image...")
img = cv2.imread("test.jpg")

if img is None:
    print("❌ ERROR: test.jpg not found or cannot read")
    exit()

img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

print("Predicting...")
prediction = model.predict(img)

print("Raw prediction:", prediction)

if prediction > 0.5:
    print("⚠️ Leukemia Detected")
else:
    print("✅ Normal")