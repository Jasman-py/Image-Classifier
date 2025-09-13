import cv2 as cv
import numpy as np
import json
from tensorflow.keras.models import load_model

# --- Load model and metadata ---
model = load_model('model/image_classifier.keras')
with open('model/metadata.json', 'r') as f:
    metadata = json.load(f)

class_names = metadata["class_names"]

# --- Ask user for image ---
image_path = input("Enter image path: ")

try:
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")

    img = cv.resize(img, (32, 32))             # Resize to match training size
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   # Convert to RGB
    img = img.astype("float32") / 255.0        # Normalize
    img = np.expand_dims(img, axis=0)          # Make it (1, 32, 32, 3)

    prediction = model.predict(img)
    index = np.argmax(prediction)

    print(f"✅ Prediction: {class_names[index]}")
except Exception as e:
    print(f"❌ Error: {e}")
