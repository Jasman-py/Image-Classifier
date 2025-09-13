import json
import numpy as np
from tensorflow.keras import datasets, layers, models

# --- 1. Load + preprocess data ---
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reduce data for faster training (optional)
train_images, train_labels = train_images[:20000], train_labels[:20000]
test_images, test_labels = test_images[:4000], test_labels[:4000]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 2. Build the CNN model ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. Train the model ---
print("🚀 Training model...")
history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

# --- 4. Evaluate model ---
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")

# --- 5. Save model + metadata ---
import os
os.makedirs("model", exist_ok=True)

model.save('model/image_classifier.keras')

metadata = {
    "input_shape": [32, 32, 3],
    "num_classes": 10,
    "class_names": class_names,
    "train_samples": len(train_images),
    "test_samples": len(test_images)
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("📁 Model and metadata saved to 'model/' folder.")
