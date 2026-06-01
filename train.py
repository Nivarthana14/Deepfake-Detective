import os
import cv2
import numpy as np
from model import create_model

# 📁 Dataset paths
REAL_PATH = "data/real"
FAKE_PATH = "data/fake"

IMG_SIZE = 224

# 🔹 Load images
def load_images(path, label):
    data = []
    labels = []

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0

        data.append(image)
        labels.append(label)

    return data, labels


# 🔹 Load dataset
real_data, real_labels = load_images(REAL_PATH, 0)
fake_data, fake_labels = load_images(FAKE_PATH, 1)

# Combine
X = np.array(real_data + fake_data)
y = np.array(real_labels + fake_labels)

# 🔹 Shuffle data
from sklearn.utils import shuffle
X, y = shuffle(X, y)

# 🔹 Create model
model = create_model()

# 🔹 Train
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# 🔹 Save model
model.save("model.h5")

print("✅ Training completed and model saved!")
