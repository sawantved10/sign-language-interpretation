import os
import cv2
import numpy as np

data = []
labels = []

dataset_path = r"E:\sign lan\Sign-Language-To-Text-and-Speech-Conversion-master\AtoZ_3.1"

for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        data.append(img)
        labels.append(ord(label) - ord('A'))

data = np.array(data) / 255.0
data = data.reshape(-1, 224, 224, 1)
labels = np.array(labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")

model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
model.save("sign_model.h5")