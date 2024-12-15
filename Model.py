import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

path = r"C:\Users\kkati\Personal Projects\SignLanguageRecognition\asl_dataset"
img_size = 400
batch_size = 32
epochs = 20

categories = os.listdir(path)
categories.sort

images = []
labels = []

label_map = {category: label for label, category in enumerate(categories)}

for category in categories:
    category_path = os.path.join(path, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            image_path = os.path.join(category_path, file)

            img = cv2.imread(image_path)
            images.append(img)
            labels.append(label_map[category])

images = np.array(images) / 255
labels = np.array(labels)
labels = to_categorical(labels, num_classes = 36)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
model.save('asl_model.h5')


