import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_data():
    RAW_DATA_DIR = 'data/raw/'
    PROCESSED_DATA_DIR = 'data/processed/'

    # Initialize data lists
    X = []
    y = []

    # Set up data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Iterate through each class folder in raw data directory
    for class_label, class_name in enumerate(os.listdir(RAW_DATA_DIR)):
        class_dir = os.path.join(RAW_DATA_DIR, class_name)

        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class folder: {class_name}")

        # Iterate through each image in the class folder
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # Load image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            # Resize the image to the target size (64x64)
            img = cv2.resize(img, (64, 64))

            # Convert the image to RGB (OpenCV loads images in BGR format by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize the pixel values to [0, 1]
            img = img / 255.0

            # Append original image to dataset
            X.append(img)
            y.append(class_label)

            # Apply data augmentation to generate more images
            img_expanded = np.expand_dims(img, axis=0)  # Expand dims for ImageDataGenerator
            augmented_iter = datagen.flow(img_expanded, batch_size=1)

            # Generate a few augmented images (e.g., 5 for each original image)
            for _ in range(5):
                aug_img = next(augmented_iter)[0]  # Extract augmented image from the iterator
                X.append(aug_img)
                y.append(class_label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Save the processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y)

    print(f"Data preprocessing completed. Total images: {len(X)}")


if __name__ == '__main__':
    preprocess_data()
