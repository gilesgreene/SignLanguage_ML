import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np

def build_improved_model(input_shape=(64, 64, 3), num_classes=37):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    # Load preprocessed data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')

    # Build and compile the model
    model = build_improved_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

    # Save the trained model
    model.save('models/sign_language_model.h5')
    print("Model training completed and saved.")

def evaluate_model():
    # Load preprocessed data for evaluation
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')

    # Use a portion of the data as a test set
    X_test = X_train[-500:]
    y_test = y_train[-500:]

    # Load the trained model
    model = tf.keras.models.load_model('models/sign_language_model.h5')

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_model()
        elif sys.argv[1] == "evaluate":
            evaluate_model()
        else:
            print("Invalid argument. Use 'train' or 'evaluate'.")
    else:
        print("Please specify 'train' or 'evaluate'.")
