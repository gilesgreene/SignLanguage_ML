import cv2
import tensorflow as tf
import cvzone
import numpy as np

model = tf.keras.models.load_model('asl_model.h5')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

categories = ["0", "1", "2", "3", "4", "5",
              "6", "7", "8", "9", "a", "b",
              "c", "d", "e", "f", "g", "h",
              "i", "j", "k", "l", "m", "n",
              "o", "p", "q", "r", "s", "t",
              "u", "v", "w", "x", "y", "z"]

img_size = 400

while True:
    success, img = cap.read()

    if not success:
        print("Failed to grab frame from the webcam.")
        break

    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_normalized = np.uint8(img_normalized * 255)
    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0)

    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction)
    print(f"Predicted class index: {predicted_class}")
    print(f"Categories length: {len(categories)}")

    if predicted_class >= 0 and predicted_class < len(categories):
        predicted_label = categories[predicted_class]
    else:
        print(f"Error: Predicted class index {predicted_class} is out of range.")
        predicted_label = "Unknown"

    text = f"Gesture: {predicted_label}"
    img_with_text, _ = cvzone.putTextRect(img, text, (50, 50), scale=2, colorR=(0, 255, 0), thickness=2,
                                          colorT=(255, 255, 255))

    # Debugging: Check if img_with_text is a valid NumPy array
    print(f"img_with_text type: {type(img_with_text)}")
    print(f"img_with_text shape: {img_with_text.shape}")

    if isinstance(img_with_text, np.ndarray):
        cv2.imshow("Real-Time ASL Gesture Detection", img_with_text)
    else:
        print("Error: img_with_text is not a valid image.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
