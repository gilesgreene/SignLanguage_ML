import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

def run_camera():
    # Load the trained model
    model = tf.keras.models.load_model('models/sign_language_model.h5')

    # Define a dictionary to map the model's predicted class to the corresponding sign/label
    label_map = {i: chr(65 + i) for i in range(26)}  # Labels for A-Z
    label_map.update({26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: 'Nothing'})

    # Mediapipe hands initialization
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Flip the frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB (as Mediapipe works with RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            results = hands.process(rgb_frame)

            # If hands are detected, proceed with gesture recognition
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame for visualization
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get bounding box coordinates of the detected hand
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

                    # Convert bounding box coordinates to integers
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                    # Add padding to ensure the entire hand is captured
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)

                    # Extract the hand region of interest (ROI)
                    roi = frame[y_min:y_max, x_min:x_max]

                    # Only proceed if ROI is non-empty
                    if roi.size > 0:
                        # Preprocess the ROI for prediction
                        roi = cv2.resize(roi, (64, 64))
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi = roi / 255.0
                        roi = np.expand_dims(roi, axis=0)

                        # Make a prediction
                        predictions = model.predict(roi)
                        predicted_class = np.argmax(predictions)
                        predicted_label = label_map[predicted_class]
                        confidence = np.max(predictions) * 100

                        # Draw a bounding box around the hand
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Display the predicted label and confidence on the video frame
                        cv2.putText(frame, f'Prediction: {predicted_label} ({confidence:.2f}%)', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the frame with the prediction
            cv2.imshow('Sign Language Recognition', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()
