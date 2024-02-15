import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained models for age and gender detection
age_model = load_model('age_model.keras')
gender_model = load_model('gender_model.keras')

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Function to detect and classify age and gender
def detect_age_gender(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(1, 1))

    # Process each detected face
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]

        # Preprocess the face for age and gender prediction
        face = cv2.resize(face, (64, 64))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        # Predict age
        age_preds = age_model.predict(face)
        age = int(age_preds[0][0] * 100)

        # Predict gender
        gender_preds = gender_model.predict(face)
        gender = "Male" if gender_preds[0][0] > 0.5 else "Female"

        # Draw a rectangle around the face and display age and gender
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Age: {age}, Gender: {gender}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image with results
    cv2.imshow('Age and Gender Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'ak.jpg'  # Replace with the path to your image
detect_age_gender(image_path)
