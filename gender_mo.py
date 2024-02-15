import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
import os
from PIL import Image
import pandas as pd
import csv

# Define the path to your image directory
image_directory = "unsamples"

# Load image data from the image directory and resize them to (64, 64)
loaded_images = []
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):  # Change the file extension to match your image files
        image = Image.open(os.path.join(image_directory, filename))
        image = image.resize((64, 64))  # Resize the image to (64, 64)
        loaded_images.append(np.array(image))

images = np.array(loaded_images)

# Load gender labels from a CSV file
gender_labels_file = 'age_gender.csv'  # Replace with the actual file path
genders = []

with open(gender_labels_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        genders.append(int(row['gender']))

# Convert gender labels to binary labels (0 for one gender, 1 for the other)
# For example, if you want to classify males as 0 and females as 1:
genders = [0 if gender == 0 else 1 for gender in genders]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, genders, test_size=0.2, random_state=42)

# Convert the data to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Create a simple convolutional neural network (CNN) model
input_shape = (64, 64, 3)

model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output layer for gender classification (softmax for binary classification)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Function to get gender prediction (0 for Male, 1 for Female)
def get_gender_prediction(image):
    image = np.array(image)
    image = cv2.resize(image, (64, 64))  # Resize the image to (64, 64)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)  # Add a batch dimension
    prediction = model.predict(image)
    predicted_gender = 'Male' if np.argmax(prediction) == 0 else 'Female'
    return predicted_gender

# Example usage
image_to_predict = Image.open('aiony-haust-jmATI5Q_YgY-unsplash.jpg')  # Replace with the path to the image you want to predict
predicted_gender = get_gender_prediction(image_to_predict)
print(f'Predicted Gender: {predicted_gender}')
