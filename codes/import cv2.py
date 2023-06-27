import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained model for emotion classification
emotion_model = load_model('C:\\Users\\student\\AppData\\Local\\Programs\\Python\\Python311\\Scripts\\my_model3.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Read the input image
path = r'C:\Users\student\Desktop\codes\091ed75b-8f3b-4db0-8073-d64fe5989974.jpeg'
img = cv2.imread(path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Process each face
for (x, y, w, h) in faces:
    # Extract the face region
    face_img = gray[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, (48, 48))  
    face_img = np.expand_dims(face_img, axis=0) 
    face_img = np.expand_dims(face_img, axis=-1)  
    # Predict the emotion
    emotion_scores = emotion_model.predict(face_img)
    emotion_index = np.argmax(emotion_scores[0])
    emotion_label = emotion_labels[emotion_index]

    # Draw rectangle and label for the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 9)
    cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the output image
cv2.imshow('Facial Expression Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
