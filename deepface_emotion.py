import cv2
import numpy as np
from keras.models import load_model
import pygame

# Initialize pygame
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)

# Load model
model = load_model('model.hdf5', compile=False)

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    r'E:\emotion-detector\haarcascade_frontalface_default.xml'
)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion → Music mapping
emotion_music = {
    "Happy": r"E:\emotion-detector\songs\happy.mp3",
    "Sad": r"E:\emotion-detector\songs\sad.mp3",
    "Angry": r"E:\emotion-detector\songs\angry.mp3",
    "Neutral": r"E:\emotion-detector\songs\neutral.mp3",
    "Surprise": r"E:\emotion-detector\songs\surprise.mp3",
    "Fear": r"E:\emotion-detector\songs\sad.mp3",
    "Disgust": r"E:\emotion-detector\songs\angry.mp3"
}

# Track current emotion
current_emotion = ""

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working ❌")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Resize to model size
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        # Predict emotion
        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        # DEBUG: print emotion
        print("Raw Emotion:", emotion)

        # 🎵 Play song when emotion changes
        if emotion != current_emotion:
            print("Switching to:", emotion)
            current_emotion = emotion

            if emotion in emotion_music:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(emotion_music[emotion])
                    pygame.mixer.music.play()
                except Exception as e:
                    print("Music Error:", e)

        # Draw rectangle + emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # Show last emotion if face disappears
    if current_emotion:
        cv2.putText(frame, current_emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    cv2.imshow("Emotion Music Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()