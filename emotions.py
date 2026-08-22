import cv2
from deepface import DeepFace
import pygame
import time

# Initialize music
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)

# Emotion → Music mapping
emotion_music = {
    "happy": r"E:\emotion-detector\songs\happy.mp3",
    "sad": r"E:\emotion-detector\songs\sad.mp3",
    "angry": r"E:\emotion-detector\songs\angry.mp3",
    "neutral": r"E:\emotion-detector\songs\neutral.mp3",
    "surprise": r"E:\emotion-detector\songs\surprise.mp3",
    "fear": r"E:\emotion-detector\songs\sad.mp3",
    "disgust": r"E:\emotion-detector\songs\angry.mp3"
}

current_emotion = ""
last_switch_time = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        print("Detected:", emotion)

        # ⏱️ Only allow change every 3 seconds
        if emotion != current_emotion and time.time() - last_switch_time > 3:
            print("Switching to:", emotion)

            current_emotion = emotion
            last_switch_time = time.time()

            if emotion in emotion_music:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(emotion_music[emotion])
                    pygame.mixer.music.play()
                except Exception as e:
                    print("Music Error:", e)

        # Display emotion
        cv2.putText(frame, emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Music Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()