import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import time

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="/home/pi/Emotion-detection/model_mobilenet_4class.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the emotion labels
# emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotions = ('happy', 'neutral', 'sad', 'surprise')

count_happy = 0
count_neutral = 0
count_sad = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if w < 50:
            continue

        # face = np.clip(gray[y:y+h, x:x+w]/0.25, 0, 255).astype(dtype=np.uint8)
        face = np.clip(cv2.equalizeHist(gray[y:y+h, x:x+w]) * 1 + 0, 0, 255).astype(np.uint8)
        # cv2.imwrite('face.png', face)
        face = cv2.resize(face, (224, 224)).reshape(224, 224, 1)
        #face = np.array(Image.fromarray(/face))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        # print(predictions)
        # max_index = np.argmax(predictions[0])
        # emotion = emotions[max_index]
        score = predictions[0][0]
        if score > 0.70:
            emotion = 'happy'
            count_happy += 1
            count_neutral = 0
            count_sad = 0
        elif score > 0.5:
            emotion = 'neutral'
            count_happy = 0
            count_neutral += 1
            count_sad = 0
        else:
            emotion = 'sad'
            count_happy = 0
            count_neutral = 0
            count_sad += 1
        
        print(emotion)

        if (count_happy > 3) | (count_neutral > 3) | (count_sad > 3):
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Real-time Facial Expression Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
