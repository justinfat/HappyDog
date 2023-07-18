import cv2
import numpy as np

videoHeight = 240
videoWidth = 320

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 10)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read()
    if not ret:
        print('unable to read the video...')
        break

    frame = cv2.resize(frame, (videoWidth, videoHeight))
    frame = np.frombuffer(frame, dtype=np.uint8).reshape(videoHeight, videoWidth, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        faceCenter = (int(x+w/2), int(y+h/2))
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # face region
        cv2.putText(frame, 'x: %s, y: %s'%faceCenter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # face center coordinates
        cv2.line(frame, faceCenter, (int(videoWidth/2), int(videoHeight/2)), (0, 0, 255), 2) # face center to frame center line
        cv2.rectangle(frame, (int(videoWidth*0.25), int(videoHeight*0.25)), (int(videoWidth*0.75), int(videoHeight*0.75)), (255, 0, 0), 2)
        # face = gray[y:y+h, x:x+w] # choose the face region from gray

        if faceCenter[0] > videoWidth*0.75:
            print('Too right...')
        elif faceCenter[0] < videoWidth*0.25:
            print('Too left...')

        if faceCenter[1] > videoHeight*0.75:
            print('Too low...')
        elif faceCenter[1] < videoHeight*0.25:
            print('Too high...')


        
    cv2.imshow('Captured Video', frame)

    # stop video calling if type q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()