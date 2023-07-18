import cv2
import socket
import numpy as np
import struct
from tflite_runtime.interpreter import Interpreter
from flask import Flask, Response
from flask_cors import CORS
import threading
import pyaudio
import pickle
import ctypes

videoHeight = 240
videoWidth = 320
video_data_ready = False

global_buffer = None
app = Flask(__name__)
CORS(app)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 0.5

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
#   print('messages are yummy')
    return
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

def generate_frames():
    global global_buffer

    while True:
        if global_buffer is not None:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + global_buffer + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

video_connection_socket = None
audio_connection_socket = None
video_frame = None

class InteractController:
    def __init__(self, communication_queues):
        self._motion_queue = communication_queues['motion_queue']
        self._socket_queue = communication_queues['socket_queue']
        # Hide the warning from pyaudio (ALSA)
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)

    def run(self, communication_queues):
        global video_connection_socket
        global audio_connection_socket 

        controller = InteractController(communication_queues)
        video_connection_socket = self._socket_queue.get(block=True)
        audio_connection_socket = self._socket_queue.get(block=True)
        face_track_thread = threading.Thread(target=controller.face_track, args=())
        emotion_recognize_thread = threading.Thread(target=controller.emotion_recognize, args=())
        face_track_thread.start()
        emotion_recognize_thread.start()
        app.run(host='0.0.0.0', port=8586)
        face_track_thread.join()
        emotion_recognize_thread.join()

    def face_track(self):
        global video_data_ready
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FPS, 10)

        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = capture.read()
            if not ret:
                print('unable to read the video...')
                break

            frame = cv2.resize(frame, (320, 240))
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
                    self._motion_queue.put('TooRight', timeout=60)
                    # print('Too right...')
                elif faceCenter[0] < videoWidth*0.25:
                    self._motion_queue.put('TooLeft', timeout=60)
                    # print('Too left...')

                if faceCenter[1] > videoHeight*0.75:
                    self._motion_queue.put('TooLow', timeout=60)
                    # print('Too low...')
                elif faceCenter[1] < videoHeight*0.25:
                    self._motion_queue.put('TooHigh', timeout=60)
                    # print('Too high...')

            self.video_data = frame.tobytes() # for send_video()
            if video_data_ready is False:
                video_data_ready = True

        capture.release()

    def emotion_recognize(self):
        global video_frame
        # Load the TFLite model and allocate tensors.
        interpreter = Interpreter(model_path="/home/pi/SpotLink/main/output_controller/model_mobilenet_4class.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Define the emotion labels
        # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotions = ('happy', 'neutral', 'sad', 'surprise')

        count_happy = 0
        count_neutral = 0
        count_sad = 0

        while True:
            if video_frame is None or video_frame.size == 0:
                print("Error: video frame is empty")
            else:
                gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(video_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # if w < 50:
                    #     continue

                    # face = gray[y:y+h, x:x+w] # choose the face region from gray
                    face = np.clip(cv2.equalizeHist(gray[y:y+h, x:x+w]) * 1 + 0, 0, 255).astype(np.uint8)
                    face = cv2.resize(face, (224, 224)).reshape(224, 224, 1)
                    #face = np.array(Image.fromarray(face))
                    face = face.astype('float32') / 255.0
                    face = np.expand_dims(face, axis=0)

                    interpreter.set_tensor(input_details[0]['index'], face)
                    interpreter.invoke()

                    predictions = interpreter.get_tensor(output_details[0]['index'])
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

                    # self._motion_queue.put('happy', timeout=60)

                    cv2.putText(video_frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # if (count_happy > 3) | (count_neutral > 3) | (count_sad > 3):
                    #     cv2.putText(video_frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #     self._motion_queue.put(emotion, timeout=60)

# if __name__ == '__main__':
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # client socket declaration: ipv4, TCP
#     server_socket.bind((sever_ip, sever_port))
#     server_socket.listen(1)

#     connection_socket, client_address = server_socket.accept()

#     OutputController().recv_video(connection_socket)

#     # connection_socket.shutdown(socket.SHUT_RDWR)
#     connection_socket.close()
#     server_socket.close()