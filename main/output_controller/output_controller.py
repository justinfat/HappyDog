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

sever_ip = '0.0.0.0'
sever_port = 8485

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

class OutputController:
    def __init__(self, communication_queues):
        self._motion_queue = communication_queues['motion_queue']
        self._socket_queue = communication_queues['socket_queue']
        # Hide the warning from pyaudio (ALSA)
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)

    def run(self, communication_queues):
        global video_connection_socket
        global audio_connection_socket 

        controller = OutputController(communication_queues)
        video_connection_socket = self._socket_queue.get(block=True)
        audio_connection_socket = self._socket_queue.get(block=True)
        recv_video_thread = threading.Thread(target=controller.recv_video, args=())
        emotion_recognize_thread = threading.Thread(target=controller.emotion_recognize, args=())
        recv_video_thread.start()
        emotion_recognize_thread.start()
        app.run(host='0.0.0.0', port=8586)
        recv_video_thread.join()
        emotion_recognize_thread.join()

    def recv_video(self):
        global global_buffer
        global video_frame

        payload_size = struct.calcsize("L")

        # Video
        video_data = b'' # empty btytes

        # Audio
        audio_data = "".encode("utf-8")
        audio = pyaudio.PyAudio()
        audio_stream = audio.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    output=True,
                                    frames_per_buffer = CHUNK
                                    )
        
        

        while True:
            try:
                # Video
                while len(video_data) < payload_size:
                    video_data += video_connection_socket.recv(81920)
                video_packed_size = video_data[:payload_size]
                video_data = video_data[payload_size:]
                video_data_size = struct.unpack("L", video_packed_size)[0]
                while len(video_data) < video_data_size:
                    video_data += video_connection_socket.recv(81920)
                video_frame_data = video_data[:video_data_size]
                video_data = video_data[video_data_size:]
                video_frame = np.frombuffer(video_frame_data, dtype=np.uint8).reshape(240, 320, 3)
                # cv2.imshow('Received Video', video_frame)

                # Audio
                while len(audio_data) < payload_size:
                    audio_data += audio_connection_socket.recv(81920)
                audio_packed_size = audio_data[:payload_size]
                audio_data = audio_data[payload_size:]
                audio_data_size = struct.unpack("L", audio_packed_size)[0]
                while len(audio_data) < audio_data_size:
                    audio_data += audio_connection_socket.recv(81920)
                audio_frame_data = audio_data[:audio_data_size]
                audio_data = audio_data[audio_data_size:]
                audio_frames = pickle.loads(audio_frame_data)
                for audio_frame in audio_frames:
                    audio_stream.write(audio_frame, CHUNK)

                # Flask
                ret, jpeg = cv2.imencode('.jpg', video_frame)
                if not ret:
                    print('unable to encode the video...')
                    break
                global_buffer = jpeg.tobytes()

                # stop video calling if type q
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     video_connection_socket.shutdown(socket.SHUT_RDWR)
                #     video_connection_socket.close()
                #     break
                
            except socket.error as e:
                print("Receive video socket error:", e)
                # connection_socket.shutdown(socket.SHUT_RDWR)
                break

        cv2.destroyAllWindows()
        audio_stream.stop_stream()
        audio_stream.close()
        audio.terminate()

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