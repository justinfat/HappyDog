import cv2
import socket
import numpy as np
import struct
import threading
import pyaudio
import pickle
import ctypes

sever_ip = '0.0.0.0'
sever_port = 8485
videoHeight = 240
videoWidth = 320
video_data_ready = False

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

class InputController:
    def __init__(self, communication_queues):
        self._motion_queue = communication_queues['motion_queue']
        self._socket_queue = communication_queues['socket_queue']
        # Hide the warning from pyaudio (ALSA)
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
    
    def run(self, communication_queues):
        controller = InputController(communication_queues)
        send_video_thread = threading.Thread(target=controller.send_video, args=())
        face_track_thread = threading.Thread(target=controller.face_track, args=())
        send_video_thread.start()
        face_track_thread.start()
        send_video_thread.join()
        face_track_thread.join()

    def send_video(self):
        global video_data_ready
        # socket setting
        video_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # client socket declaration: ipv4, TCP
        audio_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_server_socket.bind((sever_ip, sever_port))
        audio_server_socket.bind((sever_ip, sever_port+1))
        video_server_socket.listen(1)
        audio_server_socket.listen(1)

        video_connection_socket, client_address = video_server_socket.accept()
        audio_connection_socket, client_address = audio_server_socket.accept()
        self._socket_queue.put(video_connection_socket)
        self._socket_queue.put(audio_connection_socket)

        # Audio #
        audio = pyaudio.PyAudio()
        audio_stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer = CHUNK)

        while True:
            try:
                # Video
                if video_data_ready is True:
                    video_connection_socket.sendall(struct.pack("L", len(self.video_data)) + self.video_data)

                #Audio
                audio_frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    audio_raw = audio_stream.read(CHUNK)
                    audio_frames.append(audio_raw)
                audio_data = pickle.dumps(audio_frames)
                audio_connection_socket.sendall(struct.pack("L", len(audio_data)) + audio_data)

            except socket.error as e:
                print("Send video socket error:", e)
                # connection_socket.shutdown(socket.SHUT_RDWR)
                break
        
        video_connection_socket.close()
        audio_connection_socket.close()
        audio_stream.stop_stream()
        audio_stream.close()
        audio.terminate()

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

# if __name__ == '__main__':
#     InputController().send_video()