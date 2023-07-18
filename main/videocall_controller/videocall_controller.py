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
stop_threads = False

CHUNK = 1024
FORMAT = pyaudio.paInt16    # 格式
CHANNELS = 2    # 输入/输出通道数
RATE = 44100    # 音频数据的采样频率
RECORD_SECONDS = 0.5    # 记录秒

def send_video(video_connection_socket, audio_connection_socket):
    global stop_threads

    # Video #
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 24)

    # Audio #
    audio = pyaudio.PyAudio()
    audio_stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer = CHUNK)

    while not stop_threads:
        try:
            # Video #
            ret, video_frame = capture.read()
            if not ret:
                print('unable to read the video...')
                break
            video_frame = cv2.resize(video_frame, (320, 240))
            video_data = video_frame.tobytes()
            video_connection_socket.sendall(struct.pack("L", len(video_data)) + video_data)

            # Audio #
            audio_frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                audio_data = audio_stream.read(CHUNK)
                audio_frames.append(audio_data)
            client_audio_senddata = pickle.dumps(audio_frames)
            audio_connection_socket.sendall(struct.pack("L", len(client_audio_senddata)) + client_audio_senddata)

        except socket.error as e:
            print("Socket error:", e)
            break
    # End of send_video
    capture.release()
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()

    stop_threads = True

def recv_video(video_connection_socket, audio_connection_socket):
    global stop_threads
    payload_size = struct.calcsize("L")

    # Video #
    video_data = b'' # empty btytes
    
    # Audio #
    audio_data = "".encode("utf-8")
    audio = pyaudio.PyAudio()
    audio_stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                output=True,
                                frames_per_buffer = CHUNK
                                )
    
    while not stop_threads:
        try:
            # Video #
            # process the data that received
            while len(video_data) < payload_size:
                video_data += video_connection_socket.recv(81920)
            video_packed_size = video_data[:payload_size]
            video_data = video_data[payload_size:]
            video_data_size = struct.unpack("L", video_packed_size)[0]
            while len(video_data) < video_data_size:
                video_data += video_connection_socket.recv(81920)
            video_frame_data = video_data[:video_data_size]
            video_data = video_data[video_data_size:]
            # show the video
            video_frame = np.frombuffer(video_frame_data, dtype=np.uint8).reshape(240, 320, 3)
            cv2.imshow('Received Video', video_frame)

            # Audio #
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

            # stop streaming if type q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True
                break
        except socket.error as e:
            print("Socket error:", e)
            break

    cv2.destroyAllWindows()
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()

    stop_threads = True

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
  print('messages are yummy')
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

if __name__ == '__main__':
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    
    video_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # client socket declaration: ipv4, TCP
    audio_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_server_socket.bind((sever_ip, sever_port))
    audio_server_socket.bind((sever_ip, sever_port+1))
    video_server_socket.listen(1)
    audio_server_socket.listen(1)

    video_connection_socket, video_client_address = video_server_socket.accept()
    audio_connection_socket, audio_client_address = audio_server_socket.accept()

    send_video_thread = threading.Thread(target=send_video, args=(video_connection_socket, audio_connection_socket,))
    recv_video_thread = threading.Thread(target=recv_video, args=(video_connection_socket, audio_connection_socket,))

    send_video_thread.start()
    recv_video_thread.start()

    send_video_thread.join()
    recv_video_thread.join()

    # connection_socket.shutdown(socket.SHUT_RDWR)
    video_connection_socket.close()
    video_server_socket.close()
    audio_connection_socket.close()
    audio_server_socket.close()
