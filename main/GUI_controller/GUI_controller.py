import socket
import os
from queue import Empty

# The address of the UNIX domain socket
server_address = '/home/pi/uds_socket'

class GUIController:
    def __init__(self, communication_queues):
        self._motion_queue = communication_queues['motion_queue']
        self._GUI_queue = communication_queues['GUI_queue']

    def GUI_comm(self):
        # Make sure the socket does not already exist
        try:
            os.unlink(server_address)
        except OSError:
            if os.path.exists(server_address):
                raise

        UDS_sever = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) # Create a Unix Domain Socket
        UDS_sever.bind(server_address) # Bind the socket to the address
        UDS_sever.listen(1) # Listen for incoming connections
        connection_socket, client_address = UDS_sever.accept()

        while True:
            try:
                while True:
                    ## receive data from the client
                    data = connection_socket.recv(1024).decode()
                    if data:
                        self._motion_queue.put(data, timeout=60)
                        # print(data)
                        # print('received {!r}'.format(data))
                        # connection_socket.sendall(b'Hello from Python')
                    else:
                        # print('no more data, socket closing')
                        break

                    ## Send data from GUI_queue to the client
                    try:
                        GUI_data = self._GUI_queue.get_nowait()
                        connection_socket.sendall(GUI_data.encode())
                    except Empty:
                        pass

            finally:
                # Clean up the connection
                connection_socket.close()
