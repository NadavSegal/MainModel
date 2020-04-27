import socket
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

sock = socket.socket()
data= np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()
sock = socket.socket()
data= np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()

sock = socket.socket()
data= np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()
sock = socket.socket()
data= np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()
sock = socket.socket()
data= np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()
sock = socket.socket()
data= np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()