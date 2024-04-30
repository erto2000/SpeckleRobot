import socket


def startServer():
    host = '0.0.0.0'
    port = 12345

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    print(f"Server listening on {host}:{port}")
    conn, addr = s.accept()
    if conn:
        print(f"Connected by {addr}")
        return s, conn, addr


def receiveShot(connection, filename='received_image.jpg'):
    connection.sendall('take_shot'.encode())

    print("Receiving image...")
    data_buffer = b''
    while True:
        data = connection.recv(1024)
        if data.endswith(b'EOF'):
            data_buffer += data[:-3]  # Append data excluding the last 3 bytes ('EOF')
            break
        data_buffer += data
    with open(filename, 'wb') as f:
        f.write(data_buffer)
    print(f"Image received and saved as '{filename}'")


if __name__ == "__main__":
    s, conn, addr = startServer()
    receiveShot(conn, 'heeeyyo.jpg')
