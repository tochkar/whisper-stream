import socket

TCP_IP = '0.0.0.0'  # Можно оставить так, чтобы слушать на всех интерфейсах
TCP_PORT = 8082    # <-- Укажите порт, на который FreeSWITCH отправляет поток

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen(1)

print(f"Listening TCP stream on {TCP_IP}:{TCP_PORT}")

conn, addr = server_socket.accept()
print(f"Connection from: {addr}")

try:
    while True:
        data = conn.recv(4096)
        if not data:
            break
        print(f"Received {len(data)} bytes")
        # Можно сохранить в файл для анализа:
        # with open('tcp_audio_stream.raw', 'ab') as f:
        #     f.write(data)

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    conn.close()
    server_socket.close()
