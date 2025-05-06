import socket

HOST = '0.0.0.0'
PORT = 8082

print(f"[SERVER] Ждет подключения на {HOST}:{PORT}...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(5)
    while True:
        conn, addr = server_sock.accept()
        print(f"[CONNECT] От {addr}")
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    print(f"[DISCONNECT] {addr}")
                    break
                print(f"[{addr}] Получено {len(data)} байт")
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            conn.close()
