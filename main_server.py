import socket
import multiprocessing as mp
import numpy as np
import requests
from dotenv import load_dotenv

HOST = "0.0.0.0"
PORT = 8082
BUFFER_SIZE = 48000 * 2
OVERLAP_SIZE = 48000 // 2

MODEL_SERVICE_URL = "http://localhost:7000/address/"

def handle_client_proc(conn, addr):
    try:
        audio_buffer = bytearray()
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                audio_buffer.extend(data)
                while len(audio_buffer) >= BUFFER_SIZE:
                    block = audio_buffer[:BUFFER_SIZE]
                    # 1. Отправляем аудиоблок в модельный сервис (bin)
                    result = None
                    try:
                        resp = requests.post(
                            MODEL_SERVICE_URL,
                            files={"audio": ("audio.raw", block)}
                        )
                        if resp.ok:
                            result = resp.json()
                    except Exception as e:
                        print(f"[{addr}] ошибка подключения к сервису: {e}")
                    print(f"[{addr}] Ответ сервиса: {result}")
                    audio_buffer = audio_buffer[BUFFER_SIZE - OVERLAP_SIZE:]
    except Exception as e:
        print(f"[{addr}] Ошибка: {e}")
    finally:
        conn.close()
        print(f"[{addr}] Завершение сессии")

def main():
    load_dotenv()
    print(f"[SERVER] Слушаем на {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(32)
        while True:
            conn, addr = server_sock.accept()
            proc = mp.Process(target=handle_client_proc, args=(conn, addr), daemon=True)
            proc.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
