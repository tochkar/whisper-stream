import socket
import multiprocessing as mp
import numpy as np
import os
import requests
from dotenv import load_dotenv

HOST = "0.0.0.0"
PORT = 8082
BUFFER_SIZE = 48000 * 2
OVERLAP_SIZE = 48000 // 2
SAMPLE_RATE = 16000

# Настрой сервисов
MODEL_SERVER_URL = "http://localhost:7000"

def extract_ner_remotely(text):
    resp = requests.post(f"{MODEL_SERVER_URL}/ner/", data={"text": text})
    if resp.ok:
        return resp.json()
    else:
        return {}

def transcribe_remotely(audio_bytes):
    resp = requests.post(
        f"{MODEL_SERVER_URL}/asr/",
        files={"audio": ("audio.raw", audio_bytes)}
    )
    if resp.ok:
        return resp.json()["text"]
    else:
        return ""

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
                    audio_data = np.frombuffer(block, np.int16)
                    
                    # 1. Транскрибация через REST API
                    audio_bytes = audio_data.tobytes()
                    transcription = transcribe_remotely(audio_bytes)
                    print(f"[{addr}] Транскрипция:", transcription)

                    # 2. NER через REST API
                    ner_res = extract_ner_remotely(transcription)
                    print(f"[{addr}] NER:", ner_res)

                    # 3. Дальше ваша канонизация, спец.логика, вывод, etc.
                    # (сюда вставь postprocessing — см. свои фрагменты выше)

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
