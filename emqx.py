import socket
import multiprocessing as mp
import numpy as np
import requests
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import json
import time

HOST = "0.0.0.0"
PORT = 8082
BUFFER_SIZE = 48000 * 2
OVERLAP_SIZE = 48000 // 2
MODEL_SERVICE_URL = "http://localhost:7000/address/"

# --- MQTT параметры ---
MQTT_BROKER = "socket.taxi135.by"
MQTT_PORT = 443
MQTT_USERNAME = "admin"
MQTT_PASSWORD = "6BHK2pGn3d"
MQTT_CLIENT_ID = "ai_service"
MQTT_TOPIC = f"ai/recognize/taxi"

def send_to_mqtt(result):
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, transport="websockets")
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
    client.loop_start()
    # Ждём подключения (очень быстро в пределах 0.5-1 сек)
    for _ in range(30):
        if client.is_connected():
            break
        time.sleep(0.1)
    msg = str(result) if result is not None else ''
    print(f"[MQTT] Публикуем: {msg}")
    client.publish(MQTT_TOPIC, msg, retain=False)
    # Отключение
    client.loop_stop()
    client.disconnect()

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
                    # Отправляем аудиоблок в модельный сервис (bin)
                    result = None
                    try:
                        resp = requests.post(
                            MODEL_SERVICE_URL,
                            files={"audio": ("audio.raw", block)}
                        )
                        if resp.ok:
                            result_json = resp.json()
                            result = result_json.get('result')
                            print(f"[{addr}] Ответ сервиса: {result_json}")
                            # --- Отправка в MQTT ---
                            if result and result.strip():  # не посылать пустые
                                send_to_mqtt(result)
                        else:
                            print(f"[{addr}] Ошибка, сервис дал не-200: {resp.status_code}")

                    except Exception as e:
                        print(f"[{addr}] ошибка подключения к сервису: {e}")

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
