import asyncio
import websockets
import numpy as np
import requests
import time
import paho.mqtt.client as mqtt
import os
import soundfile as sf
import scipy.signal
from io import BytesIO

HOST = "0.0.0.0"
PORT = 8082
SAMPLE_RATE_IN = 16000  # легко менять тут
SAMPLE_RATE_OUT = 16000
CHUNK_SECONDS = 4
OVERLAP_SECONDS = 1

MODEL_SERVICE_URL = "http://localhost:7000/address/"

MQTT_BROKER = "socket.taxi135.by"
MQTT_PORT = 443
MQTT_USERNAME = "admin"
MQTT_PASSWORD = "6BHK2pGn3d"
MQTT_CLIENT_ID = "ai_service"
MQTT_TOPIC = "ai/recognize/taxi"

BYTES_PER_SAMPLE = 2
samples_per_chunk = SAMPLE_RATE_IN * CHUNK_SECONDS
samples_overlap = SAMPLE_RATE_IN * OVERLAP_SECONDS
bytes_per_chunk = samples_per_chunk * BYTES_PER_SAMPLE
bytes_overlap = samples_overlap * BYTES_PER_SAMPLE


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate):
    buf = BytesIO()
    sf.write(buf, pcm, sample_rate, subtype='PCM_16', format='WAV')
    buf.seek(0)
    return buf.read()


def fade_edges(wave, fade_len=128):
    window = np.ones_like(wave, dtype=np.float32)
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)
    return (wave.astype(np.float32) * window).astype(np.int16)


def lowpass_filter(wave, cutoff=3400, fs=16000, order=4):
    b, a = scipy.signal.butter(order, cutoff / (0.5 * fs), btype='low')
    return scipy.signal.lfilter(b, a, wave).astype(np.int16)


def send_to_mqtt(result):
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, transport="websockets")
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
    client.loop_start()
    for _ in range(30):
        if client.is_connected():
            break
        time.sleep(0.1)
    msg = str(result) if result is not None else ''
    print(f"[MQTT] Публикуем: {msg}")
    client.publish(MQTT_TOPIC, msg, retain=False)
    client.loop_stop()
    client.disconnect()


async def handle_stream(websocket):
    print(f"[WS] connect: {websocket.remote_address}")
    audio_buffer = bytearray()
    try:
        while True:
            data = await websocket.recv()
            if not isinstance(data, (bytes, bytearray)):
                continue
            audio_buffer += data

            # работа по чанкам
            while len(audio_buffer) >= bytes_per_chunk:
                chunk_bytes = audio_buffer[:bytes_per_chunk]
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16)

                # DSP: lowpass, fade, нормализация
                chunk = lowpass_filter(chunk, fs=SAMPLE_RATE_IN)
                chunk = fade_edges(chunk)
                max_val = np.max(np.abs(chunk))
                if max_val > 0:
                    chunk = (chunk.astype(np.float32) * (0.9 * 32767.0 / max_val)).astype(np.int16)
                # ресемплинг 8k->16k
                chunk_16k = scipy.signal.resample_poly(chunk, SAMPLE_RATE_OUT, SAMPLE_RATE_IN).astype(np.int16)
                wav_bytes = pcm_to_wav_bytes(chunk_16k, SAMPLE_RATE_OUT)

                # Отправляем в Speech-to-text сервис
                try:
                    resp = requests.post(
                        MODEL_SERVICE_URL,
                        files={"audio": ("audio.wav", wav_bytes)}
                    )
                    if resp.ok:
                        result_json = resp.json()
                        result = result_json.get('result')
                        print(f"[{websocket.remote_address}] Ответ сервиса: {result_json}")
                        # --- Отправка в MQTT ---
                        if result and str(result).strip():
                            send_to_mqtt(result)
                    else:
                        print(f"[{websocket.remote_address}] Ошибка, сервис дал не-200: {resp.status_code}")
                except Exception as e:
                    print(f"[{websocket.remote_address}] ошибка подключения к сервису: {e}")

                audio_buffer = audio_buffer[bytes_per_chunk - bytes_overlap:]

    except websockets.ConnectionClosed:
        print(f"[WS] disconnect: {websocket.remote_address}")
    except Exception as e:
        print(f"[WS] ошибка: {e}")


async def main():
    print(f"[WS SERVER] ws://{HOST}:{PORT}")
    async with websockets.serve(handle_stream, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
