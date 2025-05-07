import asyncio
import websockets
import numpy as np
import soundfile as sf
from io import BytesIO
from datetime import datetime
import boto3
import os
import scipy.signal

SAMPLE_RATE_OUT = 16000     # Whisper-ready
CHANNELS = 2
BYTES_PER_SAMPLE = 2
CHUNK_SECONDS = 4
OVERLAP_SECONDS = 1

S3_BUCKET = "taxi-ai-rec"
S3_REGION = "eu-west-1"

FRAME_BYTES = CHANNELS * BYTES_PER_SAMPLE

aws_access_key_id = "ASIAWPAAGPMMIGYDZCMS"
aws_secret_access_key = "FQESR+JXrLUatKAO2QdVw3FGxcsmTw69Ybouq6pN"
aws_session_token = "IQoJb3JpZ2luX2VjELX//////////wEaCWV1LXdlc3QtMSJGMEQCIBilUc3zaO4wSXPBk8/ArrGVuOFMWBqV7PHINYxvjBVZAiBEVC63gmYOhCRz/4C3GSi7cEkut6T1ULBz7ghN>

s3 = boto3.client(
    's3',
    region_name=S3_REGION,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token  # <--- Важно если юзаешь временные креды
)


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate=16000):
    buf = BytesIO()
    sf.write(buf, pcm, sample_rate, subtype='PCM_16', format='WAV')
    buf.seek(0)
    return buf.read()

def smart_chunk(array_bytes, length_samples, overlap_samples, verbose=None):
    """Yield chunks of given size with overlap from byte buffer"""
    offset = 0
    chunk_bytes = length_samples * BYTES_PER_SAMPLE
    overlap_bytes = overlap_samples * BYTES_PER_SAMPLE
    while len(array_bytes) - offset >= chunk_bytes:
        yield array_bytes[offset:offset+chunk_bytes]
        offset += chunk_bytes - overlap_bytes

async def handle_stream(websocket):
    print(f"[WS] connect: {websocket.remote_address}")
    interleaved_buffer = bytearray()
    operator_buffer = bytearray()
    operator_samples_per_sec = None

    chunk_counter = 0
    try:
        while True:
            data = await websocket.recv()
            if not isinstance(data, (bytes, bytearray)):
                continue
            interleaved_buffer += data

            # Всегда обрабатываем только целые фреймы!
            nframes = len(interleaved_buffer) // FRAME_BYTES
            usable_bytes = nframes * FRAME_BYTES
            if usable_bytes == 0:
                continue
                
            interleaved_buffer += data

            # Всегда обрабатываем только целые фреймы!
            nframes = len(interleaved_buffer) // FRAME_BYTES
            usable_bytes = nframes * FRAME_BYTES
            if usable_bytes == 0:
                continue
            frames = np.frombuffer(interleaved_buffer[:usable_bytes], dtype='<i2').reshape(-1, CHANNELS)
            operator_buffer += frames[:, 0].tobytes()
            interleaved_buffer = interleaved_buffer[usable_bytes:]

            # Для автоопределения истинного sample rate регистрируем размер первого чанка
            while True:
                # Пробуем chunk'овать по любому sample rate, начиная с 16k (Whisper), если не хватает - пробуем 8k
                # Это важно для реальной адаптации к разному входу!
                if operator_samples_per_sec is None:
                    guess_rates = [16000, 8000, 48000, 44100]
                else:
                    guess_rates = [operator_samples_per_sec]
                found = False
                for srate in guess_rates:
                    samples_needed = srate * CHUNK_SECONDS
                    chunk_bytes_needed = samples_needed * BYTES_PER_SAMPLE
                    overlap_samples = srate * OVERLAP_SECONDS
                    overlap_bytes = overlap_samples * BYTES_PER_SAMPLE
                    if len(operator_buffer) >= chunk_bytes_needed:
                        chunk_bytes = operator_buffer[:chunk_bytes_needed]
                        chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16)
                        duration = len(chunk_array) / srate
                        # Auto-fix/reject дурацкие значения
                        if 3.7 < duration < 4.3:  # Окно 4 сек (±15%)
                            operator_samples_per_sec = srate
                            found = True
                            # Ресемплим если надо в 16к для Whisper
                            if srate != SAMPLE_RATE_OUT:
                                chunk_16k = scipy.signal.resample_poly(chunk_array, SAMPLE_RATE_OUT, srate).astype(np.int16)
                                info = f"{srate}to16k"
                            else:
                                chunk_16k = chunk_array
                                info = f"{srate}k"
                            # Нормализация уровня
                            if np.max(np.abs(chunk_16k)) > 0:
                                chunk_16k = (chunk_16k.astype(np.float32) * (32767.0 / np.max(np.abs(chunk_16k)))).astype(np.int16)
                            wav_bytes = pcm_to_wav_bytes(chunk_16k, SAMPLE_RATE_OUT)
                            now_str = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
                            s3key = f"audio-chunks/operator_{now_str}_{chunk_counter:03d}_{info}.wav"
                            try:
                                s3.upload_fileobj(BytesIO(wav_bytes), S3_BUCKET, s3key)
                                print(f"[WS] Uploaded to s3://{S3_BUCKET}/{s3key} ({info})")
                            except Exception as e:
                                print("[WS] S3 upload error:", e)
                            chunk_counter += 1
                            operator_buffer = operator_buffer[chunk_bytes_needed - overlap_bytes:]
                            break
                if not found:
                    break

    except websockets.ConnectionClosed:
        print(f"[WS] disconnect: {websocket.remote_address}")
    except Exception as e:
        print(f"[WS] ошибка:", e)

async def main():
    print(f"[WS SERVER] ws://0.0.0.0:8082")
    async with websockets.serve(handle_stream, "0.0.0.0", 8082):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

