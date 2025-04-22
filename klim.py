import socket
import torch
import numpy as np
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv

# Загрузить .env
load_dotenv()
huggingface_token = os.environ['HF_TOKEN']

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"

# Загрузка модели и процессоров
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_auth_token=huggingface_token
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=huggingface_token)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="ru", task="transcribe", use_auth_token=huggingface_token)

def transcribe_audio(audio_data, sample_rate=16000):
    if np.max(np.abs(audio_data)) != 0:
        audio_data = (audio_data / np.max(np.abs(audio_data))).astype(np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device).to(torch.float16)
    predicted_ids = model.generate(
        input_features,
        max_length=1024,
        do_sample=True,
        temperature=1.5,
        top_k=100,
        top_p=0.15,
        no_repeat_ngram_size=2
    )
    transcription = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(f"Транскрипция: {transcription}")

HOST = "0.0.0.0"
PORT = 8084

BUFFER_SIZE = 32000 * 2    # 2 секунды при 16кГц 16bit mono
OVERLAP_SIZE = 32000 // 2  # 0.5 секунды перекрытия

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Жду подключения FreeSWITCH на {HOST}:{PORT} ...")
    conn, addr = srv.accept()
    print(f'Подключено: {addr}')
    with conn:
        audio_buffer = bytearray()
        while True:
            data = conn.recv(4096)
            if not data:
                break
            audio_buffer.extend(data)
            # Если накопили >= BUFFER_SIZE, то отправляем буфер в Whisper
            while len(audio_buffer) >= BUFFER_SIZE:
                block = audio_buffer[:BUFFER_SIZE]
                audio_data = np.frombuffer(block, np.int16)
                transcribe_audio(audio_data, sample_rate=16000)
                # Оставляем в буфере OVERLAP_SIZE байт для перекрытия
                audio_buffer = audio_buffer[BUFFER_SIZE - OVERLAP_SIZE:]
