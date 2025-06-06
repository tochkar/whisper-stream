import torch
import numpy as np
import os
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv

# Загрузить переменные из .env
load_dotenv()

# Получить токен из переменной окружения
huggingface_token = os.environ['HF_TOKEN']

# Настройки устройства и модели
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"

# Загрузка модели
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_auth_token=huggingface_token
).to(device)

# Загрузка процессора
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=huggingface_token)

# Настройка токенизатора на русский язык
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="ru", task="transcribe", use_auth_token=huggingface_token)

# Функция для обработки аудиоданных
def transcribe_audio(audio_data, sample_rate):
    audio_data = (audio_data / np.max(np.abs(audio_data), axis=0)).astype(np.float32)
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
    print(f"Transcription: {transcription}")

# URL вашего аудиопотока
stream_url = "https://media.govoritmoskva.ru/radio/rufm.mp3"

# Использование ffmpeg для декодирования аудиопотока
process = (
    ffmpeg
    .input(stream_url, threads=0)
    .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
    .run_async(pipe_stdout=True, pipe_stderr=True)
)

# Настройки буфера и перекрытия
buffer_size = 16000 * 4  # 4 секунды аудио
overlap_size = 16000 * 1  # 1 секунда перекрытия
audio_buffer = bytearray()

try:
    while True:
        in_bytes = process.stdout.read(4096)
        if not in_bytes:
            break

        audio_buffer.extend(in_bytes)

        if len(audio_buffer) >= buffer_size:
            # Передача в модель полного сегмента
            audio_data = np.frombuffer(audio_buffer, np.int16)
            transcribe_audio(audio_data, 16000)
            
            # Удерживайте перекрытие для следующего сегмента
            audio_buffer = audio_buffer[-overlap_size:]
except KeyboardInterrupt:
    pass
finally:
    process.stdout.close()
    process.wait()
