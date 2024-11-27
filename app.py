import torch
import numpy as np
import os
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv

# Загрузить переменные из .env
load_dotenv()

# Получить токен из переменной окружения
huggingface_token = os.environ['HUGGINGFACE_TOKEN']

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
    input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    transcription = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(f"Transcription: {transcription}")

# URL вашего аудиопотока
stream_url = "http://your-audio-stream-url"

# Использование ffmpeg для декодирования аудиопотока
process = (
    ffmpeg
    .input(stream_url, threads=0)
    .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
    .run_async(pipe_stdout=True, pipe_stderr=True, bufsize=10**8)
)

try:
    while True:
        in_bytes = process.stdout.read(4096)  # Читаем 4096 байт из потока
        if not in_bytes:
            break
        audio_data = np.frombuffer(in_bytes, np.int16)
        transcribe_audio(audio_data, 16000)
except KeyboardInterrupt:
    pass
finally:
    process.stdout.close()
    process.wait()
