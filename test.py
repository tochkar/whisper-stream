import os
import torch
import numpy as np
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv

# Загрузить переменные из .env
load_dotenv()
huggingface_token = os.environ['HF_TOKEN']

# Настройки модели
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"

# Загрузка модели и процессора
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    token=huggingface_token
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=huggingface_token)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="ru", task="transcribe", token=huggingface_token)

# Функция для обработки аудиоданных
def transcribe_audio(audio_file):
    process = (
        ffmpeg
        .input(audio_file)
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    audio_buffer = process.stdout.read()
    audio_data = np.frombuffer(audio_buffer, np.int16)

    audio_data = (audio_data / np.max(np.abs(audio_data), axis=0)).astype(np.float32)
    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device).to(torch.float16)
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
    return transcription

# Основная функция
def main():
    # Замените 'speech.mp3' на путь к вашему аудиофайлу
    audio_file = 'audio.mp3'
    if os.path.isfile(audio_file):
        transcribe_audio(audio_file)
    else:
        print(f"Файл {audio_file} не найден.")

if __name__ == "__main__":
    main()
