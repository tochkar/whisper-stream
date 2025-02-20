import os
import torch
import numpy as np
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from openai import OpenAI
from dotenv import load_dotenv

# Загрузить переменные из .env
load_dotenv()
huggingface_token = os.environ["HF_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]

# Настройки модели
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"

# Загрузка модели и процессора
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, torch_dtype=torch_dtype, low_cpu_mem_usage=True, token=huggingface_token
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=huggingface_token)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language="ru", task="transcribe", token=huggingface_token
)

def transcribe_audio(audio_file):
    process = (
        ffmpeg.input(audio_file)
        .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar="16000")
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
        no_repeat_ngram_size=2,
    )
    transcription = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(f"Transcription: {transcription}")
    return transcription

def extract_addresses(text):
    # Инициализация клиента OpenAI
    openai.api_key = openai_api_key

    # Формируем запрос
    prompt = (
        f"Текст: {text}\n"
        f"Определите адрес начала и конца поездки в формате JSON:\n"
        f"{{\"startAddress\": \"начальный адрес\", \"finalAddress\": \"конечный адрес\"}}"
    )

    # Запрос к GPT
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.3,
    )

    address_info = response.choices[0].text.strip()
    print(f"Extracted Addresses: {address_info}")
    return address_info

def main():
    audio_file = "speech.mp3"  # файл аудио для транскрибции

    if os.path.isfile(audio_file):
        transcription = transcribe_audio(audio_file)
        address_info = extract_addresses(transcription)
        # Здесь вы можете сохранить или использовать address_info по вашему усмотрению
    else:
        print(f"Файл {audio_file} не найден.")

if __name__ == "__main__":
    main()
