import os
import io
import time
import openai
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 4  # длина одного фрагмента в секундах

def record_audio_chunk(duration=BLOCK_DURATION):
    print(f"Говорите {duration} секунд...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    print("Запись завершена.")
    return audio

def numpy_audio_to_wav_bytes(audio, sample_rate=SAMPLE_RATE):
    buf = io.BytesIO()
    wav_write(buf, sample_rate, audio)
    buf.seek(0)
    return buf

def stt_whisper_memory(wav_buffer):
    print("Распознаю через Whisper OpenAI...")
    wav_buffer.seek(0)
    transcript = openai.Audio.transcribe("whisper-1", wav_buffer, language="ru")
    print("Распознанный текст:", transcript['text'])
    return transcript['text']

def extract_addresses(text):
    PROMPT = (
        """Ты ассистент для оператора службы такси в Минске. 
        В расшифровке диалога выдели, если есть, адрес посадки и адрес назначения (по стандарту Минска). 
        Исправь опечатки в названиях улиц ориентируясь на улицы города Минска. 
        Выдай результат в формате:
        Откуда: <адрес посадки>, Куда: <адрес назначения>
        Если адреса нет, напиши "Адреса не найдено".
        Текст: """
        + text
    )
    print("Извлекаю адреса через GPT...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # или "gpt-4"
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200,
        temperature=0.1,
    )
    result = response.choices[0].message["content"]
    print("Ответ ассистента:", result)
    return result

def main():
    print("Система готова. Говорите для такси...")
    i = 1
    while True:
        audio = record_audio_chunk()
        wav_buffer = numpy_audio_to_wav_bytes(audio)
        text = stt_whisper_memory(wav_buffer)
        if text.strip():
            extract_addresses(text)
        else:
            print("Речь не распознана.")
        print("-" * 40)
        i += 1
        time.sleep(0.2)  # короткая пауза для циклa, можно уменьшить
        # Для завершения нажмите Ctrl+C

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
