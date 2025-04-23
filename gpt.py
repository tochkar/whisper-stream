import socket
import multiprocessing as mp
import numpy as np
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv
import torch

from openai import OpenAI

##################### Параметры ########################
HOST = "0.0.0.0"
PORT = 8084
BUFFER_SIZE = 32000 * 2    # 2 сек для 16kHz 16bit mono
OVERLAP_SIZE = 32000 // 2  # 0.5 сек перекрытия
SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
########################################################

SYSTEM_PROMPT = (
    "Ты будешь получать куски разговора. "
    "ЕСЛИ ЕСТЬ в куске адрес посадки и адрес назначения, то исправь орфографические ошибки в соответствии с названиями улиц Минска. "
    "Твоя задача: из этих кусков диалога извлечь адрес посадки пассажира и адрес назначения. Не отвечай лишнего, просто ответь в виде:\n"
    "Подача: <адрес1>\n"
    "Назначение: <адрес2>\n"
    "Если в куске нет полной информации - напиши 'нет информации'."
)

def process_text_with_openai(text, openai_key):
    client = OpenAI(api_key=openai_key)
    # Для gpt-3.5-turbo, можно использовать другую модель
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}],
            temperature=0.1,
            max_tokens=70,
            top_p=1.0,
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        return f"OpenAI API error: {e}"

# Функция обработки аудиопотока в процессе
def handle_client_proc(conn, addr, huggingface_token, openai_key):
    print(f"[{addr}] Новый процесс обработчика клиента")
    try:
        # Загрузка модели только внутри процесса!
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            use_auth_token=huggingface_token
        ).to(device)
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=huggingface_token)
        tokenizer = WhisperTokenizer.from_pretrained(
            MODEL_NAME, language=LANGUAGE, task="transcribe", use_auth_token=huggingface_token
        )

        def transcribe_audio(audio_data):
            # Нормализация
            if np.max(np.abs(audio_data)) != 0:
                audio_data = (audio_data / np.max(np.abs(audio_data))).astype(np.float32)
            else:
                audio_data = audio_data.astype(np.float32)
            input_features = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device).to(torch_dtype)
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
            print(f"[{addr}] Транскрипция: {transcription}")

            # ОТПРАВКА В OPENAI GPT
            extracted_info = process_text_with_openai(transcription, openai_key)
            print(f"[{addr}] OpenAI результат:\n{extracted_info}")
            # Можете отправлять назад клиенту или дальше по цепочке
            # conn.sendall(extracted_info.encode('utf-8'))

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
                    transcribe_audio(audio_data)
                    # Оставляем перекрытие для плавности
                    audio_buffer = audio_buffer[BUFFER_SIZE - OVERLAP_SIZE:]
    except Exception as e:
        print(f"[{addr}] Ошибка: {e}")
    finally:
        conn.close()
        print(f"[{addr}] Завершение сессии")

def main():
    load_dotenv()
    huggingface_token = os.environ['HF_TOKEN']
    openai_key = os.environ['OPENAI_API_KEY']
    print(f"Запуск сервера на {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(32)
        while True:
            conn, addr = server_sock.accept()
            # Передаём коннект, адрес и токены в новый процесс обработчика
            proc = mp.Process(target=handle_client_proc, args=(conn, addr, huggingface_token, openai_key), daemon=True)
            proc.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')  # для совместимости на Mac/Win (spawn лучше для PyTorch/transformers)
    main()
