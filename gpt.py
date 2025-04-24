import socket
import multiprocessing as mp
import numpy as np
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv
import torch

# Natasha and pymorphy2
from natasha import AddrExtractor
from pymorphy2 import MorphAnalyzer

##################### Параметры ########################
HOST = "0.0.0.0"
PORT = 8082
BUFFER_SIZE = 32000 * 2    # 2 сек для 16kHz 16bit mono
OVERLAP_SIZE = 32000 // 2  # 0.5 сек перекрытия
SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
########################################################

# ---- Морфанализатор инициализируем один раз глобально ----
print("Инициализация морфологического анализатора Natasha ...")
morph = MorphAnalyzer()
print("Морфанализатор создан.")

def handle_client_proc(conn, addr, huggingface_token):
    print(f"[{addr}] Новый процесс обработчика клиента")
    # Используем глобальный morph
    addr_extractor = AddrExtractor(morph)

    try:
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

            # ---- Используем Natasha для поиска адресов ----
            matches = addr_extractor(transcription)
            for match in matches:
                print(f"[{addr}] НАЙДЕН АДРЕС НАТАШЕЙ: {match.fact}")
                print(f"[{addr}] В ТЕКСТЕ: {transcription[match.span[0]:match.span[1]]}")

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
                    audio_buffer = audio_buffer[BUFFER_SIZE - OVERLAP_SIZE:]
    except Exception as e:
        print(f"[{addr}] Ошибка: {e}")
    finally:
        conn.close()
        print(f"[{addr}] Завершение сессии")

def main():
    load_dotenv()
    huggingface_token = os.environ['HF_TOKEN']
    print(f"Запуск сервера на {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(32)
        while True:
            conn, addr = server_sock.accept()
            proc = mp.Process(target=handle_client_proc, args=(conn, addr, huggingface_token), daemon=True)
            proc.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
