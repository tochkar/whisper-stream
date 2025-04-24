import socket
import multiprocessing as mp
import numpy as np
import os

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from dotenv import load_dotenv
import torch

##################### Параметры ########################
HOST = "0.0.0.0"
PORT = 8084
BUFFER_SIZE = 32000 * 2    # 2 сек для 16kHz 16bit mono
OVERLAP_SIZE = 32000 // 2  # 0.5 сек перекрытия
SAMPLE_RATE = 16000
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
YANDEX_LLM = "yandex/YandexGPT-5-Lite-8B-instruct"
########################################################

SYSTEM_PROMPT = (
    "Ты ассистент такси Минска. Ты будешь получать фрагменты диалогов между пассажиром и водителем.\n"
    "В каждом фрагменте может быть адрес подачи (откуда забирать пассажира) и адрес назначения (куда ехать), а может не быть ни одного из них.\n"
    "Адреса могут быть произнесены с ошибками, сокращённо, а номер дома – словами. Иногда говорят только один адрес.\n"
    "Твоя задача:\n"
    "1. Найди, есть ли в данном тексте адрес подачи или адрес назначения.\n"
    "2. Если адрес найден — исправь орфографические ошибки (названия улиц и проспектов должны точно совпадать с официальными улицами города Минска, можешь “догадаться” по похожести).\n"
    "3. Каждый адрес пиши полностью — тип улицы, название и номер дома (если есть).\n"
    "4. Если в тексте нет какого-то адреса — продублируй фразу 'нет информации'.\n"
    "Формат ответа:\n"
    "Подача: <адрес подачи или 'нет информации'>\n"
    "Назначение: <адрес назначения или 'нет информации'>"
    "\n\n"
    "Некоторые официальные улицы Минска для ориентира: проспект Победителей, улица Ленина, проспект Независимости, улица Якуба Коласа, улица Кальварийская, проспект Машерова, улица Немига, улица Короля, улица Куйбышева, улица Аэродромная."
)
########################################################

def build_llm_tokenizer_and_pipeline(huggingface_token):
    tokenizer = AutoTokenizer.from_pretrained(YANDEX_LLM, use_auth_token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(
        YANDEX_LLM,
        use_auth_token=huggingface_token,
        # quantization_config=...,  # если нужен quantized — раскомментируй и задай config!
        # device_map="auto",        # если только CPU, можно "cpu"
    )
    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=160,
        device_map="auto"  # или "cpu"
    )
    return text_pipe

def handle_client_proc(conn, addr, huggingface_token):
    print(f"[{addr}] Новый процесс обработчика клиента")
    try:
        # Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=torch_dtype,
            use_auth_token=huggingface_token
        ).to(device)
        processor = AutoProcessor.from_pretrained(WHISPER_MODEL, use_auth_token=huggingface_token)
        tokenizer = WhisperTokenizer.from_pretrained(
            WHISPER_MODEL, language=LANGUAGE, task="transcribe", use_auth_token=huggingface_token
        )

        # LLM pipeline
        text_pipe = build_llm_tokenizer_and_pipeline(huggingface_token)

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

            # Формируем prompt
            prompt = f"<s>[INST]{SYSTEM_PROMPT}\n{transcription}[/INST]"
            llm_response = text_pipe(prompt)[0]['generated_text']
            print(f"[{addr}] LLM результат:\n{llm_response}")
            # Передача клиенту при необходимости:
            # conn.sendall(llm_response.encode('utf-8'))

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
