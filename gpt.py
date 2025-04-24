import socket
import multiprocessing as mp
import numpy as np
import os
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer,
    AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
)
from dotenv import load_dotenv
import torch

##################### Параметры ########################
HOST = "0.0.0.0"
PORT = 8084
BUFFER_SIZE = 32000 * 2    # 2 сек для 16kHz 16bit mono
OVERLAP_SIZE = 32000 // 2  # 0.5 сек перекрытия
SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-llm-7b-instruct-v1.5"
########################################################

PROMPT = (
    "Ты ассистент такси Минска. Ты получаешь фрагменты диалогов между пассажиром и водителем."
    "Твоя задача — найти адрес подачи и назначения, если они есть, исправить орфографию в соответствии с улицами Минска."
    "Формат ответа:\nПодача: <адрес/нет информации>\nНазначение: <адрес/нет информации>."
    "Не повторяй инструкцию и пиши только в этом формате."
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype='float16',
)

def process_text_with_llm(text, llm_request_q, llm_response_q, worker_id):
    import os
    # Для уникального id запроса ― вдруг параллельность
    request_id = (os.getpid(), worker_id, np.random.randint(0, 1e9))
    full_prompt = f"<|user|>\n{PROMPT}\n{text}\n<|assistant|>\n"
    llm_request_q.put((request_id, full_prompt))
    while True:
        resp_id, result = llm_response_q.get()
        if resp_id == request_id:
            return result

def handle_client_proc(conn, addr, huggingface_token, llm_request_q, llm_response_q, worker_id):
    print(f"[{addr}] Новый процесс обработчика клиента")
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
            # --- Отправляем транскрипт в Deepseek ---
            llm_response = process_text_with_llm(transcription, llm_request_q, llm_response_q, worker_id)
            print(f"[{addr}] Deepseek результат:\n{llm_response}")

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

def llm_worker(llm_request_q, llm_response_q, huggingface_token):
    print("Deepseek: Загружаю модель (LLM, держать терпение 5-15 минут первый запуск)...")
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(
        DEEPSEEK_MODEL,
        token=huggingface_token,
        quantization_config=quant_config,
        device_map="auto"
    )
    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=160,
        device_map="auto"
    )
    print("Deepseek: Модель готова к работе!")
    while True:
        request = llm_request_q.get()
        if request is None:
            break
        request_id, full_prompt = request
        try:
            output = text_pipe(full_prompt)[0]['generated_text']
        except Exception as e:
            output = f"Deepseek LLM error: {e}"
        llm_response_q.put((request_id, output))

def main():
    load_dotenv()
    huggingface_token = os.environ['HF_TOKEN']
    llm_request_q = mp.Queue()
    llm_response_q = mp.Queue()
    # Запускаем Deepseek один раз!
    llm_proc = mp.Process(target=llm_worker, args=(llm_request_q, llm_response_q, huggingface_token), daemon=True)
    llm_proc.start()
    print(f"Запуск сервера на {HOST}:{PORT}")
    worker_id = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(32)
        while True:
            conn, addr = server_sock.accept()
            worker_id += 1
            proc = mp.Process(
                target=handle_client_proc,
                args=(conn, addr, huggingface_token, llm_request_q, llm_response_q, worker_id),
                daemon=True
            )
            proc.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
