import json
import base64
import paho.mqtt.client as mqtt
import torch
import numpy as np
import os
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv

# Конфигурация подключения
BROKER = "socket.taxi135.by"
PORT = 1883
USERNAME = "admin"
PASSWORD = "6BHK2pGn3d"
CLIENT_ID = "ai_service"
DISPATCHER_ID = "{dispatcher_id}"  # Замените на фактический ID диспетчера
OUTPUT_TOPIC_PATTERN = f"dispatcher/{DISPATCHER_ID}/output/ai/"

# Загрузить переменные из .env
load_dotenv()
huggingface_token = os.environ['HF_TOKEN']

# Настройки устройства и модели
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"

# Загрузка модели и процессора
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_auth_token=huggingface_token
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=huggingface_token)
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
    return transcription

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Подключено с кодом результата {rc}")
    client.subscribe("dispatcher/{dispatcher_id}/input/ai/")

def on_message(client, userdata, msg):
    print(f"Получено сообщение в топике {msg.topic}")

    try:
        # Декодирование сообщения
        message_data = json.loads(msg.payload)
        if "audio" in message_data:
            audio_b64 = message_data["audio"]
            
            # Декодируем base64
            audio_bytes = base64.b64decode(audio_b64)
            
            tmp_file = "temp_audio.mp3"
            with open(tmp_file, "wb") as f:
                f.write(audio_bytes)

            # Чтение и преобразование mp3 в wav
            process = (
                ffmpeg
                .input(tmp_file)
                .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            audio_buffer = process.stdout.read()
            audio_data = np.frombuffer(audio_buffer, np.int16)
            
            # Транскрибировать
            transcription = transcribe_audio(audio_data, 16000)
            
            # Публикация результата
            client.publish(OUTPUT_TOPIC_PATTERN, json.dumps({"transcription": transcription}))

    except json.JSONDecodeError:
        print("Ошибка декодирования JSON")

def main():
    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(USERNAME, PASSWORD)
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()
