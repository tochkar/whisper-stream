from socket import socket, AF_INET, SOCK_STREAM
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv
import os

# Загрузка переменных окружения
load_dotenv()
huggingface_token = os.environ['HF_TOKEN']

class FreeswitchAMI:
    def __init__(self, host, port, password):
        # Инициализация подключения к FreeSWITCH
        self.host = host
        self.port = port
        self.password = password
        self.socket = None
        
        # Инициализация Whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16
        self.MODEL_NAME = "openai/whisper-large-v3-turbo"
        
        # Загрузка модели и процессора
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_auth_token=huggingface_token
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_NAME, 
            use_auth_token=huggingface_token
        )
        
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.MODEL_NAME, 
            language="ru", 
            task="transcribe", 
            use_auth_token=huggingface_token
        )
        
        # Буфер для аудио
        self.audio_buffer = bytearray()
        self.buffer_size = 16000 * 4  # 4 секунды
        self.overlap_size = 16000 * 1  # 1 секунда

    def connect(self):
        try:
            self.socket = socket(AF_INET, SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
            welcome = self.socket.recv(1024).decode()
            print(f"Connected: {welcome}")

            auth_command = f"auth {self.password}\n\n"
            self.socket.send(auth_command.encode())
            
            auth_response = self.socket.recv(1024).decode()
            print(f"Auth response: {auth_response}")

            self.socket.send("event plain CHANNEL_ANSWER\n\n".encode())
            response = self.socket.recv(1024).decode()
            print(f"Subscribe response: {response}")

            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def transcribe_audio(self, audio_data):
        try:
            # Преобразование байтов в numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / np.max(np.abs(audio_np))

            # Подготовка входных данных для модели
            input_features = self.processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device).to(torch.float16)

            # Генерация транскрипции
            predicted_ids = self.model.generate(
                input_features,
                max_length=1024,
                do_sample=True,
                temperature=1.5,
                top_k=100,
                top_p=0.15,
                no_repeat_ngram_size=2
            )

            # Декодирование результата
            transcription = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print(f"Transcription: {transcription}")

        except Exception as e:
            print(f"Error in transcription: {e}")

    def listen_events(self):
        try:
            while True:
                event = self.socket.recv(4096)
                if event:
                    try:
                        # Пробуем декодировать как текстовое событие
                        event_str = event.decode()
                        event_data = self.parse_event(event_str)
                        
                        if event_data:
                            if (event_data.get('Event-Name') == 'CHANNEL_ANSWER' and 
                                event_data.get('Channel-Call-State') == 'RINGING'):
                                print("Обнаружен новый звонок")
                                
                    except UnicodeDecodeError:
                        # Если не удалось декодировать - это аудио данные
                        self.audio_buffer.extend(event)
                        
                        # Если накопили достаточно данных
                        if len(self.audio_buffer) >= self.buffer_size:
                            # Транскрибируем накопленный буфер
                            self.transcribe_audio(self.audio_buffer)
                            # Оставляем перекрытие
                            self.audio_buffer = self.audio_buffer[-self.overlap_size:]

        except Exception as e:
            print(f"Error in event listener: {e}")

    def parse_event(self, event):
        try:
            lines = event.split('\n')
            event_data = {}
            for line in lines:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    event_data[key] = value
            return event_data
        except Exception as e:
            print(f"Error parsing event: {e}")
            return None

    def close(self):
        if self.socket:
            self.socket.close()

def main():
    host = "192.168.100.226"
    port = 8021
    password = "gobots"

    ami = FreeswitchAMI(host, port, password)

    if ami.connect():
        print("Successfully connected to FreeSWITCH AMI")
        try:
            ami.listen_events()
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
        finally:
            ami.close()
    else:
        print("Failed to connect")

if __name__ == "__main__":
    main()
