from socket import socket, AF_INET, SOCK_STREAM
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.environ['HF_TOKEN']

class FreeswitchAMI:
    def __init__(self, host, port, password):
        self.host = host
        self.port = port
        self.password = password
        self.socket = None
        
        print("Инициализация Whisper модели...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используемое устройство: {self.device}")
        self.torch_dtype = torch.float16
        self.MODEL_NAME = "openai/whisper-large-v3-turbo"
        
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
        print("Модель Whisper инициализирована")

        self.audio_buffer = bytearray()
        self.buffer_size = 8000 * 2  # 2 секунды (8kHz * 2)
        self.overlap_size = 8000 * 0.5  # 0.5 секунды перекрытия
        print(f"Размер буфера: {self.buffer_size} байт")
        print(f"Размер перекрытия: {self.overlap_size} байт")

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

            self.socket.send("event plain all\n\n".encode())
            response = self.socket.recv(1024).decode()
            print(f"Subscribe response: {response}")

            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def start_audio_stream(self, call_uuid):
        try:
            print(f"\nПодключаемся к аудио потоку звонка: {call_uuid}")
            command = f"uuid_audio {call_uuid} start\n\n"
            self.socket.send(command.encode())
            response = self.socket.recv(4096)
            try:
                print(f"Ответ на команду start_audio_stream: {response.decode()}")
            except UnicodeDecodeError:
                print(f"Получен бинарный ответ размером {len(response)} байт")
            print("Ожидаем поступления аудио потока...")
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def transcribe_audio(self, audio_bytes):
        try:
            print("\n--- Начало обработки аудио фрагмента ---")
            print(f"Размер фрагмента: {len(audio_bytes)} байт")
            print(f"Первые 20 байт (hex): {audio_bytes[:20].hex()}")

            # Преобразуем байты в numpy array и нормализуем
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            print(f"Форма массива: {audio_np.shape}")
            print(f"Мин/макс значения: {np.min(audio_np):.3f}/{np.max(audio_np):.3f}")

            # Обработка аудио
            print("Обработка через процессор Whisper...")
            input_features = self.processor(
                audio_np,
                sampling_rate=8000,
                return_tensors="pt"
            ).input_features.to(self.device).to(torch.float16)
            print("Аудио обработано процессором")

            # Генерация транскрипции
            print("Генерация транскрипции...")
            predicted_ids = self.model.generate(
                input_features,
                max_length=1024,
                do_sample=True,
                temperature=1.5,
                top_k=100,
                top_p=0.15,
                no_repeat_ngram_size=2
            )
            print("Генерация завершена")

            # Декодирование результата
            transcription = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print("\n=== Результат транскрипции ===")
            if transcription.strip():
                print(f"Текст: {transcription}")
            else:
                print("Текст не обнаружен")
            print("=" * 50)

        except Exception as e:
            print(f"Ошибка при транскрипции: {e}")
            import traceback
            print(traceback.format_exc())

    def listen_events(self):
        try:
            print("Ожидаем события...")
            while True:
                event = self.socket.recv(4096)
                if event:
                    try:
                        # Пробуем декодировать как текстовое событие
                        event_str = event.decode()
                        event_data = self.parse_event(event_str)
                        
                        if event_data:
                            event_name = event_data.get('Event-Name')
                            print(f"\nПолучено событие: {event_name}")
                            
                            if event_name == 'CHANNEL_ANSWER':
                                call_uuid = event_data.get('Unique-ID')
                                print(f"\nНовый звонок: {call_uuid}")
                                print("Детали звонка:")
                                for key in ['Channel-Read-Codec-Name', 'Channel-Read-Codec-Rate']:
                                    print(f"{key}: {event_data.get(key)}")
                                self.start_audio_stream(call_uuid)

                    except UnicodeDecodeError:
                        # Аудио данные
                        print(f"\rПолучены аудио данные: {len(event)} байт", end="")
                        self.audio_buffer.extend(event)
                        
                        if len(self.audio_buffer) >= self.buffer_size:
                            self.transcribe_audio(bytes(self.audio_buffer))
                            self.audio_buffer = self.audio_buffer[-int(self.overlap_size):]
                            print(f"Буфер обновлен, новый размер: {len(self.audio_buffer)}")

        except Exception as e:
            print(f"Error in event listener: {e}")
            import traceback
            print(traceback.format_exc())

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
