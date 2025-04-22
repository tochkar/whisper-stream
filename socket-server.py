import socket

HOST = "0.0.0.0"
PORT = 8084

def process_audio_bytes(audio_bytes):
    # Пример: возвращаем строку (замените на свой результат - текст/байты и т.д.)
    print(f"Пачка байт: {len(audio_bytes)}, первые 16 байт: {audio_bytes[:16]}")
    return b'RESPONSE_FROM_PYTHON\r\n'  # или return любые bytes

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Жду подключения от FreeSWITCH на {HOST}:{PORT} ...")
    conn, addr = srv.accept()
    print('Подключено:', addr)
    with conn:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            result = process_audio_bytes(data)
            if result:
                if isinstance(result, str):
                    result = result.encode()
                conn.sendall(result)
