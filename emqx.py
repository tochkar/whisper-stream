import paho.mqtt.client as mqtt
import json
import time

MQTT_BROKER = "socket.taxi135.by"
MQTT_PORT = 443
MQTT_USERNAME = "admin"
MQTT_PASSWORD = "6BHK2pGn3d"
MQTT_CLIENT_ID = "ai_service"
MQTT_TOPIC = f"ai/recognize/taxi"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Соединение с брокером установлено!")
    else:
        print(f"Ошибка соединения: {rc}")

def on_publish(client, userdata, mid):
    print(f"Сообщение опубликовано (mid={mid})!")

def on_message(client, userdata, msg):
    print(f"Получено сообщение в топике '{msg.topic}': {msg.payload.decode()}")

client = mqtt.Client(client_id=MQTT_CLIENT_ID, transport="websockets")
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_publish = on_publish
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
client.loop_start()

connected = False
for _ in range(50):
    if client.is_connected():
        connected = True
        break
    time.sleep(0.1)

if not connected:
    print("Не удалось установить соединение с брокером!")
    client.loop_stop()
    exit(1)
    
payload = json.dumps({"test": "message", "number": 123, "status": "ok"})
print(f"Публикуем: {payload}")
client.publish(MQTT_TOPIC, payload)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Отключение от брокера...")
    client.loop_stop()
    client.disconnect()
    print("Всё завершено.")
