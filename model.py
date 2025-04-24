import os
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
)

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

model_name = "yandex/YandexGPT-5-Lite-8B-instruct"

# Параметры 4bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  # наилучшее качество quantization для LLM
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype='float16',
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN,
    quantization_config=quant_config,
    device_map="auto",          # можно "cuda:0" или "cpu"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=160,
    device_map="auto"           # авторазмещение, но если только CPU — укажи "cpu"
)

PROMPT = (
    "Ты бот такси Минска. Тебе в диалоге могут назвать адреса в свободной форме: 'забери с цирка' или 'отвези на победителей двадцать три'. "
    "Твоя задача — если в куске есть оба адреса, исправь орфографию по официальным названиям улиц и номеров домов Минска и ответь только так:\n"
    "Подача: <адрес1>\nНазначение: <адрес2>\n"
    "Если данных нет — напиши 'нет информации'."
)

def extract_addresses_llm(text):
    # Инструкция/промпт
    prompt = f"<s>[INST]{PROMPT}\n{text}[/INST]"
    result = pipe(prompt)[0]['generated_text']
    return result

# Пример использования
if __name__ == "__main__":
    fragment = "Забери от плошади победы и перевези на притыцкого сорок два."
    print(extract_addresses_llm(fragment))
