import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]  # ваш Hugging Face токен

model_name = "IlyaGusev/saiga-mistral-7b-qlora"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

def extract_addresses(text):
    system_prompt = (
        "Ты будешь получать куски разговора. "
        "ЕСЛИ ЕСТЬ в куске адрес посадки и адрес назначения, то исправь орфографические ошибки в соответствии с названиями улиц Минска. "
        "Твоя задача: из этих кусков диалога извлечь адрес посадки пассажира и адрес назначения. "
        "Не отвечай лишнего, просто ответь в виде:\n"
        "Подача: <адрес1>\nНазначение: <адрес2>\n"
        "Если в куске нет информации - напиши 'нет информации'."
    )
    prompt = f"<s>[INST]{system_prompt}\n{text}[/INST]"
    output = pipe(prompt)[0]['generated_text']
    return output

# Тест!
if __name__ == "__main__":
    test_text = "Меня заберите, пожалуйста, с улицы Лебедева, 12, а везите на Короля 21"
    print(extract_addresses(test_text))
