from fastapi import FastAPI, File, UploadFile, Form
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer,
    AutoModelForTokenClassification, AutoTokenizer, pipeline
)
import torch
import numpy as np
import os
import re
import uvicorn

MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
NER_MODEL = "aidarmusin/address-ner-ru"
NER_ATTRS = ["Street", "House", "Building"]
MAPPING_FILE = "important_objects_mapping.txt"    # путь до вашего mapping-файла

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
HF_TOKEN = os.environ["HF_TOKEN"]

app = FastAPI()

# --- Загрузка моделей и пайплайнов ---
print("Загрузка Whisper...")
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, token=HF_TOKEN).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe", token=HF_TOKEN)

print("Загрузка Address-NER...")
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, token=HF_TOKEN)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL, token=HF_TOKEN)
ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

# --- Маппинг --- #
def load_object_canon_mapping(filepath):
    mapping = {}
    variants = set()
    canons = set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if ':' not in line:
                continue
            var, canon = line.strip().split(':', 1)
            v = var.strip().lower()
            c = canon.strip()
            mapping[v] = c
            variants.add(v)
            canons.add(c)
    return mapping, variants, canons

obj_mapping, obj_variants, obj_canons = load_object_canon_mapping(MAPPING_FILE)

def find_canonical_object(text, mapping, variants):
    text_low = text.lower()
    for variant in sorted(variants, key=len, reverse=True):
        if re.search(rf'\b{re.escape(variant)}\b', text_low):
            return mapping[variant]
    return None

@app.post("/address/")
async def extract_address(audio: UploadFile = File(None), text: str = Form(None)):
    if audio:
        raw = await audio.read()
        audio_np = np.frombuffer(raw, np.int16)
        if np.max(np.abs(audio_np)) != 0:
            audio_np = (audio_np / np.max(np.abs(audio_np))).astype(np.float32)
        else:
            audio_np = audio_np.astype(np.float32)
        input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        predicted_ids = asr_model.generate(
            input_features,
            max_length=1024,
            do_sample=True,
            temperature=1.5,
            top_k=100, top_p=0.15,
            no_repeat_ngram_size=2
        )
        text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    # Если нет аудио — принимаем text
    elif not text:
        return {"error": "Нужно передать either audio (audio/raw) или text"}
    
    # 1. Сначала пытаем mapping (канонические объекты)
    label = find_canonical_object(text, obj_mapping, obj_variants)
    if label:
        return {"result": label, "asr": text}

    # 2. Потом NER пайплайн HuggingFace
    ner_res = ner_pipe(text)
    found = {}
    for ent in ner_res:
        group = ent['entity_group']
        if group == "O":
            continue
        if group not in found or ent['score'] > found[group]['score']:
            found[group] = {"word": ent["word"], "score": float(ent["score"])}
    return {"asr": text, "ner": found}

# (если нужно — добавь отдельные /asr или /ner эндпоинты по аналогии)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
