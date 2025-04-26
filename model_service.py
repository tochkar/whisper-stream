from fastapi import FastAPI, File, UploadFile, Form
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer,
                          AutoModelForTokenClassification, AutoTokenizer, pipeline)
import torch
import numpy as np
import uvicorn
import os

MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
NER_MODEL = "aidarmusin/address-ner-ru"
HF_TOKEN = os.environ["HF_TOKEN"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Загрузка Whisper...")
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, token=HF_TOKEN).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe", token=HF_TOKEN)

print("Загрузка NER...")
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, token=HF_TOKEN)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL, token=HF_TOKEN)
ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

app = FastAPI()

@app.post("/asr/")
async def transcribe(audio: UploadFile = File(...)):
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
    transcription = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return {"text": transcription}

@app.post("/ner/")
async def extract_ner(text: str = Form(...)):
    out = ner_pipe(text)
    return {"entities": out}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
