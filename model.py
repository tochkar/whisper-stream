from fastapi import FastAPI, File, UploadFile
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer,
    AutoModelForTokenClassification, AutoTokenizer, pipeline
)
import torch
import numpy as np
import os
import inspect
# --- Shim для Python 3.11+/3.12+, чтобы pymorphy2 работал с getargspec ---
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        from collections import namedtuple
        FullArgSpec = inspect.getfullargspec(func)
        ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
        return ArgSpec(
            args=FullArgSpec.args,
            varargs=FullArgSpec.varargs,
            keywords=FullArgSpec.varkw,
            defaults=FullArgSpec.defaults
        )
    inspect.getargspec = getargspec
import pymorphy2
import re
from rapidfuzz import process, fuzz

MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
NER_MODEL = "aidarmusin/address-ner-ru"
NER_ATTRS = ["Street", "House", "Building"]
STREETS_FILE = "minsk_unique_streets2.txt"
MAPPING_FILE = "important_objects_mapping.txt"
NUMERALS_FILE = "russian_numerals.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
HF_TOKEN = os.environ["HF_TOKEN"]

# --- Загрузка ключевых объектов из файла ---
def load_objects_with_number(fname):
    with open(fname, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

OBJECTS_WITH_NUMBER = load_objects_with_number('objects_with_number.txt')

# --- Фильтр результата ---
def is_bad_result(result):
    if not result:
        return True
    if '#' in result:
        return True
    if result.replace(' ', '').isdigit():
        return True
    return False

# --- КОНЕЦ импортов и фильтров, остальные функции не менялись ---

app = FastAPI()
morph = pymorphy2.MorphAnalyzer()
print("Загрузка Whisper...")
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, token=HF_TOKEN).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe", token=HF_TOKEN)
print("Загрузка Address-NER...")
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, token=HF_TOKEN)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL, token=HF_TOKEN)
ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

def load_streets_dict(filepath):
    original, lowered = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                original.append(s)
                lowered.append(s.lower())
    return original, lowered

def load_dict(filepath, sep=":"):
    d = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if sep in line:
                k, v = line.strip().split(sep, 1)
                d[k.strip()] = int(v.strip())
    return d

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

streets_original, streets_lower = load_streets_dict(STREETS_FILE)
obj_mapping, obj_variants, obj_canons = load_object_canon_mapping(MAPPING_FILE)
NUM_WORDS = load_dict(NUMERALS_FILE)

def lemmatize_word(word):
    cleaned = re.sub(r'[^а-яё-]', '', word.lower())
    if not cleaned:
        return word
    return morph.parse(cleaned)[0].normal_form

def extract_number(words, NUM_WORDS):
    w = words.lower()
    w = w.replace('-', ' ')
    mnum = re.match(r"(\d+)", w)
    if mnum:
        return int(mnum.group(1))
    if w in NUM_WORDS:
        return NUM_WORDS[w]
    parts = w.split()
    for n in range(len(parts), 0, -1):
        sub = " ".join(parts[:n])
        if sub in NUM_WORDS:
            return NUM_WORDS[sub]
    return None

def make_lemma(s):
    return ' '.join(lemmatize_word(w) for w in re.findall(r'\w+', s.lower()))

variant_lemmas_phrase = {}
for v in obj_variants:
    l = make_lemma(v)
    if len(l.replace(' ', '')) >= 4:
        variant_lemmas_phrase[l] = obj_mapping[v]

def find_canonical_object_ngram(text, variant_lemmas):
    words = re.findall(r'\w+', text.lower())
    if not words:
        return None
    ngram_min = 1
    ngram_max = max(len(l.split()) for l in variant_lemmas)
    for n in range(ngram_max, ngram_min-1, -1):
        for i in range(len(words)-n+1):
            frag = words[i:i+n]
            frag_lemma = ' '.join(lemmatize_word(w) for w in frag)
            if frag_lemma in variant_lemmas:
                return variant_lemmas[frag_lemma]
    return None

def build_special_object_patterns(canons):
    objpat = '|'.join([re.escape(x) for x in sorted(canons, key=len, reverse=True)])
    numpat = r'(\d+|[а-яё\- ]+)'
    pattern1 = re.compile(
        rf'(?P<object>{objpat})\s*(?:№|номер|num|n\.?|n|N|N\.|#)?\s*(?P<num>{numpat})\b',
        re.IGNORECASE)
    pattern2 = re.compile(
        rf'(?P<num>{numpat})\s*(?P<object>{objpat})\b',
        re.IGNORECASE)
    return [pattern1, pattern2]
obj_patterns = build_special_object_patterns(obj_canons)

def extract_object_with_number(text, object_keywords, NUM_WORDS):
    words = re.findall(r'\w+', text.lower())
    lemmas = [lemmatize_word(w) for w in words]
    object_found = None
    for win in [3, 2]:
        for i in range(len(words) - win + 1):
            window = words[i:i+win]
            window_lemmas = [lemmatize_word(w) for w in window]
            for obj in object_keywords:
                obj_words = obj.split()
                obj_lemmas = [lemmatize_word(w) for w in obj_words]
                # односоставные объекты
                if len(obj_lemmas) == 1:
                    for idx, lemma in enumerate(window_lemmas):
                        if lemma == obj_lemmas[0]:
                            for j, w in enumerate(window):
                                if j == idx:
                                    continue
                                num = extract_number(w, NUM_WORDS)
                                if num:
                                    return f"{obj} {num}"
                            object_found = obj
                # двусоставные объекты ("детский сад", "родильный дом")
                elif len(obj_lemmas) == 2 and win >= 2:
                    for k in range(len(window_lemmas)-1):
                        if window_lemmas[k] == obj_lemmas[0] and window_lemmas[k+1] == obj_lemmas[1]:
                            before = extract_number(window[k-1], NUM_WORDS) if k-1 >= 0 else None
                            after = extract_number(window[k+2], NUM_WORDS) if k+2 < len(window) else None
                            if before:
                                return f"{obj} {before}"
                            if after:
                                return f"{obj} {after}"
                            object_found = obj
    if object_found:
        return object_found
    return None

def correct_street_name(name, streets_lower, streets_original, min_score=80):
    result = process.extractOne(name.lower(), streets_lower, scorer=fuzz.WRatio)
    if result:
        match, score, idx = result
        if score >= min_score:
            return streets_original[idx]
    return name

def correct_address(attrs_dict, streets_lower, streets_original):
    attrs_dict_corr = dict(attrs_dict)
    if "Street" in attrs_dict_corr:
        orig, score = attrs_dict_corr["Street"]
        fixed = correct_street_name(orig, streets_lower, streets_original)
        attrs_dict_corr["Street"] = (fixed, score)
    return attrs_dict_corr

def extract_address_entities(text, ner_pipe):
    entities = [ent for ent in ner_pipe(text) if ent['entity_group'] != 'O']
    return entities

def extract_selected_attributes(entities, attributes=None):
    if attributes is None:
        attributes = ['Street', 'House', 'Building']
    attr_map = {}
    for ent in entities:
        label = ent["entity_group"]
        word = ent["word"]
        score = ent["score"]
        if label not in attributes:
            continue
        if label not in attr_map:
            attr_map[label] = (word, score)
        else:
            prev_word, prev_score = attr_map[label]
            if score > prev_score:
                attr_map[label] = (word, score)
            elif score == prev_score:
                if word != prev_word and word not in prev_word.split(" / "):
                    attr_map[label] = (prev_word + " / " + word, prev_score)
    return attr_map

@app.post("/address/")
async def extract_address(audio: UploadFile = File(...)):
    raw = await audio.read()
    audio_np = np.frombuffer(raw, np.int16)
    if np.max(np.abs(audio_np)) != 0:
        audio_np = (audio_np / np.max(np.abs(audio_np))).astype(np.float32)
    else:
        audio_np = audio_np.astype(np.float32)
    input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    input_features = input_features.to(dtype=torch_dtype)
    predicted_ids = asr_model.generate(
        input_features,
        max_length=1024,
        do_sample=True,
        temperature=1.5,
        top_k=100, top_p=0.15,
        no_repeat_ngram_size=2
    )
    text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
    # Проверка: если только одно слово из street_types — result = ''
    street_types = ['проспект', 'улица', 'аллея', 'переулок']
    words = re.findall(r'\w+', text.lower())
    only_word = len(words) == 1 and words[0] in street_types
    too_short = len(text.strip()) < 4
    if only_word or too_short:
        return {"result": "", "asr": text}

    # 0. Кейсы с объектами и числительными
    label = extract_object_with_number(text, OBJECTS_WITH_NUMBER, NUM_WORDS)
    if label and not is_bad_result(label):
        return {"result": label, "asr": text}

    # 1. mapping (по имени)
    label = find_canonical_object_ngram(text, variant_lemmas_phrase)
    if label and not is_bad_result(label):
        return {"result": label, "asr": text}

    # 2. спец паттерны
    for pattern in obj_patterns:
        for match in pattern.finditer(text):
            gd = match.groupdict()
            obj_type = gd.get('object')
            num_raw = gd.get('num')
            if obj_type and num_raw:
                obj_type = obj_type.lower()
                num = extract_number(num_raw, NUM_WORDS)
                if num:
                    label = f"{obj_type} {num}"
                else:
                    label = f"{obj_type} {num_raw}"
                if not is_bad_result(label):
                    return {"result": label, "asr": text}
                else:
                    return {"result": "", "asr": text}

    # 3. NER типа улица-дом-корпус
    ents = extract_address_entities(text, ner_pipe)
    selected = extract_selected_attributes(ents, attributes=NER_ATTRS)
    selected = correct_address(selected, streets_lower, streets_original)
    out = []
    for attr in NER_ATTRS:
        if attr in selected:
            word, score = selected[attr]
            out.append(word)
    addr_str = '-'.join(out)
    if addr_str and not is_bad_result(addr_str):
        return {"result": addr_str, "asr": text}
    else:
        return {"result": "", "asr": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
