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

import socket
import multiprocessing as mp
import numpy as np
import os
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer,
    AutoTokenizer, AutoModelForTokenClassification, pipeline
)
from dotenv import load_dotenv
import torch
from rapidfuzz import fuzz, process
import pymorphy2
import re

##################### Параметры ########################
HOST = "0.0.0.0"
PORT = 8082
BUFFER_SIZE = 48000 * 2
OVERLAP_SIZE = 48000 // 2
SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-large-v3-turbo"
LANGUAGE = "ru"
NER_MODEL = "aidarmusin/address-ner-ru"
NER_ATTRS = ["Street", "House", "Building"]
STREETS_FILE = "minsk_unique_streets2.txt"
OBJECT_MAPPING_FILE = "important_objects_mapping.txt"
NUMERALS_FILE = "russian_numerals.txt"
########################################################

morph = pymorphy2.MorphAnalyzer()

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

# --- Загрузка маппинга вариантов важных мест ---
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

# --- Быстрый поиск канонического важного объекта ---
def find_canonical_object(text, mapping, variants):
    text_low = text.lower()
    for variant in sorted(variants, key=len, reverse=True):
        if re.search(rf'\b{re.escape(variant)}\b', text_low):
            return mapping[variant]
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

# Паттерны для больниц, поликлиник и др. с номерами (из canons!)
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

def lemmatize_word(word):
    cleaned = re.sub(r'[^а-яё-]', '', word.lower())
    if not cleaned:
        return word
    return morph.parse(cleaned)[0].normal_form

def wordnum_to_int(word, NUM_WORDS):
    word_clean = word.lower().replace("-", " ").replace("ё", "е").strip()
    if word_clean in NUM_WORDS:
        return NUM_WORDS[word_clean]
    total = 0
    for w in word_clean.split():
        if w in NUM_WORDS:
            total += NUM_WORDS[w]
    if total:
        return total
    return None

def extract_number(word, NUM_WORDS):
    w = word.lower()
    mnum = re.match(r"(\d+)", w)
    if mnum:
        return int(mnum.group(1))
    w = re.sub(r'[-–—]?(й|я|е|ая|ое|ый|ий|ую|ой|ем|ым|ом|их|ыми|ого|ий|ие|ье|ая)?$', '', w)
    if w.isdigit():
        return int(w)
    n = wordnum_to_int(w, NUM_WORDS)
    return n

def find_special_object(text, patterns, NUM_WORDS, canons_set=None):
    text_low = text.lower()
    words = text_low.split()
    lemmas = [lemmatize_word(w) for w in words]
    if canons_set is not None:
        # num + obj
        for i in range(len(words) - 1):
            obj_candidate = lemmas[i+1]
            num_candidate = extract_number(words[i], NUM_WORDS)
            for obj in canons_set:
                obj_lemma = lemmatize_word(obj)
                if obj_candidate == obj_lemma and num_candidate:
                    return f"{obj} {num_candidate}"
        # obj + num
        for i in range(len(words) - 1):
            obj_candidate = lemmas[i]
            num_candidate = extract_number(words[i+1], NUM_WORDS)
            for obj in canons_set:
                obj_lemma = lemmatize_word(obj)
                if obj_candidate == obj_lemma and num_candidate:
                    return f"{obj} {num_candidate}"
        # Просто объект
        for i, lemma in enumerate(lemmas):
            for obj in canons_set:
                if lemma == lemmatize_word(obj):
                    return obj
    for pattern in patterns:
        for match in pattern.finditer(text):
            gd = match.groupdict()
            obj_type = gd.get('object')
            num_raw = gd.get('num')
            if obj_type and num_raw:
                obj_type = obj_type.lower()
                num = extract_number(num_raw, NUM_WORDS)
                if num:
                    label = f"{obj_type} {num}"
                    return label
                else:
                    return f"{obj_type} {num_raw}"
    return None

def handle_client_proc(conn, addr, huggingface_token):
    print(f"[{addr}] Новый процесс обработчика клиента")
    try:
        def transcribe_audio(audio_data):
            if np.max(np.abs(audio_data)) != 0:
                audio_data = (audio_data / np.max(np.abs(audio_data))).astype(np.float32)
            else:
                audio_data = audio_data.astype(np.float32)
            input_features = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device).to(torch_dtype)
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
            print(f"[{addr}] Транскрипция: {transcription}")
            # 1. (самое приоритетное) Найти по mapping'у все нестандартные ТЦ/места (корона, мома, ...)
            label = find_canonical_object(transcription, obj_mapping, obj_variants)
            if label:
                print(f"[{addr}] Найден важный объект (канонизация): {label}")
                return
            # 2. Найти по canons — все больницы/поликлиники с номерами/без
            label = find_special_object(transcription, obj_patterns, NUM_WORDS, obj_canons)
            if label:
                print(f"[{addr}] Найден важный объект: {label}")
                return
            # 3. Классический адрес через NER
            ents = extract_address_entities(transcription, ner_pipe)
            selected = extract_selected_attributes(ents, attributes=NER_ATTRS)
            selected = correct_address(selected, streets_lower, streets_original)
            out = []
            for attr in NER_ATTRS:
                if attr in selected:
                    word, score = selected[attr]
                    out.append(word)
            addr_str = '-'.join(out)
            if addr_str:
                print(f"[{addr}] Адрес найден: {addr_str}")
            else:
                print(f"[{addr}] Адрес не найден в этом фрагменте.")
        audio_buffer = bytearray()
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                audio_buffer.extend(data)
                while len(audio_buffer) >= BUFFER_SIZE:
                    block = audio_buffer[:BUFFER_SIZE]
                    audio_data = np.frombuffer(block, np.int16)
                    transcribe_audio(audio_data)
                    audio_buffer = audio_buffer[BUFFER_SIZE - OVERLAP_SIZE:]
    except Exception as e:
        print(f"[{addr}] Ошибка: {e}")
    finally:
        conn.close()
        print(f"[{addr}] Завершение сессии")

def main():
    load_dotenv()
    huggingface_token = os.environ['HF_TOKEN']
    print(f"Запуск сервера на {HOST}:{PORT}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            use_auth_token=huggingface_token
        ).to(device)
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=huggingface_token)
        tokenizer = WhisperTokenizer.from_pretrained(
            MODEL_NAME, language=LANGUAGE, task="transcribe", use_auth_token=huggingface_token
        )
        print(f"[{addr}] Загружаем Address-NER модель...")
        ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, token=huggingface_token)
        ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL, token=huggingface_token)
        ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
        print(f"[{addr}] Address-NER загружена.")
        streets_original, streets_lower = load_streets_dict(STREETS_FILE)
        obj_mapping, obj_variants, obj_canons = load_object_canon_mapping(OBJECT_MAPPING_FILE)
        NUM_WORDS = load_dict(NUMERALS_FILE)
        obj_patterns = build_special_object_patterns(obj_canons)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(32)
        while True:
            conn, addr = server_sock.accept()
            proc = mp.Process(target=handle_client_proc, args=(conn, addr, huggingface_token), daemon=True)
            proc.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
