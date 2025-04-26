import socket
import multiprocessing as mp
import numpy as np
import os
import requests
from dotenv import load_dotenv
import inspect
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
from rapidfuzz import fuzz, process

HOST = "0.0.0.0"
PORT = 8082
BUFFER_SIZE = 48000 * 2
OVERLAP_SIZE = 48000 // 2
SAMPLE_RATE = 16000
MODEL_SERVER_URL = "http://localhost:7000"

morph = pymorphy2.MorphAnalyzer()

# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def load_streets_dict(filepath):
    original, lowered = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                original.append(s)
                lowered.append(s.lower())
    return original, lowered

def correct_street_name(name, streets_lower, streets_original, min_score=80):
    result = process.extractOne(name.lower(), streets_lower, scorer=fuzz.WRatio)
    if result:
        match, score, idx = result
        if score >= min_score:
            return streets_original[idx]
    return name

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
            v = var.strip().lower(); c = canon.strip()
            mapping[v] = c; variants.add(v); canons.add(c)
    return mapping, variants, canons

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

def correct_address(attrs_dict, streets_lower, streets_original):
    attrs_dict_corr = dict(attrs_dict)
    if "Street" in attrs_dict_corr:
        orig, score = attrs_dict_corr["Street"]
        fixed = correct_street_name(orig, streets_lower, streets_original)
        attrs_dict_corr["Street"] = (fixed, score)
    return attrs_dict_corr

def lemmatize_word(word):
    cleaned = re.sub(r'[^а-яё-]', '', word.lower())
    if not cleaned:
        return word
    return morph.parse(cleaned)[0].normal_form

def find_canonical_object(text, mapping, variants):
    text_low = text.lower()
    for variant in sorted(variants, key=len, reverse=True):
        if re.search(rf'\b{re.escape(variant)}\b', text_low):
            return mapping[variant]
    return None

# --- функции wordnum_to_int, extract_number, find_special_object — как в ранних ответах ---

def extract_ner_remotely(text):
    resp = requests.post(f"{MODEL_SERVER_URL}/ner/", data={"text": text})
    if resp.ok:
        return resp.json()["entities"]
    else:
        return []

def transcribe_remotely(audio_bytes):
    resp = requests.post(
        f"{MODEL_SERVER_URL}/asr/",
        files={"audio": ("audio.raw", audio_bytes)}
    )
    if resp.ok:
        return resp.json()["text"]
    else:
        return ""

def handle_client_proc(conn, addr, streets_original, streets_lower, obj_mapping, obj_variants):
    try:
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
                    
                    # 1. Транскрибация через MODEL SERVICE (Whisper)
                    audio_bytes = audio_data.tobytes()
                    transcription = transcribe_remotely(audio_bytes)
                    print(f"[{addr}] Транскрипция:", transcription)

                    # 2. NER через MODEL SERVICE
                    entities = extract_ner_remotely(transcription)
                    print(f"[{addr}] Raw NER:", entities)

                    # 3. Здесь ниже скопируй/подкрути свою канонизацию ВСЕХ объектов — 
                    # как это делал раньше, до разделения моделей и логики
                    selected = extract_selected_attributes(entities)
                    selected = correct_address(selected, streets_lower, streets_original)
                    out = []
                    for attr in ['Street', 'House', 'Building']:
                        if attr in selected:
                            word, score = selected[attr]
                            out.append(word)
                    addr_str = '-'.join(out)
                    print(f"[{addr}] Адрес: {addr_str if addr_str else 'не найден'}")
                    # добавить сюда логику важные места, медобъекты с номерами и др.
                    # например, find_canonical_object и пр.

                    audio_buffer = audio_buffer[BUFFER_SIZE - OVERLAP_SIZE:]
    except Exception as e:
        print(f"[{addr}] Ошибка: {e}")
    finally:
        conn.close()
        print(f"[{addr}] Завершение сессии")

def main():
    load_dotenv()
    streets_original, streets_lower = load_streets_dict("minsk_unique_streets2.txt")
    obj_mapping, obj_variants, obj_canons = load_object_canon_mapping("important_objects_mapping.txt")
    print(f"[SERVER] Слушаем на {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(32)
        while True:
            conn, addr = server_sock.accept()
            proc = mp.Process(
                target=handle_client_proc,
                args=(conn, addr, streets_original, streets_lower, obj_mapping, obj_variants),
                daemon=True
            )
            proc.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
