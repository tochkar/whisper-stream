from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

def extract_addresses_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    addresses = []
    for span in doc.spans:
        if span.type == 'LOC':    # 'LOC' — локации: улицы, города, площади
            span.normalize(morph_vocab)
            addresses.append(span.text)
    return addresses

# Пример:
input_text = "Заберите меня с улицы Ленина, дом 10, отвезите на проспект Победителей, 5"
print(extract_addresses_natasha(input_text))  # выведет список адресов
