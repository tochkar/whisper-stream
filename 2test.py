from natasha import AddrExtractor
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()
extractor = AddrExtractor(morph)

texts = [
    'Москва, проспект Мира, дом 51',
    'г. Самара, Пр-т Ленина, дом 7',
    'Санкт-Петербург, пр. Каменноостровский, 44',
    'г. Тула, пер. Красноармейский, дом 11',
    'Новосибирск, пл. Калинина, дом 1',
    'г. Москва, бульвар Дмитрия Донского, д. 13',
    'шоссе Энтузиастов, д. 65',
    'г. Томск, ул. Ленина, д. 22'
]
for t in texts:
    matches = extractor(t)
    for m in matches:
        print(t)
        print('  fact:', m.fact)
        print('  строка в тексте:', t[m.start:m.stop])
