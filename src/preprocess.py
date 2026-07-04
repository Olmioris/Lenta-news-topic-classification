import re
from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()

def preprocess_text(text: str) -> str:
    # нижний регистр
    text = text.lower()

    # оставляем только буквы и пробелы
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # лемматизация
    tokens = text.split()
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]

    return " ".join(lemmas)
