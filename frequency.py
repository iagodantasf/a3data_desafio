import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re


nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')


def token_frequency(text: str, language: str = "english") -> Counter:
    text = text.lower()
    text = re.sub(r'[.,]', '', text)
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text)
    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]
    return Counter(filtered_tokens)
