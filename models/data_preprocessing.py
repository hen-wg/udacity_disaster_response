"""This module contains functions for data preprocessing."""
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])


def replace_all_urls(text: str) -> str:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    return text


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    test = text.strip()  # remove whitespaces
    return test


def tokenize(text: str) -> list:
    words = word_tokenize(text)
    return words


def remove_stopwords(text: list) -> list:
    stop_words = stopwords.words("english")
    words = [w for w in text if w not in stop_words]
    return words


def lemmatize(text: list) -> list:
    lemmatized_all = []
    for word in text:
        lemmatizer = WordNetLemmatizer()
        lemmatized = lemmatizer.lemmatize(word, pos='v')
        lemmatized_all.append(lemmatized)
    return lemmatized_all


def clean_and_tokenize(text: str) -> list:
    text = replace_all_urls(text)
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens
