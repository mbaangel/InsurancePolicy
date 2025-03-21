from nltk.corpus import stopwords
import nltk


def get_nltk_stop_words():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))
    return stop_words

def remove_stopwords(text, stop_words):
    # Dividir el texto en palabras
    words = text.split()
    # Filtrar las stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Unir las palabras filtradas en un solo texto
    return " ".join(filtered_words)

def filter_stopwords(stop_words,docs_default):
    text = " ".join(doc["content"] for doc in docs_default)
    # Filtrar las stopwords
    filtered_text = remove_stopwords(text, stop_words)
    return filtered_text

