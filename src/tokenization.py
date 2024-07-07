import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

def word_tokenize_text(text):
    return word_tokenize(text)

def sent_tokenize_text(text):
    return sent_tokenize(text)
