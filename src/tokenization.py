import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import DistilBertTokenizer

nltk.download('punkt')

# Initialize DistilBERT tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def word_tokenize_text(text):
    return word_tokenize(text)

def sent_tokenize_text(text):
    return sent_tokenize(text)

def tokenize_with_distilbert(text):
    return distilbert_tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')