import os
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def save_model(model, tokenizer, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved to {path}")

def load_model(path):
    model = DistilBertForSequenceClassification.from_pretrained(path)
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    return model, tokenizer
