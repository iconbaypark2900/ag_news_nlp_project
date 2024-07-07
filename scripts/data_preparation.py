from datasets import load_dataset
import pandas as pd
from src.preprocessing import preprocess_text, stem_text, lemmatize_text_wordnet, lemmatize_text_spacy

# Load the AG News dataset
ag_news = load_dataset('ag_news')

# Save the raw data
ag_news['train'].to_csv('data/raw/ag_news_r/train.csv', index=False)
ag_news['test'].to_csv('data/raw/ag_news_r/test.csv', index=False)

# Load raw data
train_data = pd.read_csv('data/raw/ag_news_r/train.csv')
test_data = pd.read_csv('data/raw/ag_news_r/test.csv')

# Preprocess the text data
train_data['text'] = train_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

# Apply stemming
train_data['text_stemmed'] = train_data['text'].apply(stem_text)
test_data['text_stemmed'] = test_data['text'].apply(stem_text)

# Apply WordNet lemmatization
train_data['text_lemmatized_wordnet'] = train_data['text'].apply(lemmatize_text_wordnet)
test_data['text_lemmatized_wordnet'] = test_data['text'].apply(lemmatize_text_wordnet)

# Apply spaCy lemmatization
train_data['text_lemmatized_spacy'] = train_data['text'].apply(lemmatize_text_spacy)
test_data['text_lemmatized_spacy'] = test_data['text'].apply(lemmatize_text_spacy)

# Save the processed data
train_data.to_csv('data/processed/ag_news_p/train_stemmed.csv', index=False)
test_data.to_csv('data/processed/ag_news_p/test_stemmed.csv', index=False)
train_data.to_csv('data/processed/ag_news_p/train_lemmatized_wordnet.csv', index=False)
test_data.to_csv('data/processed/ag_news_p/test_lemmatized_wordnet.csv', index=False)
train_data.to_csv('data/processed/ag_news_p/train_lemmatized_spacy.csv', index=False)
test_data.to_csv('data/processed/ag_news_p/test_lemmatized_spacy.csv', index=False)
