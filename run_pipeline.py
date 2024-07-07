import os
import pandas as pd
from datasets import load_dataset
from src.preprocessing import preprocess_text, stem_text, lemmatize_text_wordnet, lemmatize_text_spacy
from src.tokenization import tokenize_with_distilbert
from src.modeling import fine_tune_distilbert
from src.utils import save_model, load_model
from src.embedding import get_distilbert_embeddings

def data_exists():
    required_files = [
        'data/raw/ag_news_r/train.csv',
        'data/raw/ag_news_r/test.csv',
        'data/processed/ag_news_p/train_stemmed.csv',
        'data/processed/ag_news_p/test_stemmed.csv',
        'data/processed/ag_news_p/train_lemmatized_wordnet.csv',
        'data/processed/ag_news_p/test_lemmatized_wordnet.csv',
        'data/processed/ag_news_p/train_lemmatized_spacy.csv',
        'data/processed/ag_news_p/test_lemmatized_spacy.csv'
    ]
    return all(os.path.exists(file) for file in required_files)

def main():
    if data_exists():
        print("Data already exists. Skipping data preparation steps.")
    else:
        # Step 1: Data Preparation
        print("Loading AG News dataset...")
        ag_news = load_dataset('ag_news')

        # Save the raw data
        print("Saving raw data...")
        os.makedirs('data/raw/ag_news_r', exist_ok=True)
        ag_news['train'].to_csv('data/raw/ag_news_r/train.csv', index=False)
        ag_news['test'].to_csv('data/raw/ag_news_r/test.csv', index=False)

        # Load raw data
        print("Loading raw data...")
        train_data = pd.read_csv('data/raw/ag_news_r/train.csv')
        test_data = pd.read_csv('data/raw/ag_news_r/test.csv')

        # Preprocess the text data
        print("Preprocessing text data...")
        train_data['text'] = train_data['text'].apply(preprocess_text)
        test_data['text'] = test_data['text'].apply(preprocess_text)

        # Apply stemming
        print("Applying stemming...")
        train_data['text_stemmed'] = train_data['text'].apply(stem_text)
        test_data['text_stemmed'] = test_data['text'].apply(stem_text)

        # Apply WordNet lemmatization
        print("Applying WordNet lemmatization...")
        train_data['text_lemmatized_wordnet'] = train_data['text'].apply(lemmatize_text_wordnet)
        test_data['text_lemmatized_wordnet'] = test_data['text'].apply(lemmatize_text_wordnet)

        # Apply spaCy lemmatization
        print("Applying spaCy lemmatization...")
        train_data['text_lemmatized_spacy'] = train_data['text'].apply(lemmatize_text_spacy)
        test_data['text_lemmatized_spacy'] = test_data['text'].apply(lemmatize_text_spacy)

        # Save the processed data
        print("Saving processed data...")
        os.makedirs('data/processed/ag_news_p', exist_ok=True)
        train_data.to_csv('data/processed/ag_news_p/train_stemmed.csv', index=False)
        test_data.to_csv('data/processed/ag_news_p/test_stemmed.csv', index=False)
        train_data.to_csv('data/processed/ag_news_p/train_lemmatized_wordnet.csv', index=False)
        test_data.to_csv('data/processed/ag_news_p/test_lemmatized_wordnet.csv', index=False)
        train_data.to_csv('data/processed/ag_news_p/train_lemmatized_spacy.csv', index=False)
        test_data.to_csv('data/processed/ag_news_p/test_lemmatized_spacy.csv', index=False)

    # Step 2: Tokenization (already integrated in preprocessing)

    # Step 3: Modeling
    print("Fine-tuning DistilBERT model...")
    model, tokenizer = fine_tune_distilbert('ag_news')

    # Step 4: Save the model and tokenizer
    save_dir = './models/ag_news_distilbert'
    print(f"Saving model and tokenizer to {save_dir}...")
    save_model(model, tokenizer, save_dir)

    # Step 5: Load the model and tokenizer
    print(f"Loading model and tokenizer from {save_dir}...")
    loaded_model, loaded_tokenizer = load_model(save_dir)

    # Step 6: Test loading and generate embeddings
    sample_text = "The stock market crashed yesterday."
    print(f"Generating embeddings for sample text: '{sample_text}'")
    embeddings = get_distilbert_embeddings(sample_text)
    print("Embeddings generated:", embeddings)

if __name__ == "__main__":
    main()