from src.preprocessing import preprocess_text
from src.tokenization import word_tokenize_text, sent_tokenize_text
from src.embedding import get_bert_embeddings
from src.modeling import fine_tune_bert
from src.utils import save_model, load_model

# Example usage
if __name__ == "__main__":
    # Fine-tune BERT model on AG News dataset
    model, tokenizer = fine_tune_bert('ag_news')

    # Save the model and tokenizer
    save_model(model, tokenizer, './models/ag_news_bert')

    # Load the model and tokenizer
    loaded_model, loaded_tokenizer = load_model('./models/ag_news_bert')

    # Test loading
    sample_text = "The stock market crashed yesterday."
    inputs = loaded_tokenizer(sample_text, return_tensors='pt')
    outputs = loaded_model(**inputs)
    print(outputs)
