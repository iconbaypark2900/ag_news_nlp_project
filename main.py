import os
from src.modeling import fine_tune_distilbert
from src.utils import save_model, load_model

def main():
    # Define the directory to save the model
    save_dir = './models/ag_news_distilbert'
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Fine-tune BERT model on AG News dataset
    model, tokenizer = fine_tune_distilbert('ag_news')

    # Save the model and tokenizer
    save_model(model, tokenizer, save_dir)

    # Load the model and tokenizer
    loaded_model, loaded_tokenizer = load_model(save_dir)

    # Test loading
    sample_text = "The stock market crashed yesterday."
    inputs = loaded_tokenizer(sample_text, return_tensors='pt')
    outputs = loaded_model(**inputs)
    print(outputs)

if __name__ == "__main__":
    main()
