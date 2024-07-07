from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_dataset

def preprocess_data(dataset, tokenizer, max_length):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_dataset

def fine_tune_bert(dataset_name, max_length=128, epochs=3):
    # Load dataset
    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # Preprocess dataset
    train_dataset = preprocess_data(train_dataset, tokenizer, max_length)
    test_dataset = preprocess_data(test_dataset, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
    output_dir='../models/results',  # Save results in the models directory
    num_train_epochs=epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../models/logs',    # Save logs in the models directory
    eval_strategy='epoch'            # Updated parameter name
)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train model
    trainer.train()

    return model, tokenizer

# Example usage
if __name__ == "__main__":
    fine_tune_bert('ag_news')
