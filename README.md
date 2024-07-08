# AG News NLP Project

This project focuses on fine-tuning a DistilBERT model for text classification on the AG News dataset. The pipeline includes data preprocessing, tokenization, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The AG News NLP Project aims to classify news articles into one of four categories: World, Sports, Business, and Sci/Tech. The project leverages the DistilBERT model, a smaller and faster version of BERT, to achieve high accuracy with reduced computational resources.

## Directory Structure

## Project Structure
- `data/`: Contains raw and processed data files.
  - `raw/`: Contains the raw AG News dataset files.
    - `ag_news/`: Directory for the AG News dataset.
      - `train.csv`: Raw training data.
      - `test.csv`: Raw test data.
  - `processed/`: Contains the preprocessed AG News dataset files.
    - `ag_news/`: Directory for the preprocessed AG News dataset.
      - `train_preprocessed.csv`: Preprocessed training data.
      - `test_preprocessed.csv`: Preprocessed test data.
- `models/`: Directory for saved models.
- `notebooks/`: Jupyter notebooks for experimentation.
- `scripts/`: Scripts for data processing and model training.
  - `data_preparation.py`: Script to prepare data.
- `src/`: Source code.
  - `preprocessing.py`: Text preprocessing functions.
  - `tokenization.py`: Tokenization functions.
  - `embedding.py`: Embedding functions.
  - `modeling.py`: Model training and evaluation.
  - `utils.py`: Utility functions.
- `.gitignore`: Git ignore file.
- `README.md`: Project documentation.
- `requirements.txt`: Required packages.

## Getting Started
1. Set up the virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. **Fine-tune the model and evaluate:**

    The `run_pipeline.py` script will also handle the fine-tuning of the DistilBERT model on different versions of the dataset (raw, stemmed, lemmatized with WordNet, and lemmatized with spaCy) and evaluate the performance.

## Data Preparation

The data preparation steps include:

- Downloading the AG News dataset.
- Preprocessing the text data (tokenization, stemming, lemmatization).
- Saving the processed data for future use.

## Model Training and Evaluation

The model training and evaluation steps include:

- Loading the pre-trained DistilBERT model and tokenizer.
- Fine-tuning the model on the training data.
- Evaluating the model on the test data using accuracy, precision, recall, and F1 score.
- Saving the fine-tuned model and tokenizer.

## Results

The evaluation results for each version of the dataset (raw, stemmed, lemmatized with WordNet, and lemmatized with spaCy) will be printed and saved.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.