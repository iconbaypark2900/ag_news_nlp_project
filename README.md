# AG News NLP Project

## Overview
This project demonstrates text preprocessing, tokenization, embedding techniques, and model fine-tuning using the AG News dataset.

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
