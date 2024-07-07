import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from src.tokenization import tokenize_with_distilbert  # Import the function from tokenization.py

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def lemmatize_text_wordnet(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def lemmatize_text_spacy(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc]
    return ' '.join(words)