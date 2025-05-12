import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random

# Download required NLTK data
nltk.download('punkt')

class MarkovChain:
    def __init__(self, n=2):
        self.n = n
        self.model = defaultdict(list)
        
    def train(self, texts):
        for text in texts:
            words = text.split()
            for i in range(len(words) - self.n):
                key = tuple(words[i:i + self.n])
                self.model[key].append(words[i + self.n])
    
    def predict_next(self, sequence):
        key = tuple(sequence[-self.n:])
        if key in self.model:
            return random.choice(self.model[key])
        return None

def prepare_text_prediction_data(texts, sequence_length=10):
    """Prepare data for text prediction"""
    # For Markov Chain
    markov_data = texts
    
    # For Neural Network
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    X, y = [], []
    for seq in sequences:
        for i in range(len(seq) - sequence_length):
            X.append(seq[i:i + sequence_length])
            y.append(seq[i + sequence_length])
    
    return np.array(X), np.array(y), tokenizer, markov_data

def create_traditional_text_prediction_model():
    """Create traditional ML model for text prediction using Markov Chain"""
    return MarkovChain(n=2)

def create_nn_text_prediction_model(vocab_size):
    """Create neural network model for text prediction"""
    model = Sequential([
        Embedding(vocab_size, 100, input_length=10),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_markov_model(model, test_texts, tokenizer):
    """Evaluate Markov Chain model"""
    correct = 0
    total = 0
    
    for text in test_texts:
        words = text.split()
        for i in range(len(words) - 2):
            sequence = words[i:i+2]
            actual_next = words[i+2]
            predicted_next = model.predict_next(sequence)
            
            if predicted_next == actual_next:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0

def plot_results(results):
    """Plot comparison of model performances"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Text Prediction Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('text_prediction_comparison.png')
    plt.close()

def main():
    # Load and prepare data (using IMDB dataset as an example)
    print("Loading and preparing data...")
    (x_train, _), (_, _) = tf.keras.datasets.imdb.load_data(num_words=10000)
    
    # Convert sequences back to text for demonstration
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    texts = [decode_review(text) for text in x_train[:1000]]  # Using subset for demonstration
    
    # Split data for both models
    train_texts, test_texts = train_test_split(texts, test_size=0.2, random_state=42)
    
    # Text Prediction
    print("\nTraining Text Prediction Models...")
    X_text, y_text, tokenizer, markov_data = prepare_text_prediction_data(train_texts)
    X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
        X_text, y_text, test_size=0.2, random_state=42
    )
    
    # Train traditional model (Markov Chain)
    trad_text_model = create_traditional_text_prediction_model()
    trad_text_model.train(markov_data)
    
    # Train neural network model
    nn_text_model = create_nn_text_prediction_model(len(tokenizer.word_index) + 1)
    nn_text_model.fit(X_text_train, y_text_train, epochs=5, batch_size=32, validation_split=0.2)
    
    # Evaluate models
    results = {}
    results['Markov Chain Text Prediction'] = evaluate_markov_model(trad_text_model, test_texts, tokenizer)
    results['Neural Network Text Prediction'] = nn_text_model.evaluate(X_text_test, y_text_test)[1]
    
    # Plot results
    plot_results(results)
    print("\nResults have been plotted and saved as 'text_prediction_comparison.png'")

if __name__ == "__main__":
    main() 