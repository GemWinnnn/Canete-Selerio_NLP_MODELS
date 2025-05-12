import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def prepare_sentiment_data(texts, labels):
    """Prepare data for sentiment classification"""
    # Traditional ML approach
    vectorizer = TfidfVectorizer(max_features=5000)
    X_trad = vectorizer.fit_transform(texts)
    
    # Neural Network approach
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    X_nn = tokenizer.texts_to_sequences(texts)
    X_nn = pad_sequences(X_nn, maxlen=100)
    
    return X_trad, X_nn, vectorizer, tokenizer

def create_traditional_sentiment_model():
    """Create traditional ML model for sentiment classification"""
    return MultinomialNB()

def create_nn_sentiment_model(vocab_size):
    """Create neural network model for sentiment classification"""
    model = Sequential([
        Embedding(vocab_size, 100, input_length=100),
        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_models(model_name, y_true, y_pred):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    return accuracy

def plot_results(results):
    """Plot comparison of model performances"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Sentiment Classification Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_classification_comparison.png')
    plt.close()

def main():
    # Load and prepare data (using IMDB dataset as an example)
    print("Loading and preparing data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    
    # Convert sequences back to text for demonstration
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    texts = [decode_review(text) for text in x_train[:1000]]  # Using subset for demonstration
    labels = y_train[:1000]
    
    # Sentiment Classification
    print("\nTraining Sentiment Classification Models...")
    X_trad, X_nn, vectorizer, tokenizer = prepare_sentiment_data(texts, labels)
    X_trad_train, X_trad_test, y_train, y_test = train_test_split(
        X_trad, labels, test_size=0.2, random_state=42
    )
    X_nn_train, X_nn_test, _, _ = train_test_split(
        X_nn, labels, test_size=0.2, random_state=42
    )
    
    # Train traditional model
    trad_sent_model = create_traditional_sentiment_model()
    trad_sent_model.fit(X_trad_train, y_train)
    trad_pred = trad_sent_model.predict(X_trad_test)
    
    # Train neural network model
    nn_sent_model = create_nn_sentiment_model(len(tokenizer.word_index) + 1)
    nn_sent_model.fit(X_nn_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    nn_pred = (nn_sent_model.predict(X_nn_test) > 0.5).astype(int)
    
    # Evaluate models
    results = {}
    results['Traditional Sentiment'] = evaluate_models('Traditional Sentiment', y_test, trad_pred)
    results['Neural Network Sentiment'] = evaluate_models('Neural Network Sentiment', y_test, nn_pred)
    
    # Plot results
    plot_results(results)
    print("\nResults have been plotted and saved as 'sentiment_classification_comparison.png'")

if __name__ == "__main__":
    main() 