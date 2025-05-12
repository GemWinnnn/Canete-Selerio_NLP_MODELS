# NLP Tasks: Text Prediction and Sentiment Classification

This project implements text prediction and sentiment classification using both traditional machine learning and neural network approaches. The implementation uses the IMDB dataset for demonstration purposes.

## Project Structure

The project is split into two separate files:
1. `text_prediction.py` - Implements text prediction models
2. `sentiment_classification.py` - Implements sentiment classification models

## Features

### Text Prediction (`text_prediction.py`)
- Traditional ML approach using Markov Chain (n-gram based prediction)
- Deep Learning approach using LSTM networks

### Sentiment Classification (`sentiment_classification.py`)
- Traditional ML approach using Naive Bayes
- Deep Learning approach using LSTM networks


## Results

Each script will output:
- Accuracy scores for each model
- Classification reports (for sentiment classification)
- A bar plot comparing model performances:
  - `text_prediction_comparison.png` for text prediction
  - `sentiment_classification_comparison.png` for sentiment classification

## Model Architecture

### Text Prediction
- Traditional: Markov Chain model (2-gram based prediction)
- Neural Network: LSTM-based architecture with multiple layers

### Sentiment Classification
- Traditional: Naive Bayes with TF-IDF features
- Neural Network: LSTM-based architecture with embedding layer

