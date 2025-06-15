# Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A comprehensive machine learning system for detecting fake news articles using multiple natural language processing approaches. This project compares the effectiveness of traditional machine learning methods with modern deep learning techniques for binary classification of news articles as either "REAL" or "FAKE".

## Key Features

- **Multi-Model Approach**: Implements and compares three different classification methods
- **Advanced Text Preprocessing**: Comprehensive text cleaning, tokenization, lemmatization, and stopword removal
- **Feature Engineering**: Both traditional TF-IDF and modern Word2Vec embeddings
- **Deep Learning**: LSTM neural network with dropout regularization
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Production Ready**: Clean, modular code with proper error handling

## Performance Results

| Model | Accuracy | Precision (FAKE) | Recall (FAKE) | F1-Score (FAKE) |
|-------|----------|------------------|---------------|-----------------|
| **TF-IDF + Logistic Regression** | **92.94%** | **0.91** | **0.95** | **0.93** |
| Word2Vec + Logistic Regression | 89.37% | 0.89 | 0.90 | 0.89 |
| LSTM Neural Network | 84.00% | 0.87 | 0.80 | 0.83 |

## Quick Start

### Prerequisites

```bash
pip install nltk pandas numpy scikit-learn tensorflow gensim matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vijaybartaula/news_text_pipeline.git
cd news_text_pipeline
```

2. Download NLTK data:
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

### Usage

1. **Prepare your dataset**: Ensure your CSV file has columns named `title`, `text`, and `label`

2. **Use individual components**:
```python
from fake_news_detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Load and preprocess data
detector.load_data('your_dataset.csv')
detector.preprocess_data()

# Train models
detector.train_all_models()

# Make predictions
prediction = detector.predict("Your news article text here")
```

## System Architecture

### Data Pipeline
1. **Data Loading**: CSV file with news articles and labels
2. **Text Preprocessing**: 
   - Lowercasing and punctuation removal
   - Tokenization and stopword removal
   - Lemmatization with POS tagging
3. **Feature Engineering**:
   - TF-IDF vectorization (max 5000 features, 1-2 grams)
   - Word2Vec embeddings (100 dimensions)
   - LSTM tokenization and padding

### Models

#### 1. TF-IDF + Logistic Regression
- **Best performing model** with 92.94% accuracy
- Uses traditional bag-of-words approach with TF-IDF weighting
- Handles n-grams for better context understanding
- Fast training and inference

#### 2. Word2Vec + Logistic Regression
- Word embeddings capture semantic relationships
- Document vectors created by averaging word vectors
- 89.37% accuracy with good generalization

#### 3. LSTM Neural Network
- Deep learning approach with embedding layer
- Bidirectional LSTM for sequence processing
- Dropout regularization to prevent overfitting
- 84.00% accuracy (may improve with more data/tuning)

## Dataset Information

- **Total Articles**: 6,335 (after preprocessing: 6,298)
- **Features**: Article title, full text content, binary label
- **Classes**: REAL vs FAKE (balanced distribution)
- **Train/Test Split**: 80/20 with stratification

## Technical Details

### Text Preprocessing Pipeline
```python
def full_preprocess(text):
    text = clean_text(text)                    # Remove special chars, digits
    tokens = tokenize_and_remove_stopwords(text) # NLTK tokenization
    text = ' '.join(tokens)
    text = lemmatize_text(text)                # POS-aware lemmatization
    return text
```

### Model Hyperparameters

**LSTM Configuration:**
- Embedding dimension: 100
- LSTM units: 128 → 64 (stacked)
- Dropout: 0.5
- Dense layer: 64 units
- Max sequence length: 200
- Vocabulary size: 10,000

**TF-IDF Configuration:**
- Max features: 5,000
- N-gram range: (1, 2)
- Sublinear TF scaling

## Evaluation Metrics

The system provides comprehensive evaluation including:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Training History**: Loss and accuracy curves for LSTM
- **Feature Analysis**: Top TF-IDF features visualization

## Key Findings

1. **Traditional Methods Excel**: TF-IDF with logistic regression outperformed deep learning approaches
2. **Feature Engineering Matters**: Proper text preprocessing significantly impacts performance
3. **Balanced Performance**: High precision and recall for both classes
4. **Computational Efficiency**: Simpler models train faster with comparable results

## Future Improvements

- [ ] Implement BERT/transformer-based models
- [ ] Add cross-validation for robust evaluation
- [ ] Ensemble methods combining multiple models
- [ ] Real-time prediction API
- [ ] Extended feature engineering (sentiment, readability)
- [ ] Multi-language support

## References

- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12, pp. 2825-2830.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

---

**Note**: This system is designed for research and educational purposes. Always verify results with domain experts before deploying in production environments.

---

**Built with ❤️ for combating misinformation**
