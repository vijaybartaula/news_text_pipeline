# News Text Preprocessing Pipeline

A comprehensive Python pipeline for preprocessing news text data with advanced NLP techniques including tokenization, stemming, lemmatization, and machine learning model training.

## Features

- **Robust Text Cleaning**: URL removal, email filtering, special character handling
- **Advanced Tokenization**: NLTK-powered tokenization with stopword removal
- **Multiple Stemming Algorithms**: Porter, Lancaster, and Snowball stemmers
- **POS-Aware Lemmatization**: Context-aware word normalization using part-of-speech tagging
- **Automated ML Pipeline**: Built-in TF-IDF vectorization and logistic regression classification
- **Error Handling**: Comprehensive error handling and fallback mechanisms
- **Flexible Data Loading**: Automatic column detection and CSV processing

## Requirements

### Dependencies
```bash
pip install nltk pandas scikit-learn
```

### Required NLTK Data
The pipeline automatically downloads the following NLTK datasets:
- `punkt_tab` - Sentence tokenization
- `stopwords` - English stopwords
- `wordnet` - WordNet lexical database
- `averaged_perceptron_tagger` - POS tagging
- `averaged_perceptron_tagger_eng` - English POS tagging

## 🛠Installation

1. **Clone or download the pipeline script**
2. **Install dependencies**:
   ```bash
   pip install nltk pandas scikit-learn
   ```
3. **Run the pipeline** - NLTK data will be downloaded automatically on first run

## Input Data Format

The pipeline expects a CSV file named `news.csv` with at least one text column. Supported column names:
- `text`
- `content` 
- `article`
- `news_text`
- `body`

Optional label columns for ML training:
- `label`
- `category`
- `class`
- `target`
- `sentiment`

### Example CSV Structure
```csv
text,category
"Breaking news: Stock market reaches new highs...",business
"Scientists discover new species in Amazon rainforest...",science
```

## Usage

### Basic Usage
```python
# Simply run the main function
main()
```

### Individual Functions
```python
# Load data
df = load_data('your_file.csv')

# Clean individual text
clean = clean_text("Your raw text here!")

# Full preprocessing
processed = full_preprocess("Your text here", method='lemmatize')

# Train ML model
model, vectorizer = train_model(df, 'text_column', 'label_column')
```

### Preprocessing Options
```python
# Use lemmatization (default - recommended)
processed_lemma = full_preprocess(text, method='lemmatize')

# Use stemming
processed_stem = full_preprocess(text, method='stem')
```

## Pipeline Stages

### Stage 1: Data Loading
- CSV file detection and loading
- Column identification and validation
- Error handling for missing files

### Stage 2: Text Cleaning
- Lowercase conversion
- URL and email removal
- Special character filtering
- Whitespace normalization

### Stage 3: Tokenization
- NLTK word tokenization
- Stopword removal (English)
- Single character filtering

### Stage 4: Text Normalization
Choose between:
- **Lemmatization**: Context-aware word reduction using POS tags
- **Stemming**: Rule-based word reduction (Porter/Lancaster/Snowball)

### Stage 5: Machine Learning (Optional)
- TF-IDF vectorization (max 5000 features)
- Logistic regression classification
- Train/test split (80/20)
- Performance evaluation with classification report

## Output

### Processed Data
- Original CSV with added `processed_text` column
- Saved as `processed_news.csv`
- Preprocessing statistics and performance metrics

### Console Output
```
Starting News Text Preprocessing Pipeline
==================================================

Step 14a: Loading data...
  Successfully loaded news.csv
  Dataset shape: 1000 rows, 3 columns

Step 14d: Processing 1,000 texts...
  Average original length: 245.3 characters
  Average processed length: 156.7 characters
  Reduction: 36.1%

Step 14i: Training machine learning model...
  Model Accuracy: 0.847
  Classification Report:
              precision    recall  f1-score   support
     business       0.85      0.82      0.83        50
      science       0.84      0.87      0.86        50
```

## Configuration

### Stemming Algorithms
```python
# Available options in apply_stemming()
'porter'     # Porter Stemmer - most common
'lancaster'  # Lancaster Stemmer - most aggressive  
'snowball'   # Snowball Stemmer - default, balanced
```

### TF-IDF Parameters
```python
# Configurable in train_model()
max_features=5000  # Maximum vocabulary size
min_df=2          # Minimum document frequency
max_df=0.8        # Maximum document frequency (80%)
```

## Troubleshooting

### Common Issues

**FileNotFoundError: 'news.csv'**
- Ensure your CSV file is in the same directory as the script
- Check the filename matches exactly (case-sensitive)

**NLTK Download Errors**
- Check internet connection
- Run manually: `nltk.download('punkt')`

**Memory Issues with Large Datasets**
- Reduce `max_features` in TF-IDF vectorizer
- Process data in chunks

**No Text Column Found**
- Rename your text column to one of: `text`, `content`, `article`, `news_text`, `body`
- Or modify the `possible_text_cols` list in the code

## Performance

### Typical Processing Times
- **Small Dataset** (1K articles): ~30 seconds
- **Medium Dataset** (10K articles): ~3-5 minutes  
- **Large Dataset** (100K articles): ~30-45 minutes

### Memory Usage
- Approximately 100MB RAM per 10,000 articles
- TF-IDF vectorization is the most memory-intensive step

## Advanced Usage

### Custom Stopwords
```python
# Add domain-specific stopwords
custom_stops = get_stopwords()
custom_stops.update(['reuters', 'associated', 'press'])
```

### Batch Processing
```python
# Process multiple files
files = ['news1.csv', 'news2.csv', 'news3.csv']
for file in files:
    df = load_data(file)
    # Process each file...
```

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional preprocessing techniques
- More ML algorithms
- Performance optimizations
- Better visualization of results

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console output for specific error messages
3. Ensure all dependencies are properly installed

---

**Version**: 1.0  
**Last Updated**: 31 May 2025  
**Compatibility**: Python 3.7+
