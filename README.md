# 📰 Fake News Detection System

A comprehensive machine learning project that detects fake news using Natural Language Processing (NLP) and various ML algorithms, featuring both classical ML models and deep learning approaches with a user-friendly Streamlit web interface.

## 🎯 Project Overview

This system analyzes news articles and predicts whether they are **Real** or **Fake** using advanced NLP techniques. The project includes exploratory data analysis, multiple ML model implementations, and a deployable web application.

### Key Features
- **Binary Classification**: Real (1) vs Fake (0) news detection
- **Multiple ML Approaches**: Classical ML and Deep Learning models
- **Interactive Web App**: User-friendly Streamlit interface
- **Comprehensive EDA**: Detailed data analysis and visualization
- **High Accuracy**: Optimized models with robust performance

## 📊 Dataset

- **Size**: ~45,000 news articles
- **Distribution**: ~22k Real + ~22k Fake articles
- **Sources**: 
  - `True.csv` - Real news articles
  - `Fake.csv` - Fake news articles
- **Features**: Title, Text content, Subject, Date

## 🏗️ Project Structure

```
fake-news-detection/
├── notebooks/
│   ├── 01_EDA_Analysis.ipynb          # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb        # Model training and evaluation
├── app/
│   ├── streamlit_app.py               # Web application
│   ├── model.pkl                      # Trained model
│   └── vectorizer.pkl                 # Text vectorizer
├── data/
│   ├── True.csv                       # Real news dataset
│   └── Fake.csv                       # Fake news dataset
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Class Distribution Analysis**: Balanced dataset with equal real/fake articles
- **Text Length Analysis**: Real news articles are generally longer
- **Word Cloud Generation**: Visual representation of common terms
- **Sentiment Analysis**: Fake news tends to be more subjective and emotional

### 2. Text Preprocessing
- Remove non-alphabetic characters
- Convert to lowercase
- Remove stopwords
- Apply stemming (Porter Stemmer)
- Combine title and text content

### 3. Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **N-gram Analysis**: Unigrams and bigrams
- **Sentiment Features**: Polarity and subjectivity scores
- **Text Statistics**: Length, word count, etc.

### 4. Model Implementation

#### Classical ML Models
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble method with 100 estimators
- **TF-IDF Vectorization**: Max 10,000 features with 1-2 n-grams

#### Deep Learning Model
- **LSTM Architecture**: Sequential model with embedding layer
- **Embedding Dimension**: 50-dimensional word embeddings
- **Architecture**: Embedding → LSTM(50) → Dense(32) → Dense(1)
- **Optimization**: Adam optimizer with binary crossentropy loss

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
```

### 4. Prepare Dataset
- Download `True.csv` and `Fake.csv` files
- Place them in the `data/` directory

## 💻 Usage

### Training the Model
```python
from fake_news_detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Train the complete pipeline
best_model, accuracy = detector.run_complete_pipeline()

# Save the trained model
detector.save_best_model()
```

### Using the Web App
```bash
# Navigate to the app directory
cd app/

# Run Streamlit app
streamlit run streamlit_app.py
```

### Making Predictions
```python
# Load trained model
detector.load_model()

# Make prediction
sample_text = "Your news article text here..."
result, confidence = detector.predict(sample_text)
print(f"Prediction: {result} (Confidence: {confidence:.2f})")
```
