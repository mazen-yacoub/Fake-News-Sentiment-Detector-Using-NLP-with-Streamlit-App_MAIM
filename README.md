# 📰 Fake News Detection System

A comprehensive **machine learning project** that detects fake news using **Natural Language Processing (NLP)** and a mix of **classical ML algorithms** and **deep learning approaches**, featuring a **Streamlit web interface** for easy use.

---

## 🎯 Project Overview

This system analyzes news articles and predicts whether they are **Real** or **Fake** using advanced NLP techniques.  
The project includes **exploratory data analysis (EDA)**, multiple **ML model implementations**, and a **deployable web application**.

### ✅ Key Features
- **Binary Classification**: Real (1) vs Fake (0) detection  
- **Multiple ML Approaches**: Classical ML & Deep Learning  
- **Interactive Web App**: Streamlit-powered interface  
- **Comprehensive EDA**: Data analysis & visualization  
- **High Accuracy**: Optimized, robust models  

---

## 📊 Dataset

- **Size**: ~45,000 news articles  
- **Distribution**: ~22k Real + ~22k Fake articles  
- **Sources**:  
  - `True.csv` → Real news  
  - `Fake.csv` → Fake news  
- **Features**: Title, Text, Subject, Date  

---

## 🏗 Project Structure

```bash
fake-news-detection/
├── notebooks/
│   ├── exploration.ipynb         # Exploratory Data Analysis
│   └── modeling.ipynb            # Model training & evaluation
├── app/
│   ├── app.py                    # Streamlit Web App
│   ├── model.pkl                 # Trained ML model
│   └── vectorizer.pkl            # TF-IDF vectorizer
├── data/
│   ├── True.csv                  # Real news dataset
│   └── Fake.csv                  # Fake news dataset
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
