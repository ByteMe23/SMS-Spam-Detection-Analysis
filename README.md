# Spam Detection in SMS Messages 🔍

## Table of Contents
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
  - [Preprocessing](#preprocessing)  
  - [Feature Extraction](#feature-extraction)  
  - [Model--Training](#model--training)  
- [Results & Evaluation](#results--evaluation)  
- [Challenges & Learnings](#challenges--learnings)  
- [Future Work](#future-work)  
- [Usage](#usage)  
- [References](#references)  

---

## Overview
This project builds a machine learning system that classifies SMS messages as **spam** or **ham** (non-spam).  
It applies natural language processing (NLP) and classical ML algorithms to tackle a real-world text classification problem.

---

## Dataset
- Contains **5,572 English SMS messages**, labeled as `spam` or `ham`.  
- Data is divided into training and testing sets for reliable evaluation.  

---

## Methodology

### Preprocessing
Steps applied to clean and normalize SMS text:
1. Remove punctuation, numbers, and symbols  
2. Convert all text to lowercase  
3. Tokenize into words  
4. Remove stop-words  
5. Apply stemming/lemmatization  

### Feature Extraction
- Used **CountVectorizer** (scikit-learn) to convert text into a matrix of token counts.  
- Each SMS is represented as a vector of word frequencies.  

### Model--Training
- **Multinomial Naive Bayes** classifier chosen for its effectiveness in text classification.  
- Model trained on vectorized features and labels from the training set.  

---

## Results & Evaluation
- Achieved **98.83% accuracy** on the test dataset.  
- Confusion Matrix:

```sh
[[961 5]
[ 8 141]]
```

- **True Negatives (ham correct):** 961  
- **False Positives (ham → spam):** 5  
- **False Negatives (spam → ham):** 8  
- **True Positives (spam correct):** 141  

 <img width="580" height="479" alt="output" src="https://github.com/user-attachments/assets/a7a41a96-ec4d-4256-95f5-68fd6bb99a99" />


---

## Challenges & Learnings
- Class imbalance between spam and ham messages.  
- Importance of preprocessing and feature engineering.  
- Simpler models like Naive Bayes can perform surprisingly well on text classification tasks.  

---

## Future Work
- Hyperparameter tuning and cross-validation.  
- Try **TF-IDF**, **Word2Vec**, or **GloVe** embeddings.  
- Experiment with SVM, Random Forest, or deep learning (RNN, BERT).  
- Deploy in real-time as an SMS spam filter app.  
- Expand dataset with multilingual or diverse messages.  

---

## References 
- Scikit-learn Documentation
- Python's Natural Language Toolkit (NLTK)__ 
