# Topic Classification of Russian News (Lenta.ru, 100k documents).

This project implements a full, reproducible NLP pipeline for classifying Russian news articles by topic using classical machine learning methods. It includes data loading, preprocessing, vectorization, model training, hyperparameter tuning, evaluation, and error analysis.

## 1. Project Overview.

This repository contains an end‑to‑end text classification pipeline built on a 100,000‑document subset of the Lenta.ru news corpus. The goal is to evaluate how far classical NLP methods (CountVectorizer, TF‑IDF, Logistic Regression) can go on a large real‑world Russian dataset.

The pipeline includes:

 - efficient preprocessing of Russian text (normalization + lemmatization);
 - stratified splitting (60/20/20);
 - baseline evaluation (DummyClassifier);
 - two vectorization strategies (Count vs TF‑IDF);
 - hyperparameter tuning with GridSearchCV;
 - final evaluation on a held‑out test set;
 - error analysis and insights.

## 2. Dataset.

 - source: Lenta.ru news dataset (Corus / Kaggle);
 - size used: 100,000 documents;
 - fields: title, text, topic;
 - rare classes removed to ensure valid stratification;
 - example from the dataset: "Each record in the dataset represents a single news article and contains three fields: title, text, and topic".

## 3. Preprocessing Pipeline.

The preprocessing pipeline includes:

 - lowercasing;
 - removing non‑alphabetic characters;
 - normalizing whitespace;
 - lemmatization using pymorphy3;
 - removing empty texts;
 - filtering out classes with < 2 samples;
 - creating a clean text column: text_clean.

Why lemmatization?

Lemmatization reduces feature sparsity and improves classification quality, which is particularly important for morphologically rich languages like Russian.

## 4. Train/Validation/Test Split.

 - train: 60%;
 - validation: 20%;
 - test: 20%;
 - stratified by topic;
 - random_state = 42 for reproducibility.

## 5. Baseline.

 - DummyClassifier(strategy="most_frequent");
 - validation accuracy: 0.3849;
 - Macro F1: 0.061;
   
This establishes a minimal baseline.

## 6. Models & Vectorization.

Two pipelines were evaluated:

 - CountVectorizer + LogisticRegression;
 - TfidfVectorizer + LogisticRegression.
   
Hyperparameters tuned.

Vectorizer Hyperparameters:

 - max_features: 20k, 50k;
 - ngram_range: (1,1), (1,2);

Logistic Regression Hyperparameters:

 - C: 0.5, 1.0, 2.0;

GridSearchCV Parameters:

 - GridSearchCV;
 - 3‑fold CV;
 - scoring: macro F1;
 - n_jobs = -1;

Best CountVectorizer result:

{'clf__C': 0.5, 'vect__max_features': 50000, 'vect__ngram_range': (1, 1)}

**CV macro F1: 0.6507**

Best TF‑IDF result:

**CV macro F1: 0.6465**

CountVectorizer performed slightly better.

#### What Could Be Tuned in a Production Pipeline?

If this project were extended into a production‑level system, additional hyperparameters could be explored:

Vectorizer Options:
 - min_df — ignore extremely rare words (noise reduction);
 - max_df — ignore overly frequent words (stop‑word‑like behavior);
 - stop_words — custom stop‑word lists;
 - sublinear_tf=True — logarithmic term frequency scaling;
 - tokenizer / preprocessor — custom tokenization logic;

Model Options:
 - class_weight="balanced" — improve recall for rare classes;
 - penalty="l1" or "l2" — different regularization types;
 - solver="liblinear" or "saga" — optimized solvers for sparse data;
 - max_iter — increase if convergence issues appear.

Pipeline Enhancements:
 - char‑level n‑grams — robustness to typos and short texts;
 - feature engineering — text length, number of digits, presence of named entities;
 - semantic embeddings — Word2Vec, FastText, or transformer embeddings;
 - oversampling / augmentation — improve rare class performance.

## 7. Final Evaluation (Test Set).

**Accuracy: 0.8583**

**Macro F1: 0.6607**

This shows:

 - high accuracy on frequent classes;
 - lower recall on rare classes;
 - typical behavior for imbalanced news datasets.

## 8. Error Analysis.

The most frequent misclassification patterns include:

 - World → Russiaж
 - Russia → World;
 - Economy → Russia;
 - Lifestyle → World

These confusions occur because several topics share overlapping vocabulary and similar contextual cues, making them harder to separate using classical bag‑of‑words models.

## 9. Possible Improvements.

 - class balancing (oversampling, class weights);
 - adding n‑grams, char‑grams;
 - linearSVC;
 - semantic features (Word2Vec, FastText);
 - targeted augmentation for rare classes.

## 10. How to Run.

```
### install dependencies
pip install -r requirements.txt

### train the best model (CountVectorizer or TF-IDF)
python src/train.py

### evaluate on the held-out test set
python src/evaluate.py
```
