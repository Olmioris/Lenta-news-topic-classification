import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import preprocess_text

RANDOM_STATE = 42

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["text", "topic"]].dropna()
    df["topic"] = df["topic"].astype("category")
    return df

def prepare_data(df: pd.DataFrame):
    # фильтрация редких классов
    topic_counts = df["topic"].value_counts()
    valid_topics = topic_counts[topic_counts >= 5].index
    df = df[df["topic"].isin(valid_topics)].reset_index(drop=True)

    # предобработка текста
    df["text_clean"] = df["text"].apply(preprocess_text)

    # удаление пустых строк
    df = df[df["text_clean"].str.strip() != ""]
    return df

def split_data(df: pd.DataFrame):
    X = df["text_clean"].values
    y = df["topic"].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_logreg_pipeline(vectorizer_cls):
    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        [
            ("vect", vectorizer_cls()),
            ("clf", log_reg),
        ]
    )

    param_grid = {
        "vect__max_features": [20_000, 50_000],
        "vect__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1_macro",
        verbose=1,
    )

    return grid

def main():
    df = load_data("lenta-ru-news.csv")
    df = prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # бейзлайн
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)

    # CountVectorizer
    grid_count = build_logreg_pipeline(CountVectorizer)
    grid_count.fit(X_train, y_train)

    # TfidfVectorizer
    grid_tfidf = build_logreg_pipeline(TfidfVectorizer)
    grid_tfidf.fit(X_train, y_train)

    # выбор лучшей модели
    if grid_tfidf.best_score_ >= grid_count.best_score_:
        best_grid = grid_tfidf
    else:
        best_grid = grid_count

    best_model = best_grid.best_estimator_
    best_model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    
if __name__ == "__main__":
    main()
