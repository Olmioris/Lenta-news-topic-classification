import pandas as pd
import numpy as np
import joblib
import json
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from preprocess import preprocess_text

RANDOM_STATE = 42

def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[["text", "topic"]].dropna()
    df["topic"] = df["topic"].astype("category")
    df["text_clean"] = df["text"].apply(preprocess_text)
    df = df[df["text_clean"].str.strip() != ""]

    X = df["text_clean"].values
    y = df["topic"].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
    )

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    y_val_dummy = dummy.predict(X_val)
    print("Dummy accuracy:", accuracy_score(y_val, y_val_dummy))
    print("Dummy macro F1:", f1_score(y_val, y_val_dummy, average="macro"))

    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)

    pipe_count = Pipeline([("vect", CountVectorizer()), ("clf", log_reg)])
    param_grid = {
        "vect__max_features": [20_000, 50_000],
        "vect__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
    }

    grid = GridSearchCV(pipe_count, param_grid, cv=3, n_jobs=-1, scoring="f1_macro", verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, "reports/best_model.pkl")

    with open("reports/best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4, ensure_ascii=False)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

if __name__ == "__main__":
    train_model("lenta-ru-news.csv")
