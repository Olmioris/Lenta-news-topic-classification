import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
from preprocess import preprocess_text

def evaluate_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df["text_clean"] = df["text"].apply(preprocess_text)
    X = df["text_clean"].values
    y = df["topic"].values

    model = joblib.load("reports/best_model.pkl")
    y_pred = model.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Macro F1:", f1_score(y, y_pred, average="macro"))
    print(classification_report(y, y_pred))

    errors_mask = y != y_pred
    y_true_errors = y[errors_mask]
    y_pred_errors = y_pred[errors_mask]
    print(Counter(zip(y_true_errors, y_pred_errors)).most_common(10))

if __name__ == "__main__":
    evaluate_model("lenta-ru-news.csv")
