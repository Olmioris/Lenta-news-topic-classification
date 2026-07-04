import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Test macro F1:", f1_score(y_test, y_pred, average="macro"))
    print()
    print(classification_report(y_test, y_pred))
