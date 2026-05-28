"""Model utilities for TF-IDF + Logistic Regression."""

import joblib
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X, y, C=1.0, class_weight="balanced", max_iter=10000):
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        random_state=42,
    )
    model.fit(X, y)
    return model

def save_pipeline(pipeline, file_path):
    joblib.dump(pipeline, file_path)


def load_pipeline(file_path):
    return joblib.load(file_path)