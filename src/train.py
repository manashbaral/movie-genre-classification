import os

from data_loader import load_train_data
from preprocessor import clean_documents
from model import save_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


def build_tfidf_logistic_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    stop_words="english",
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=10000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_root, "Data", "train_data.txt")
    output_path = os.path.join(project_root, "outputs", "model.pkl") #save model to outputs/model.pkl

    X, y = load_train_data(train_path)
    X = clean_documents(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    pipeline = build_tfidf_logistic_pipeline()

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [8000, 10000, 12000],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.85, 0.9, 0.95],
        "clf__C": [0.5, 1.0, 2.0, 5.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    print("Best TF-IDF + Logistic Regression params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    y_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\nTest accuracy:", test_accuracy)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_pipeline(best_pipeline, output_path)
    print("Saved best TF-IDF + Logistic Regression pipeline to:", output_path)


if __name__ == "__main__":
    main()
