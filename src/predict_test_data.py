import os

from data_loader import load_test_data
from model import load_pipeline
from preprocessor import clean_text


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(project_root, "outputs", "model.pkl")
    test_path = os.path.join(project_root, "Data", "test_data.txt")
    prediction_path = os.path.join(project_root, "outputs", "predictions.txt") #save predictions to outputs/predictions.txt

    pipeline = load_pipeline(pipeline_path)
    test_texts = load_test_data(test_path)
    cleaned_texts = [clean_text(text) for text in test_texts]

    predictions = pipeline.predict(cleaned_texts)

    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    with open(prediction_path, "w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(f"{prediction}\n")

    print(f"Predicted {len(predictions)} test examples.")
    print(f"Saved predictions to: {prediction_path}")


if __name__ == "__main__":
    main()
