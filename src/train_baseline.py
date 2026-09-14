"""Train and evaluate a TF-IDF + Logistic Regression baseline."""

import json
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import settings
from src.data_loader import prepare_data
from src.evaluate import evaluate_predictions, save_error_analysis, update_model_comparison
from src.utils import ensure_directories, seed_everything


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    solver="lbfgs",
                    random_state=settings.random_seed,
                ),
            ),
        ]
    )


def train() -> dict:
    ensure_directories()
    seed_everything()
    splits = prepare_data()
    train_frame, test_frame = splits["train"], splits["test"]
    labels = sorted(train_frame[settings.label_column].unique())

    model = build_baseline()
    started = time.perf_counter()
    model.fit(train_frame[settings.text_column], train_frame[settings.label_column])
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    predictions = model.predict(test_frame[settings.text_column])
    probabilities = model.predict_proba(test_frame[settings.text_column])
    inference_time = time.perf_counter() - started
    confidences = np.max(probabilities, axis=1)

    joblib.dump(model, settings.baseline_model_path)
    metrics = evaluate_predictions(
        test_frame[settings.label_column],
        predictions,
        "baseline",
        labels=labels,
        extra_metrics={
            "training_time_seconds": training_time,
            "test_inference_time_seconds": inference_time,
            "milliseconds_per_ticket": inference_time * 1000 / len(test_frame),
            "test_examples": len(test_frame),
        },
    )
    save_error_analysis(
        test_frame[settings.text_column],
        test_frame[settings.label_column],
        predictions,
        confidences,
        "baseline",
    )
    update_model_comparison()
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
