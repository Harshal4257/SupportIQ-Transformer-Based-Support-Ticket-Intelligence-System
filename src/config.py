"""Central configuration for data, training, evaluation, and inference."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    metrics_dir: Path = PROJECT_ROOT / "reports" / "metrics"

    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    text_column: str = "instruction"
    label_column: str = "intent"

    transformer_model_name: str = "distilbert-base-uncased"
    transformer_output_dir: Path = PROJECT_ROOT / "models" / "distilbert-support-classifier"
    baseline_model_path: Path = PROJECT_ROOT / "models" / "tfidf-logistic-regression.joblib"

    random_seed: int = 42
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15

    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    max_length_cap: int = 256
    max_length_percentile: float = 0.95
    confidence_threshold: float = 0.60


settings = Settings()
