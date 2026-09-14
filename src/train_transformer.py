"""Fine-tune DistilBERT for support intent classification."""

import inspect
import json
import time
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import settings
from src.data_loader import prepare_data
from src.evaluate import classification_metrics, evaluate_predictions, save_error_analysis, update_model_comparison
from src.utils import ensure_directories, save_json, seed_everything


def choose_max_length(tokenizer, texts: list[str]) -> Tuple[int, Dict[str, float]]:
    """Choose length from the training set only, avoiding a blind 512-token default."""
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    percentile_length = int(np.ceil(np.quantile(lengths, settings.max_length_percentile)))
    chosen = max(32, min(percentile_length, settings.max_length_cap))
    return chosen, {
        "training_token_length_mean": float(np.mean(lengths)),
        "training_token_length_p95": float(np.quantile(lengths, 0.95)),
        "training_token_length_max": int(np.max(lengths)),
        "chosen_max_length": chosen,
    }


def _training_arguments() -> TrainingArguments:
    kwargs = {
        "output_dir": str(settings.models_dir / "checkpoints"),
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.train_batch_size,
        "per_device_eval_batch_size": settings.eval_batch_size,
        "num_train_epochs": settings.epochs,
        "weight_decay": settings.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": settings.random_seed,
        "data_seed": settings.random_seed,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
    }
    # Transformers renamed this argument; support both maintained APIs.
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def train() -> dict:
    ensure_directories()
    seed_everything()
    splits = prepare_data()
    labels = sorted(splits["train"][settings.label_column].unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(settings.transformer_model_name)
    max_length, length_stats = choose_max_length(
        tokenizer, splits["train"][settings.text_column].tolist()
    )

    def tokenize(batch):
        encoded = tokenizer(batch[settings.text_column], truncation=True, max_length=max_length)
        encoded["labels"] = [label2id[label] for label in batch[settings.label_column]]
        return encoded

    hf_splits = {}
    for name, frame in splits.items():
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        hf_splits[name] = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {name}",
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.transformer_model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    model.config.support_ticket_max_length = max_length

    def compute_metrics(eval_prediction):
        logits, label_ids = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return classification_metrics(label_ids, predictions)

    trainer_kwargs = {
        "model": model,
        "args": _training_arguments(),
        "train_dataset": hf_splits["train"],
        "eval_dataset": hf_splits["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=settings.early_stopping_patience)],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    started = time.perf_counter()
    train_result = trainer.train()
    training_time = time.perf_counter() - started
    trainer.save_model(settings.transformer_output_dir)
    tokenizer.save_pretrained(settings.transformer_output_dir)

    started = time.perf_counter()
    prediction_output = trainer.predict(hf_splits["test"])
    inference_time = time.perf_counter() - started
    prediction_ids = np.argmax(prediction_output.predictions, axis=-1)
    probabilities = softmax(prediction_output.predictions, axis=-1)
    confidences = probabilities.max(axis=-1)
    predicted_labels = [id2label[int(index)] for index in prediction_ids]
    actual_labels = splits["test"][settings.label_column].tolist()

    metrics = evaluate_predictions(
        actual_labels,
        predicted_labels,
        "distilbert",
        labels=labels,
        extra_metrics={
            "training_time_seconds": training_time,
            "test_inference_time_seconds": inference_time,
            "milliseconds_per_ticket": inference_time * 1000 / len(actual_labels),
            "test_examples": len(actual_labels),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            **length_stats,
        },
    )
    save_error_analysis(
        splits["test"][settings.text_column],
        actual_labels,
        predicted_labels,
        confidences,
        "distilbert",
    )
    save_json(
        {
            "labels": labels,
            "label2id": label2id,
            "id2label": id2label,
            "max_length": max_length,
            "train_metrics": train_result.metrics,
        },
        settings.transformer_output_dir / "training_summary.json",
    )
    update_model_comparison()
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
