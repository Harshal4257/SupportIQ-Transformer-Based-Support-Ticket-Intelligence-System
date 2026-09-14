"""Reproducible exploratory data analysis used by notebook 01."""

import json
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import settings
from src.data_loader import load_raw_dataframe, prepare_data
from src.utils import ensure_directories, save_json


def _save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(settings.figures_dir / filename, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def run_eda() -> Dict[str, object]:
    ensure_directories()
    raw = load_raw_dataframe()
    splits = prepare_data()
    clean = pd.concat(splits.values(), ignore_index=True)
    text = clean[settings.text_column]
    labels = clean[settings.label_column]
    word_lengths = text.str.split().str.len()
    character_lengths = text.str.len()
    counts = labels.value_counts()

    summary = {
        "raw_shape": list(raw.shape),
        "raw_columns": raw.columns.tolist(),
        "raw_missing_values": raw.isna().sum().to_dict(),
        "raw_duplicate_rows": int(raw.duplicated().sum()),
        "clean_shape": list(clean.shape),
        "number_of_intent_classes": int(labels.nunique()),
        "average_words": float(word_lengths.mean()),
        "median_words": float(word_lengths.median()),
        "average_characters": float(character_lengths.mean()),
        "shortest_ticket": text.loc[word_lengths.idxmin()],
        "shortest_ticket_words": int(word_lengths.min()),
        "longest_ticket": text.loc[word_lengths.idxmax()],
        "longest_ticket_words": int(word_lengths.max()),
        "largest_class_size": int(counts.max()),
        "smallest_class_size": int(counts.min()),
        "imbalance_ratio_largest_to_smallest": float(counts.max() / counts.min()),
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
    }
    save_json(summary, settings.metrics_dir / "eda_summary.json")

    plt.figure(figsize=(11, 8))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, palette="viridis")
    plt.title("Support ticket count by intent")
    plt.xlabel("Tickets")
    plt.ylabel("Intent")
    _save_current_figure("intent_distribution.png")

    plt.figure(figsize=(10, 5))
    sns.histplot(word_lengths, bins=30, kde=True)
    plt.axvline(word_lengths.quantile(0.95), color="darkred", linestyle="--", label="95th percentile")
    plt.title("Ticket length distribution")
    plt.xlabel("Words per ticket")
    plt.ylabel("Count")
    plt.legend()
    _save_current_figure("ticket_length_distribution.png")

    print(json.dumps(summary, indent=2))
    print("\nOne example from each intent:")
    examples = clean.groupby(settings.label_column, sort=True).first().reset_index()
    print(examples.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_eda()
