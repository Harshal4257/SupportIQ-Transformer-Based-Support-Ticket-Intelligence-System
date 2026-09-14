"""Minimal cleaning and leakage-safe splitting for support tickets."""

import re
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings


def normalize_label(label: str) -> str:
    """Convert labels to stable snake_case without changing their meaning."""
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def clean_dataframe(
    frame: pd.DataFrame,
    text_column: str = settings.text_column,
    label_column: str = settings.label_column,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove unusable/ambiguous rows while preserving natural Transformer input.

    Stopwords, punctuation, casing, spelling, and morphology are deliberately kept.
    Exact duplicate texts are removed before splitting so they cannot leak across sets.
    If one text has conflicting labels, every occurrence is removed as ambiguous.
    """
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = frame[[text_column, label_column]].copy()
    original_rows = len(data)
    data = data.dropna(subset=[text_column, label_column])
    data[text_column] = data[text_column].astype(str).str.strip()
    data[label_column] = data[label_column].map(normalize_label)
    data = data[(data[text_column].str.len() > 0) & (data[label_column].str.len() > 0)]
    valid_rows = len(data)

    normalized_text = data[text_column].str.casefold().str.replace(r"\s+", " ", regex=True)
    conflict_counts = data.assign(_text_key=normalized_text).groupby("_text_key")[label_column].nunique()
    conflicting_keys = set(conflict_counts[conflict_counts > 1].index)
    conflict_mask = normalized_text.isin(conflicting_keys)
    conflicting_rows = int(conflict_mask.sum())
    data = data.loc[~conflict_mask].copy()
    data["_text_key"] = normalized_text.loc[~conflict_mask]
    before_dedup = len(data)
    data = data.drop_duplicates(subset=["_text_key"], keep="first").drop(columns="_text_key")
    data = data.reset_index(drop=True)

    stats = {
        "original_rows": original_rows,
        "missing_or_invalid_rows_removed": original_rows - valid_rows,
        "conflicting_label_rows_removed": conflicting_rows,
        "duplicate_texts_removed": before_dedup - len(data),
        "clean_rows": len(data),
        "number_of_classes": int(data[label_column].nunique()),
    }
    return data, stats


def stratified_split(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create deterministic 70/15/15 stratified train/validation/test sets."""
    counts = data[settings.label_column].value_counts()
    if (counts < 3).any():
        rare = counts[counts < 3].to_dict()
        raise ValueError(f"Every class needs at least 3 rows for stratification; found {rare}")

    train, remainder = train_test_split(
        data,
        train_size=settings.train_size,
        random_state=settings.random_seed,
        stratify=data[settings.label_column],
    )
    relative_validation_size = settings.validation_size / (
        settings.validation_size + settings.test_size
    )
    validation, test = train_test_split(
        remainder,
        train_size=relative_validation_size,
        random_state=settings.random_seed,
        stratify=remainder[settings.label_column],
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
