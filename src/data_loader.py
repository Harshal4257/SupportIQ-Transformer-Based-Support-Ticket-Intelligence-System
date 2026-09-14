"""Download, validate, clean, split, and reload the Bitext dataset."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import settings
from src.preprocessing import clean_dataframe, stratified_split
from src.utils import ensure_directories, save_json


RAW_SNAPSHOT = settings.raw_data_dir / "bitext_customer_support.csv"


def _find_local_csv(path: Optional[Path] = None) -> Optional[Path]:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Local dataset not found: {path}")
        return path
    if RAW_SNAPSHOT.exists():
        return RAW_SNAPSHOT
    candidates = sorted(settings.raw_data_dir.glob("*.csv"))
    return candidates[0] if candidates else None


def load_raw_dataframe(local_csv: Optional[Path] = None, force_download: bool = False) -> pd.DataFrame:
    """Load a local CSV, cached snapshot, or download the public Hugging Face data."""
    ensure_directories()
    local_path = None if force_download else _find_local_csv(local_csv)
    if local_path is not None:
        return pd.read_csv(local_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install requirements with `pip install -r requirements.txt`, or place the "
            f"Bitext CSV in {settings.raw_data_dir}."
        ) from exc

    try:
        dataset = load_dataset(settings.dataset_name, split="train")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check internet access, or download the Bitext CSV "
            f"and place it at {RAW_SNAPSHOT}."
        ) from exc
    frame = dataset.to_pandas()
    frame.to_csv(RAW_SNAPSHOT, index=False)
    return frame


def load_prepared_splits() -> Dict[str, pd.DataFrame]:
    """Load already-created splits without silently recreating the test set."""
    paths = {
        name: settings.processed_data_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared splits do not exist. Run `python -m src.data_loader` first. "
            f"Missing: {missing}"
        )
    return {name: pd.read_csv(path) for name, path in paths.items()}


def prepare_data(
    local_csv: Optional[Path] = None,
    force_download: bool = False,
    force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Prepare splits once and reuse them across both models to ensure a fair comparison."""
    ensure_directories()
    split_paths = {
        name: settings.processed_data_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    if not force_rebuild and all(path.exists() for path in split_paths.values()):
        return load_prepared_splits()

    raw = load_raw_dataframe(local_csv=local_csv, force_download=force_download)
    cleaned, cleaning_stats = clean_dataframe(raw)
    splits = stratified_split(cleaned)
    labels = sorted(cleaned[settings.label_column].unique())

    for name, frame in splits.items():
        frame.to_csv(split_paths[name], index=False)
    save_json(
        {
            "dataset_name": settings.dataset_name,
            "seed": settings.random_seed,
            "text_column": settings.text_column,
            "label_column": settings.label_column,
            "split_rows": {name: len(frame) for name, frame in splits.items()},
            "split_proportions": {
                "train": settings.train_size,
                "validation": settings.validation_size,
                "test": settings.test_size,
            },
            "labels": labels,
            "label2id": {label: index for index, label in enumerate(labels)},
            "id2label": {index: label for index, label in enumerate(labels)},
            "cleaning": cleaning_stats,
        },
        settings.processed_data_dir / "dataset_metadata.json",
    )
    return splits


def main() -> None:
    splits = prepare_data()
    print("Prepared dataset:")
    for name, frame in splits.items():
        print(f"  {name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
