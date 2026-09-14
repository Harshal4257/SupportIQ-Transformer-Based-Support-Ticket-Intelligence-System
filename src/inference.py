"""Reusable local inference for a saved DistilBERT classifier."""

from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


class TicketClassifier:
    """Load a saved model once and classify one or many support tickets."""

    def __init__(
        self,
        model_path: Union[str, Path] = settings.transformer_output_dir,
        confidence_threshold: float = settings.confidence_threshold,
        max_length: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at {self.model_path}. Run: python -m src.train_transformer"
            )
        self.threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        saved_length = getattr(self.model.config, "support_ticket_max_length", None)
        self.max_length = int(max_length or saved_length or settings.max_length_cap)

    def predict(self, text: str) -> Dict[str, object]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Ticket text must be a non-empty string.")
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[Dict[str, object]]:
        if not texts:
            return []
        results: List[Dict[str, object]] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(text).strip() for text in texts[start : start + batch_size]]
            if any(not text for text in batch):
                raise ValueError("Every ticket must contain non-whitespace text.")
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_id = probabilities.max(dim=-1)

            for index, score in zip(predicted_id.cpu().tolist(), confidence.cpu().tolist()):
                intent = self.model.config.id2label[int(index)]
                results.append(
                    {
                        "intent": intent,
                        "confidence": round(float(score), 6),
                        "low_confidence": bool(score < self.threshold),
                    }
                )
        return results
