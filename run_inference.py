"""Interactive command-line inference with a locally saved model."""

import argparse

from src.config import settings
from src.inference import TicketClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a customer support ticket.")
    parser.add_argument("text", nargs="*", help="Ticket text; omit for interactive mode")
    parser.add_argument("--model-path", default=str(settings.transformer_output_dir))
    parser.add_argument("--threshold", type=float, default=settings.confidence_threshold)
    args = parser.parse_args()

    ticket = " ".join(args.text).strip()
    if not ticket:
        ticket = input("Enter ticket:\n").strip()

    classifier = TicketClassifier(args.model_path, confidence_threshold=args.threshold)
    result = classifier.predict(ticket)
    print("\nPrediction:")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")
    if result["low_confidence"]:
        print(f"Low confidence: yes (below {args.threshold:.0%})")


if __name__ == "__main__":
    main()
