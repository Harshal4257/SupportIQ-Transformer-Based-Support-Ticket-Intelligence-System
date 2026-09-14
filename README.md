# Support Ticket Intelligence System

A complete, local machine-learning project for multi-class customer-support intent classification. It compares a transparent TF-IDF + Logistic Regression baseline with a fine-tuned `distilbert-base-uncased` Transformer, evaluates both on the same held-out test set, and exposes the saved Transformer through terminal, FastAPI, and Streamlit interfaces.

## Problem and business use case

Support teams receive large volumes of free-form messages. Intent classification turns each ticket into a routing label such as `cancel_order`, `get_refund`, or `recover_password`. A company can use the prediction to prioritize queues, route work to specialists, report demand by issue type, and flag uncertain tickets for human triage. The classifier does not write support answers and should not make irreversible customer-service decisions by itself.

## Dataset

The project uses the public [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Its dataset card describes 26,872 English examples across 27 intents. The relevant fields are:

- `instruction`: customer request used as classifier input
- `intent`: target class
- `category`, `response`, and `flags`: available for analysis but intentionally excluded from model input

`python -m src.data_loader` downloads the dataset through Hugging Face Datasets and caches a CSV in `data/raw/`. If downloading is unavailable, manually download the source CSV and place any single `.csv` file containing `instruction` and `intent` columns in `data/raw/`.

The loader strips surrounding whitespace, removes empty rows, normalizes only the label spelling, removes exact duplicate text before splitting, and removes text associated with conflicting labels. It does not remove stopwords, stem, lemmatize, or discard punctuation. A pretrained Transformer tokenizer was learned from natural language, so aggressive traditional cleaning can destroy useful wording and punctuation.

## Project structure

```text
data/
  raw/                         downloaded source CSV (ignored by Git)
  processed/                   fixed train/validation/test CSVs and metadata
notebooks/
  01_eda.ipynb
  02_baseline.ipynb
  03_distilbert_training.ipynb
src/
  api.py                       FastAPI prediction service
  config.py                    paths and hyperparameters
  data_loader.py               download, cache, and split
  preprocessing.py             minimal cleaning and stratification
  eda.py                       reusable EDA and figures
  train_baseline.py            TF-IDF + Logistic Regression
  train_transformer.py         DistilBERT fine-tuning
  evaluate.py                  metrics, errors, comparison, CLI
  inference.py                 reusable TicketClassifier
models/                        generated models/checkpoints (ignored by Git)
reports/
  figures/                     EDA and confusion-matrix PNG files
  metrics/                     JSON/CSV metrics and error tables
run_inference.py               terminal inference entry point
streamlit_app.py               browser-based prediction interface
LEARNING_GUIDE.md
INTERVIEW_GUIDE.md
```

## Reproducible data split

The cleaned data is split once with seed 42 and stratification:

- 70% training
- 15% validation
- 15% testing

Duplicate texts are removed before splitting. Both models reuse the CSV files in `data/processed/`, and the test labels are not used for fitting or model selection. Delete the processed CSVs only when intentionally rebuilding the split.

The completed run retained 24,274 unique, unambiguous tickets: 16,991 train, 3,641 validation, and 3,642 test. It found 27 intents. Ticket length averaged 8.86 words, with a 95th-percentile token length of 19 and a maximum of 32 tokens. The largest class contained 997 examples and the smallest 436, an imbalance ratio of 2.29.

## Models

### Baseline

The baseline is a scikit-learn pipeline with word unigram/bigram TF-IDF features and multinomial Logistic Regression. It is fast, interpretable, and establishes whether the Transformer adds enough value to justify its compute cost.

### DistilBERT

The Transformer uses `AutoTokenizer`, `AutoModelForSequenceClassification`, PyTorch, and Hugging Face `Trainer`. A randomly initialized classification head maps DistilBERT's representation to the 27 label logits. Defaults are a 2e-5 learning rate, batch size 16, three epochs, 0.01 weight decay, evaluation/checkpointing each epoch, macro-F1 model selection, dynamic padding, and early stopping.

The maximum sequence length is the 95th percentile of tokenized training lengths, bounded to 32-256 tokens. It is learned only from the training split and saved in the model config. `DataCollatorWithPadding` pads each batch to its longest sequence instead of padding the whole dataset to a fixed size. CUDA and mixed precision are selected automatically when CUDA is available.

## Evaluation and error analysis

Both classifiers produce:

- accuracy
- macro and weighted precision/recall/F1
- JSON and CSV classification reports with per-class metrics
- a confusion matrix
- incorrectly classified tickets with actual intent, predicted intent, and confidence
- the most common directed confusion pairs
- end-to-end training time, test inference time, and milliseconds per ticket

The final `reports/metrics/model_comparison.csv` uses only metrics written by completed runs. This repository contains no invented scores. The completed local CPU run produced:

| Model | Accuracy | Macro F1 | Weighted F1 | Training time | Test inference | ms/ticket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9898 | 0.9888 | 0.9898 | 2.99 s | 0.25 s | 0.069 |
| DistilBERT | 0.9964 | 0.9964 | 0.9964 | 3,546.01 s | 55.45 s | 15.225 |

These measurements came from the same 3,642-ticket test split on the local machine. Timing is hardware-specific; the DistilBERT run used CPU. Inspect the full-precision generated JSON/CSV files or run:

```powershell
Get-Content reports/metrics/model_comparison.csv
```

The most frequent DistilBERT confusion directions were `get_invoice` -> `check_invoice`, `check_invoice` -> `get_invoice`, and `payment_issue` -> `check_payment_methods`, with two test errors each. This matches the semantic overlap between obtaining versus checking invoices and payment failures versus supported payment methods. Review the underlying rows before changing labels or training data.

## Installation

Python 3.10 or newer is recommended. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation varies by operating system and CUDA version. If you need a CUDA-specific wheel, install the appropriate PyTorch build first, then install the remaining requirements.

## Run the project

Prepare and inspect the data:

```powershell
python -m src.data_loader
python -m src.eda
jupyter lab notebooks/01_eda.ipynb
```

Train the baseline:

```powershell
python -m src.train_baseline
```

Fine-tune DistilBERT (GPU recommended but not required):

```powershell
python -m src.train_transformer
```

Re-evaluate either saved model against the unchanged test split:

```powershell
python -m src.evaluate --model baseline
python -m src.evaluate --model distilbert
```

Use interactive inference after Transformer training:

```powershell
python run_inference.py
```

Start the FastAPI backend from the project root:

```powershell
uvicorn src.api:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
In a second terminal, start the Streamlit frontend:

```powershell
streamlit run streamlit_app.py
```

Streamlit uses `http://127.0.0.1:8000` by default. Override it with the
`SUPPORT_API_URL` environment variable or the sidebar input. The backend also
accepts `SUPPORT_MODEL_PATH` and `SUPPORT_CONFIDENCE_THRESHOLD` environment
variables.

Or pass a ticket directly:

```powershell
python run_inference.py "I cannot access my account"
python run_inference.py --threshold 0.75 "I need help with my order"
```

Python usage:

```python
from src.inference import TicketClassifier

classifier = TicketClassifier("models/distilbert-support-classifier")
result = classifier.predict("My payment was deducted twice.")
print(result)
# Example from the completed run is shown below.
```

The longer original example, "My payment was deducted twice and I have not received my refund.", produced `{'intent': 'get_refund', 'confidence': 0.997003, 'low_confidence': False}`. A login request is outside the Bitext label set; the saved model returned `switch_account` at 0.469833 confidence and set `low_confidence` to `True`. This illustrates why an unknown-intent strategy and human review remain important.

The result has `intent`, softmax `confidence`, and `low_confidence`. Softmax confidence is a useful ranking signal, not a guarantee that the model is correct; production use should validate and calibrate it.

## Saved artifacts

- Baseline: `models/tfidf-logistic-regression.joblib`
- Best Transformer and tokenizer: `models/distilbert-support-classifier/`
- Temporary Transformer checkpoints: `models/checkpoints/`
- Metrics and error tables: `reports/metrics/`
- Figures: `reports/figures/`

Model binaries, raw data, processed data, and generated reports are ignored by Git.

## Future improvements

- collect real, consented, domain-specific tickets and redact personal data
- refine intent definitions and add an explicit out-of-scope/unknown class
- calibrate confidence and tune the human-review threshold on validation data
- inspect slices by ticket length, language quality, channel, and customer segment
- compare class weighting, focal loss, and targeted data augmentation if imbalance appears
- try compact domain-adapted encoders and measure latency/accuracy tradeoffs
- add batch monitoring, drift checks, an API, UI, tests, packaging, and deployment only in a later production phase

See [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for a study path and [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) for project explanations and interview questions.
