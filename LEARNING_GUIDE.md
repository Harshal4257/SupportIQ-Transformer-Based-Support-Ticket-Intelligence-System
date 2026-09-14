# Learning Guide

Study the project in this order. Run each stage, inspect its artifacts, and explain the reason for every design choice before moving on.

## 1. Dataset

Start with `src/config.py` and `src/data_loader.py`. Bitext supplies customer `instruction` text paired with an `intent`. Only those two columns enter this supervised classification task. `response` would leak answer-writing information into a system whose input at inference time is only the customer's request, so it is excluded.

Run `python -m src.data_loader`. Inspect `data/processed/dataset_metadata.json`, then verify that every split contains the same label set and no text occurs in more than one split.

## 2. Exploratory data analysis

Open `notebooks/01_eda.ipynb` and read `src/eda.py`. EDA checks shape, columns, missing data, duplicate rows, class counts, ticket lengths, extremes, and one example per class. The class-size ratio is a quick imbalance signal; the label plot shows where aggregate metrics may conceal weak minority performance.

The length distribution informs truncation. A maximum length of 512 is wasteful for short support requests, while a limit that is too small deletes useful context.

## 3. Preprocessing

Read `src/preprocessing.py`. Cleaning is deliberately conservative: discard unusable records, normalize label names, remove ambiguous label conflicts, and deduplicate before splitting. Do not remove stopwords, punctuation, spelling variants, or inflections from Transformer input. Pretrained subword tokenizers expect language closer to their pretraining distribution, and words such as "not" can reverse intent.

## 4. TF-IDF baseline

Open `notebooks/02_baseline.ipynb`, then read `src/train_baseline.py`. TF-IDF increases the importance of terms distinctive to a document while down-weighting terms common across the corpus. Bigrams capture local phrases such as "cancel order". Logistic Regression learns one linear decision score per class and turns the scores into probabilities.

Rebuild this stage yourself: load the fixed splits, fit a `TfidfVectorizer`, train Logistic Regression, and calculate macro F1 without copying the pipeline.

## 5. Tokenizer

In `notebooks/03_distilbert_training.ipynb`, tokenize several tickets and compare their words with tokenizer tokens. DistilBERT uses a WordPiece vocabulary. Unknown-looking words can be represented as known subword pieces, allowing the model to handle variants better than a fixed whole-word vocabulary.

## 6. Input IDs

Tokens are converted to integer vocabulary indices called input IDs. The model does not read strings; an embedding table maps each ID to a learned vector. Special tokens mark the sequence boundaries and are included in the chosen maximum length.

## 7. Attention mask

The attention mask is 1 for real tokens and 0 for padding positions. It tells self-attention which positions contain content. Without the mask, padding could influence the representation.

## 8. DistilBERT

DistilBERT is a smaller model distilled from BERT. Its Transformer layers repeatedly combine contextual information using self-attention and feed-forward networks. Unlike TF-IDF, the representation of a token changes with its surrounding words. This helps distinguish similar vocabulary used in different contexts.

## 9. Classification head

`AutoModelForSequenceClassification` loads the encoder plus a task-specific head. The head uses the final representation of the first sequence token, applies regularization/projection defined by the architecture, and produces one logit per intent. At initialization, the new head has not learned the support labels.

## 10. Fine-tuning

Fine-tuning updates the pretrained encoder and new head together on labeled support examples. This is not training DistilBERT from scratch: general language representations are reused, then adjusted for intent boundaries.

## 11. TrainingArguments

Read `_training_arguments()` in `src/train_transformer.py`. Important settings control learning rate, batch sizes, epochs, weight decay, reproducibility, evaluation frequency, checkpoint frequency, mixed precision, and which metric chooses the best checkpoint. Evaluation and saving both happen each epoch so `load_best_model_at_end` can restore a compatible checkpoint.

## 12. Trainer

`Trainer` owns the optimization loop, batching, device placement, evaluation, checkpointing, and callback events. The project still supplies the model, tokenized datasets, dynamic-padding collator, metrics function, and early-stopping rule. Understand these inputs before treating Trainer as a convenience abstraction.

## 13. Metrics

Accuracy is the fraction correct. Per-class precision asks how often predictions of one intent are right; recall asks how many true examples of that intent were found. F1 balances precision and recall. Macro F1 weights every class equally. Weighted F1 weights by class frequency, so majority classes contribute more.

Rebuild this stage yourself using `sklearn.metrics`, first with a tiny hand-worked example and then the model outputs.

## 14. Confusion matrix

Rows are actual intents and columns are predictions. The diagonal is correct. Large off-diagonal cells reveal systematic boundaries the model has not learned. Always read the axis labels rather than relying on visual symmetry.

## 15. Error analysis

Open `reports/metrics/distilbert_error_analysis.csv` after training. Sort by confidence to find dangerous confident mistakes, and group actual/predicted pairs to find recurring confusion. Read the ticket text: missing context, overlapping intent definitions, label noise, rare phrasing, and truncation require different remedies.

## 16. Inference

Read `src/inference.py`. Inference repeats the same tokenizer and maximum length, places tensors on CPU or CUDA, disables gradients, computes logits, applies softmax, and maps the largest-probability index back to the original label. Batch inference is much faster than invoking the model once per ticket.

## 17. Saving and loading

`trainer.save_model` saves model weights and config, while `tokenizer.save_pretrained` saves tokenizer files. The config carries `id2label`, `label2id`, and the selected maximum length. `TicketClassifier` loads everything from the local directory, so it does not retrain and does not need the original dataset.

## What to rebuild yourself

After understanding the generated version, rebuild these pieces in order:

1. cleaning plus a stratified split, including a duplicate-leakage check
2. TF-IDF + Logistic Regression and its metric calculations
3. tokenization of one batch, including input IDs and attention masks
4. a raw PyTorch forward pass and softmax conversion
5. the Trainer setup from only the documentation and config values
6. confusion-pair aggregation and a written error analysis
7. a minimal saved-model inference class

Finally, change one design choice at a time and record the result. Useful experiments include unigram versus bigram TF-IDF, 90th versus 95th-percentile truncation, learning rate, and confidence threshold. Never tune decisions on the test set.
