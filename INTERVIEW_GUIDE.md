# Interview Guide

## Two-minute project explanation

I built a multi-class Support Ticket Intelligence System that routes free-form customer requests to one of 27 support intents. I used the public Bitext customer-support dataset because it has about 27,000 labeled examples and realistic variations in customer phrasing. I cleaned only invalid, duplicate, and conflicting-label rows. I preserved normal language because Transformer tokenizers do not need stemming or stopword removal.

I created one reproducible, stratified 70/15/15 train-validation-test split with seed 42. Deduplication happened before splitting to avoid the same request leaking between sets. My first model was a TF-IDF unigram/bigram representation with Logistic Regression. That gave me a fast, interpretable baseline and a reference point for whether a Transformer is worth its cost.

The second model fine-tuned `distilbert-base-uncased` with a sequence-classification head. I tokenized with truncation, chose maximum length from the 95th percentile of training token lengths, and used dynamic padding per batch. I trained with a small 2e-5 learning rate, weight decay, validation each epoch, macro-F1 checkpoint selection, and early stopping. The best model, tokenizer, label mappings, and maximum length were saved together for local reuse.

I evaluated both models on the untouched test set using accuracy, macro and weighted precision, recall, and F1, plus per-class reports and confusion matrices. I also saved every error with its true label, predicted label, and softmax confidence, then ranked the most confused intent pairs. I report only metrics produced by actual runs. For inference, a reusable class loads the local model, returns an intent and confidence, and flags predictions below a configurable threshold for human review. The next production steps would be domain-specific data, confidence calibration, drift monitoring, and an explicit unknown-intent strategy.

## Core questions and answers

### What problem does this project solve?

It maps an unstructured support request to a routing intent. That can reduce manual triage time, improve queue assignment, and quantify why customers contact support. It predicts an intent; it does not resolve the ticket or generate an answer.

### Why did you choose this dataset?

Bitext is public, large enough for meaningful fine-tuning, centered on customer service, and has 27 intent classes rather than a toy label set. Its generated nature is also a limitation: results must be validated on real company tickets before operational use.

### Why use DistilBERT?

It preserves contextual Transformer representations while being smaller and faster than BERT-base. That makes it a practical learning and local-inference model. The decision is a latency/quality tradeoff, not a claim that it is always the best encoder.

### Why compare against TF-IDF + Logistic Regression?

A baseline tells me whether added complexity produces meaningful value. Logistic Regression trains quickly, is easy to debug, uses few resources, and can be competitive when intents have distinctive keywords.

### What is tokenization?

It converts text into the model's subword tokens, adds required special tokens, then maps tokens to numeric IDs. The same saved tokenizer must be used at training and inference.

### What are input IDs?

They are integer indices into the model's token embedding vocabulary. Their numeric values are identifiers, not quantities with ordinal meaning.

### What is the attention mask?

It marks real tokens with 1 and padding with 0, preventing padded positions from being treated as ticket content during attention.

### Why is padding needed?

Tensors in a batch need a common sequence length. Padding fills shorter sequences to that length so they can be stacked.

### What is dynamic padding?

Each batch is padded only to its longest sequence. This usually uses less memory and computation than padding every example to the global maximum.

### What is truncation?

It cuts tokens beyond a maximum length. The project derives the length from training data and caps it, balancing retained context against compute and memory.

### What does AutoModelForSequenceClassification add on top of DistilBERT?

It constructs the base encoder plus a classification head that outputs one logit for each intent, and it calculates classification loss when labels are supplied.

### What is a classification head?

It is the task-specific final network that converts a contextual sequence representation into class logits. It starts untrained for these labels and learns during fine-tuning.

### What does fine-tuning mean?

It continues training pretrained weights on the labeled task, usually with a small learning rate. The model adapts general language knowledge to support-intent boundaries.

### Why use a learning rate like 2e-5?

Pretrained representations are already useful. A small learning rate allows controlled adaptation without rapidly overwriting them. It is a sensible starting point and should be validated rather than treated as universal.

### Why use macro F1?

Macro F1 calculates F1 per class and averages equally, so weak performance on a smaller intent is visible. That aligns with a routing system where every supported intent matters.

### Why can accuracy be misleading?

With imbalance, a model can be accurate mainly by doing well on common classes. Accuracy also hides which intent pairs are confused and whether a specific class has poor recall.

### Why use stratified splitting?

It approximately preserves each label's proportion in train, validation, and test, making evaluation more stable and ensuring all classes are represented.

### How did you prevent data leakage?

I cleaned conflicting labels and deduplicated normalized ticket text before splitting. I fit TF-IDF only on training data by using a pipeline, chose Transformer length from training text only, selected checkpoints on validation macro F1, and evaluated final performance on the untouched test set.

### What are logits?

Logits are the model's unnormalized output scores, one per intent. They can be any real value and do not sum to one.

### How is confidence calculated?

Softmax converts logits into values that sum to one, and the maximum is returned with the predicted intent. Neural softmax scores can be miscalibrated, so confidence needs calibration and threshold validation before production use.

### What mistakes did the model make?

Answer this from the generated `distilbert_error_analysis.csv` and `distilbert_most_confused_classes.csv`; do not guess. Name two frequent pairs, quote or paraphrase representative tickets, and relate each mistake to overlap, missing context, label quality, or data coverage.

### Which intents were most difficult to distinguish?

Use the saved confusion-pair table from your actual run. Similar actions with different states - for example requesting, checking, changing, or cancelling the same object - are likely candidates, but only report what your results show.

### How would you improve the model?

First clarify label definitions and collect representative real errors. Then consider targeted examples, an unknown intent, calibrated thresholds, class weighting if needed, hyperparameter tuning on validation data, and a more suitable encoder. Improvements should be driven by error categories and business costs.

### Why might Logistic Regression still be useful?

It is cheap, fast, explainable through feature weights, easy to retrain, and can be strong for keyword-separable intents. It is also a fallback when hardware or latency constraints dominate.

### How would this system be used in a real company?

It could score incoming tickets and suggest a queue. Low-confidence or high-risk cases would go to humans. The company would log predictions and corrections, monitor class and language drift, protect personal data, measure per-intent service outcomes, and periodically retrain under version control.

## Follow-up discussion points

### How did you choose the best checkpoint?

Validation macro F1 is evaluated each epoch. The matching checkpoint is saved, early stopping monitors improvement, and `load_best_model_at_end` restores the best checkpoint before test evaluation and final saving.

### Why have separate validation and test sets?

Validation supports choices such as checkpoint, learning rate, sequence length policy, and threshold. The test set is reserved for one unbiased estimate after those choices. Repeatedly adjusting the model from test results turns the test set into training information.

### What does the baseline miss that DistilBERT may capture?

TF-IDF mostly treats phrases as weighted surface features. DistilBERT builds context-dependent representations, so negation, wording variation, and the role of a word in a sentence can be represented more flexibly.

### What are the main project limitations?

The source data is synthetic/hybrid rather than company traffic, intent scope is fixed, English is assumed, confidence is not calibrated, and no unknown-intent detector exists. Offline classification metrics also do not directly measure routing time saved or customer outcomes.

### How would you measure production success?

Track human-accepted routing accuracy, reroute rate, per-intent recall, low-confidence coverage, latency, handling time, escalation rate, and drift. Evaluate costs asymmetrically when sending a ticket to the wrong queue is worse than requesting manual review.
