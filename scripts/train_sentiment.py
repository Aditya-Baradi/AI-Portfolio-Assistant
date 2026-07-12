"""
Fine-tune DistilBERT on financial-headline sentiment.

Run this on Google Colab (free GPU) or any machine with a GPU:

    pip install transformers datasets torch scikit-learn
    python scripts/train_sentiment.py

It downloads the Financial PhraseBank dataset, fine-tunes DistilBERT to
classify headlines as negative/neutral/positive, prints a held-out accuracy
comparison target (VADER scores ~56% on this data, FinBERT ~86%), and saves
the model to ./finetuned-sentiment.

To use the result in the app, copy the folder next to the project and run:

    SENTIMENT_BACKEND=finbert FINBERT_MODEL=./finetuned-sentiment uvicorn api.backend:app

(api/sentiment.py already routes through any HF model via SENTIMENT_BACKEND;
point FINBERT_MODEL at your folder, or push the model to the HF Hub.)
"""
from __future__ import annotations

import numpy as np


def main() -> None:
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # sentences_allagree = the subset where every annotator agreed (cleanest labels)
    ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
    ds = ds.train_test_split(test_size=0.15, seed=7)

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )

    def encode(batch):
        return tok(batch["sentence"], truncation=True, padding="max_length", max_length=64)

    ds = ds.map(encode, batched=True).rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def metrics(pred):
        return {"accuracy": accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=1))}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./sentiment-checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            evaluation_strategy="epoch",
            logging_steps=50,
            seed=7,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=metrics,
    )
    trainer.train()
    print(trainer.evaluate())

    model.save_pretrained("./finetuned-sentiment")
    tok.save_pretrained("./finetuned-sentiment")
    print("Saved to ./finetuned-sentiment")


if __name__ == "__main__":
    main()
