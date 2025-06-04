#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from src.config.setting import settings  # Assicurati che settings.SEED sia definito


def compute_metrics(eval_pred):
    """
    Calcola l'accuratezza sui token (escludendo i -100).
    Restituisce {"token_accuracy": valore}.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    correct = (preds == labels) & mask
    total_tokens = mask.sum()
    correct_tokens = correct.sum()

    token_accuracy = float(correct_tokens) / float(total_tokens) if total_tokens > 0 else 0.0
    return {"token_accuracy": token_accuracy}


def load_and_split_datasets(data_dir: Path, seed: int, num_train: int, num_val: int, num_test: int):
    """
    Carica i file JSONL train/val/test e ritorna un DatasetDict con le prime slice di ogni split, mescolate.
    """
    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"))["train"].shuffle(seed=seed)
    val_ds = load_dataset("json", data_files=str(data_dir / "val.jsonl"))["train"].shuffle(seed=seed)
    test_ds = load_dataset("json", data_files=str(data_dir / "test.jsonl"))["train"].shuffle(seed=seed)

    train_subset = train_ds.select(range(num_train))
    val_subset = val_ds.select(range(num_val))
    test_subset = test_ds.select(range(num_test))

    return DatasetDict({
        "train": train_subset,
        "validation": val_subset,
        "test": test_subset
    })


def preprocess_and_tokenize(dataset_dict: DatasetDict, tokenizer: AutoTokenizer, max_length: int):
    """
    Applica tokenizzazione con masking dei token di prompt (label = -100),
    in modo da calcolare la loss solo sui token di risposta.
    """

    def preprocess_function(examples):
        inputs = [
            f"Input: {inp}\nOutput: {tgt}"
            for inp, tgt in zip(examples["input"], examples["target"])
        ]
        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        prompt_ids_list = []
        for inp in examples["input"]:
            prompt_only = f"Input: {inp}\nOutput:"
            tokenized_prompt = tokenizer(
                prompt_only,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length
            )
            prompt_ids_list.append(tokenized_prompt["input_ids"])

        labels = []
        for i, seq_ids in enumerate(model_inputs["input_ids"]):
            prompt_len = len(prompt_ids_list[i])
            label_ids = []
            for idx, token_id in enumerate(seq_ids):
                if token_id == tokenizer.pad_token_id or idx < prompt_len:
                    label_ids.append(-100)
                else:
                    label_ids.append(token_id)
            labels.append(label_ids)

        model_inputs["labels"] = labels
        return model_inputs

    tokenized = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=["input", "target"]
    )
    return tokenized


def train_batch_model_all(data_dir: Path):
    """
    Esegue un training batch su tutto il dataset (train/val/test),
    calcola metriche su validation e test, e stampa loss/token_accuracy/perplexity.
    """
    # 1) Carichiamo dataset
    NUM_TRAIN = 10000
    NUM_VAL = 2000
    NUM_TEST = 2000
    dataset_full = load_and_split_datasets(data_dir, settings.SEED, NUM_TRAIN, NUM_VAL, NUM_TEST)

    # 2) Tokenizzazione
    MODEL_NAME = "PoetschLab/GROVER"
    MAX_LENGTH = 128
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = preprocess_and_tokenize(dataset_full, tokenizer, MAX_LENGTH)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3) TrainingArguments e Trainer
    OUTPUT_DIR = Path("./batch_finetuned")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "hf_ckpts"),
        overwrite_output_dir=True,
        num_train_epochs=settings.BATCH_EPOCHS,
        per_device_train_batch_size=settings.BATCH_SIZE,
        per_device_eval_batch_size=settings.BATCH_SIZE,
        learning_rate=settings.BATCH_LR,
        weight_decay=settings.BATCH_WEIGHT_DECAY,
        warmup_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        greater_is_better=True,
        save_total_limit=3,
        seed=settings.SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 4) Esegui il training
    print("🎯 Avvio allenamento batch su tutto il dataset…")
    train_result = trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

    # 5) Recupera e stampa metriche Validation
    val_loss = val_metrics.get("eval_loss", None)
    val_token_acc = val_metrics.get("eval_token_accuracy", None)
    val_perplexity = math.exp(val_loss) if val_loss is not None else None

    print("\n📈 Metriche su VALIDATION SET:")
    print(f"  • Loss       = {val_loss:.4f}")
    print(f"  • Token Acc. = {val_token_acc:.4f}")
    print(f"  • Perplexity = {val_perplexity:.2f}")

    # 6) Recupera e stampa metriche Test
    test_loss = test_metrics.get("eval_loss", None)
    test_token_acc = test_metrics.get("eval_token_accuracy", None)
    test_perplexity = math.exp(test_loss) if test_loss is not None else None

    print("\n📊 Metriche su TEST SET:")
    print(f"  • Loss       = {test_loss:.4f}")
    print(f"  • Token Acc. = {test_token_acc:.4f}")
    print(f"  • Perplexity = {test_perplexity:.2f}")

    # 7) Salva modello + tokenizer finale
    final_dir = OUTPUT_DIR / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✅ Modello salvato in: {final_dir.resolve()}")


if __name__ == "__main__":
    data_dir = Path("../../data")
    train_batch_model_all(data_dir)
