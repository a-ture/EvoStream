#!/usr/bin/env python3
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

# — Configurazione —
MODEL_ID   = "PoetschLab/GROVER"
DATA_DIR   = Path(".")  # dove ci sono train.jsonl, val.jsonl, test.jsonl
OUTPUT_DIR = Path("batch_output")
SEED       = 42

set_seed(SEED)

# 1) Tokenizer e modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Assicura i token DNA/ambigui
dna_tokens = list("ATCG") + list("NRYSWKMBVDH")
new_toks = [t for t in dna_tokens if t not in tokenizer.get_vocab()]
if new_toks:
    tokenizer.add_tokens(new_toks)
    model.resize_token_embeddings(len(tokenizer))

# 2) Carica dataset
data_files = {
    "train": str(DATA_DIR/"train.jsonl"),
    "validation": str(DATA_DIR/"val.jsonl"),
}
raw_ds = load_dataset("json", data_files=data_files)

# 3) Tokenizzazione
def tokenize_fn(ex):
    in_enc = tokenizer(ex["input"],  truncation=True, padding="max_length", max_length=128)
    tgt_enc = tokenizer(ex["target"], truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": in_enc["input_ids"],
        "attention_mask": in_enc["attention_mask"],
        "labels": tgt_enc["input_ids"]
    }

tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=["input","target"])

# 4) Data collator
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=200,
    save_steps=200,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset= tokenized["validation"],
    data_collator=collator,
)

# 6) Avvia training
trainer.train()
trainer.save_model(str(OUTPUT_DIR/"final"))
tokenizer.save_pretrained(str(OUTPUT_DIR/"final"))
print("✅ Batch training completato.")
