#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from river import stream

# — CONFIGURAZIONE FLUSSO & TRAINING —
MODEL_ID        = "PoetschLab/GROVER"
CSV_FILE        = "train.csv"          # file CSV con colonne 'input','target'
CKPT_DIR        = Path("river_ckpts")  # dove salvare i checkpoint
SEED            = 42
LR              = 5e-6
MAX_LENGTH      = 128
SAVE_EVERY      = 200  # checkpoint ogni 200 esempi
PRINT_EVERY     = 10   # log ogni 10 esempi

# Seed e device
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Tokenizer & modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

# Aggiungi eventuali token DNA/ambigui
dna_tokens = list("ATCG") + list("NRYSWKMBVDH")
new_toks = [t for t in dna_tokens if t not in tokenizer.get_vocab()]
if new_toks:
    tokenizer.add_tokens(new_toks)
    model.resize_token_embeddings(len(tokenizer))

# 2) Crea lo stream River dal CSV
data_stream = stream.iter_csv(
    CSV_FILE,
    target="target",
    converters={"input": str, "target": str}
)

# 3) Loop di training online
for idx, (x, y) in enumerate(data_stream, start=1):
    seq_in  = x["input"]
    seq_tgt = y

    # Tokenizza singolo esempio → batch shape [1, MAX_LENGTH]
    enc = tokenizer(
        seq_in,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)
    labels = tokenizer(
        seq_tgt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    ).input_ids.to(device)

    # Forward / backward
    outputs = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        labels=labels
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log e checkpoint
    if idx % PRINT_EVERY == 0:
        print(f"[Step {idx}] Loss: {loss.item():.4f}")
    if idx % SAVE_EVERY == 0:
        ckpt = CKPT_DIR / f"step_{idx}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"✅ Checkpoint salvato in {ckpt}")

# Salvataggio finale
final_ckpt = CKPT_DIR / "final"
final_ckpt.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_ckpt)
tokenizer.save_pretrained(final_ckpt)
print("✅ Online training (River) completato.")
