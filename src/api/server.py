#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from river import stream
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from src.config.setting import settings

# — PREPARA MODELLO & TOKENIZER —
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(settings.MODEL_PATH).to(device)
model.eval()


def resolve_sequence(seq: str) -> str:
    enc = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=settings.INPUT_MAXLEN
    ).to(device)

    # GENERAZIONE: genero solo NEW_TOKENS token in più
    out_ids = model.generate(
        **enc,
        max_new_tokens=settings.NEW_TOKENS,
        # puoi aggiungere qui beam_search o top_k:
        # num_beams=3,
        # top_k=50,
        # no_repeat_ngram_size=2,
    )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


# — FASTAPI SERVICE —
app = FastAPI()


class SeqIn(BaseModel):
    sequence: str


class SeqOut(BaseModel):
    resolved: str


@app.post("/resolve", response_model=SeqOut)
def resolve_endpoint(req: SeqIn):
    return SeqOut(resolved=resolve_sequence(req.sequence))


# — STREAMING WORKER —
def streaming_worker():
    idx = 0
    data_stream = stream.iter_csv(
        settings.CSV_FILE,
        target="target",
        converters={"input": str, "target": str}
    )
    for x, _ in data_stream:
        idx += 1
        seq_in = x["input"]
        try:
            seq_out = resolve_sequence(seq_in)
            if idx % settings.PRINT_EVERY == 0:
                print(f"[Stream {idx}] {seq_in[:20]}… → {seq_out[:20]}…")
        except Exception as e:
            print(f"⚠️ Errore alla riga {idx}: {e}")
        time.sleep(settings.STREAM_DELAY)


if __name__ == "__main__":
    # Avvia il worker in background
    t = threading.Thread(target=streaming_worker, daemon=True)
    t.start()

    # Avvia FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
