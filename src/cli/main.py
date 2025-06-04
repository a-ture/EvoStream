#!/usr/bin/env python3
# main.py

import json, time, math, subprocess, threading
from pathlib import Path

import typer
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    set_seed, DataCollatorForLanguageModeling,
)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from river import stream

from src.config.setting import settings

app = typer.Typer()


set_seed(settings.SEED)


# --- 1) DATASET ---
@app.command()
def gen_dataset(
        fasta: Path = typer.Argument(..., help="ecoli_ref.fasta"),
        out_dir: Path = typer.Option(Path("../.."), help="cartella output"),
        n_samples: int = typer.Option(1000),
        frag_length: int = typer.Option(100),
        p_ambig: float = typer.Option(0.05),
):
    """Genera train/val/test JSONL da un FASTA di riferimento."""
    from Bio import SeqIO
    BASES = ['A', 'T', 'C', 'G']
    AMBIG = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']

    rec = SeqIO.read(str(fasta), "fasta")
    seq = str(rec.seq)
    L = len(seq)
    out_dir.mkdir(exist_ok=True, parents=True)
    files = {
        "train": open(out_dir / "train.jsonl", "w"),
        "val": open(out_dir / "val.jsonl", "w"),
        "test": open(out_dir / "test.jsonl", "w"),
    }
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    import random
    for i in range(n_samples):
        start = random.randint(0, L - frag_length)
        frag = seq[start:start + frag_length]
        amb_frag = "".join(random.choice(AMBIG) if random.random() < p_ambig else c for c in frag)
        rec = {"input": amb_frag, "target": frag}
        r = random.random()
        if r < ratios["train"]:
            dst = "train"
        elif r < ratios["train"] + ratios["val"]:
            dst = "val"
        else:
            dst = "test"
        files[dst].write(json.dumps(rec) + "\n")
    for f in files.values(): f.close()
    typer.echo(f"✅ Dataset generato in {out_dir}")


# --- 2) BATCH TRAINING ---
@app.command()
def train_batch(
        data_dir: Path = typer.Option(Path("../..")),
        output_dir: Path = typer.Option(Path("../../checkpoints/batch_output")),
        epochs: int = typer.Option(3),
        bs: int = typer.Option(8),
        lr: float = typer.Option(5e-5),
):
    """Fine-tuning batch con Trainer."""
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_ID)
    # aggiungi token DNA
    extra = list("ATCGNRYSWKMBVDH")
    new = [t for t in extra if t not in tokenizer.get_vocab()]
    if new:
        tokenizer.add_tokens(new)
        model.resize_token_embeddings(len(tokenizer))
    # dataset
    files = {"train": str(data_dir / "train.jsonl"),
             "validation": str(data_dir / "val.jsonl")}
    ds = load_dataset("json", data_files=files)

    def tok_fn(ex):
        i = tokenizer(ex["input"], padding="max_length", truncation=True, max_length=128)
        t = tokenizer(ex["target"], padding="max_length", truncation=True, max_length=128)
        return {"input_ids": i["input_ids"], "attention_mask": i["attention_mask"], "labels": t["input_ids"]}

    tokd = ds.map(tok_fn, batched=True, remove_columns=["input", "target"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=str(output_dir), overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        evaluate_during_training=True, eval_steps=200,
        save_steps=200, logging_steps=100,
        learning_rate=lr, weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model, args, train_dataset=tokd["train"], eval_dataset=tokd["validation"], data_collator=collator)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    typer.echo("✅ Batch training completato.")


# --- 3) ONLINE TRAINING ---
@app.command()
def train_online(
        train_file: Path = typer.Option(Path("../../data/train.jsonl")),
        output_dir: Path = typer.Option(Path("../../checkpoints/online_ckpts")),
        epochs: int = typer.Option(3),
        lr: float = typer.Option(5e-6),
):
    """Fine-tuning online (streaming)."""
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_ID).to("cpu")
    extra = list("ATCGNRYSWKMBVDH")
    new = [t for t in extra if t not in tokenizer.get_vocab()]
    if new:
        tokenizer.add_tokens(new)
        model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ds = load_dataset("json", data_files={"train": str(train_file)})["train"]
    for ep in range(1, epochs + 1):
        for idx, ex in enumerate(ds, start=1):
            enc = tokenizer(ex["input"], return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(
                "cpu")
            lbl = tokenizer(ex["target"], return_tensors="pt", padding="max_length", truncation=True,
                            max_length=128).input_ids.to("cpu")
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lbl)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if idx % 100 == 0:
                typer.echo(f"Epoch{ep} Step{idx} Loss={out.loss.item():.4f}")
        ckpt = output_dir / f"epoch{ep}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt);
        tokenizer.save_pretrained(ckpt)
    typer.echo("✅ Online training completato.")


# --- 4) COMPARE MODELS ---
@app.command()
def compare(test_file: Path = typer.Option(Path("../../data/test.jsonl"))):
    """Valuta batch vs online su test.jsonl (no leakage)."""
    MODELS = {"batch": "batch_output/final", "online": "online_ckpts/final"}
    for name, path in MODELS.items():
        typer.echo(f"\n▶️ {name}")
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(path).to("cpu");
        mdl.eval()
        total_l, total_t = 0.0, 0
        for rec in load_dataset("json", data_files={"test": str(test_file)})["test"]:
            inp, tgt = rec["input"], rec["target"]
            full = inp + tgt
            enc = tok(full, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            lbl = enc.input_ids.clone();
            lbl[0, :len(inp)] = -100
            with torch.no_grad():
                out = mdl(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lbl)
            total_l += out.loss.item() * (lbl != -100).sum().item()
            total_t += (lbl != -100).sum().item()
        ppl = math.exp(total_l / total_t)
        typer.echo(f"   Perplexity: {ppl:.2f}")


# --- 5) SERVE FASTAPI + STREAMING DEMO ---
def start_server():
    app_fast = FastAPI()

    class SeqIn(BaseModel):
        sequence: str

    class SeqOut(BaseModel):
        resolved: str

    tokenizer = AutoTokenizer.from_pretrained("online_ckpts/final", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("online_ckpts/final").to("cpu")
    model.eval()

    def resolve_sequence(seq: str) -> str:
        enc = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        out = model.generate(**enc, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    @app_fast.post("/resolve", response_model=SeqOut)
    def resolve_endpoint(req: SeqIn):
        return SeqOut(resolved=resolve_sequence(req.sequence))

    # streaming demo River
    def worker():
        for idx, (x, _) in enumerate(stream.iter_csv("../../data/train.csv", target="target", converters={"input": str})):
            if idx % 20 == 0:
                print("→", resolve_sequence(x["input"]))
            time.sleep(0.1)

    threading.Thread(target=worker, daemon=True).start()

    uvicorn.run(app_fast, host="0.0.0.0", port=8000)


@app.command()
def serve():
    """Avvia FastAPI + demo streaming."""
    start_server()


if __name__ == "__main__":
    app()
