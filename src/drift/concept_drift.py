#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
concept_drift_grover.py – streaming + online fine‐tuning + concept-drift per un modello di Language Modeling causale (GROVER),
con valutazioni periodiche “out‐of‐sample” sulla validation set (val.jsonl) e test finale.

Struttura:
1) Preparazione tokenizer, modello e tokenizzazione di val.jsonl/test.jsonl per la valutazione.
2) Definizione di StreamingTrainer (presupposto già implementato in src/streaming/streaming_trainer.py).
3) Funzione streaming_with_concept_drift:
   - Passo per passo streaming + online fine‐tune
   - Rilevazione drift con ADWIN
   - Ogni DRIFT_WINDOW_SIZE passi: salvo checkpoint “final” e valuto su val.jsonl usando un Trainer dedicato
   - Alla fine: checkpoint “final” e valutazione su test.jsonl
4) main(): inizializza tutto e avvia streaming.
"""

import json
import random
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from river.drift import ADWIN
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer as HfTrainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from src.config.setting import settings
from src.streaming.streaming_trainer import StreamingTrainer


# ---------------------------------------------------------------------
# 0) dispositivo
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) modello + tokenizer GROVER
# ---------------------------------------------------------------------
MODEL_NAME = "PoetschLab/GROVER"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Aggiungo pad_token se mancante e riallineo gli embeddings
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------
# 2) funzione di supporto: tokenizzazione per causal LM
# ---------------------------------------------------------------------
def tokenize_for_causal(input_text: str, target_text: str, max_length: int):
    """
    Prende (input, target) e restituisce un dizionario con:
      • "input_ids"    : tensore 1×max_length
      • "attention_mask": tensore 1×max_length
      • "labels"       : tensore 1×max_length (–100 su prompt+pad)
    """
    full_text = f"Input: {input_text}\nOutput: {target_text}"
    tok_full = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tok_full["input_ids"][0]
    attention_mask = tok_full["attention_mask"][0]

    prompt_only = f"Input: {input_text}\nOutput:"
    tok_prompt = tokenizer(
        prompt_only,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_len = tok_prompt["input_ids"].shape[1]

    labels = input_ids.clone()
    for i in range(len(labels)):
        if i < prompt_len or input_ids[i] == tokenizer.pad_token_id:
            labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------
# 3) Calcola token_accuracy per HuggingFace Trainer
# ---------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Calcola l'accuratezza sui token (escludendo i -100).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    correct = (preds == labels) & mask
    total_tokens = mask.sum()
    correct_tokens = correct.sum()

    token_accuracy = float(correct_tokens) / float(total_tokens) if total_tokens > 0 else 0.0
    return {"token_accuracy": token_accuracy}


# ---------------------------------------------------------------------
# 4) Prepara validation + test tokenized (out‐of‐sample) una volta per tutte
# ---------------------------------------------------------------------
print("📦 Preparazione validation/test set tokenizzati…")
# Carico val.jsonl
raw_val = load_dataset("json", data_files=str(settings.DATA_DIR / "val.jsonl"))["train"]
# Carico test.jsonl
raw_test = load_dataset("json", data_files=str(settings.DATA_DIR / "test.jsonl"))["train"]

def preprocess_val_test(examples):
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for inp, tgt in zip(examples["input"], examples["target"]):
        tok = tokenize_for_causal(inp, tgt, settings.MAX_LENGTH)
        out["input_ids"].append(tok["input_ids"])
        out["attention_mask"].append(tok["attention_mask"])
        out["labels"].append(tok["labels"])
    return {
        "input_ids":      torch.stack(out["input_ids"]),
        "attention_mask": torch.stack(out["attention_mask"]),
        "labels":         torch.stack(out["labels"]),
    }

val_tokenized = raw_val.map(
    preprocess_val_test,
    batched=True,
    remove_columns=["input", "target"]
)
test_tokenized = raw_test.map(
    preprocess_val_test,
    batched=True,
    remove_columns=["input", "target"]
)

val_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
test_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ---------------------------------------------------------------------
# 5) funzione per creare lo StreamingTrainer (online fine‐tune)
# ---------------------------------------------------------------------
def setup_trainer_online():
    """
    Inizializza e ritorna un oggetto StreamingTrainer (che gestisce un singolo passo
    di ottimizzazione online). Imposta optimizer, scheduler e seed.
    """
    set_seed(settings.SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LR)
    total_steps = settings.TOTAL_STEPS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    trainer = StreamingTrainer(model, tokenizer, optimizer, scheduler, settings)
    trainer.model.train()
    return trainer


# ---------------------------------------------------------------------
# 6) generatore di streaming: legge train.jsonl riga per riga
# ---------------------------------------------------------------------
def drift_text_stream(jsonl_file: Path, settings):
    """
    Generatore che legge un file JSONL con righe {"input": str, "target": str},
    e restituisce iterativamente ciascun dizionario record.
    """
    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec


# ---------------------------------------------------------------------
# 7) loop principale: streaming + online fine‐tuning + concept drift + valutazioni
# ---------------------------------------------------------------------
def streaming_with_concept_drift(trainer: StreamingTrainer, stream_file: Path):
    """
    Esegue:
      1) Per TOTAL_STEPS esempi: streaming online fine‐tune
      2) Rilevazione drift con ADWIN (su errore binario per esempio)
      3) Ogni DRIFT_WINDOW_SIZE passi: salva checkpoint “final” e valuta su val.jsonl
      4) Alla fine: checkpoint finale e valutazione su test.jsonl

    Ritorna un dict con:
      - streaming_mean_loss      : media delle loss “in‐sample” (online)
      - streaming_token_accuracy : token_accuracy cumulata “in‐sample”
      - streaming_test_metrics   : (None, ma le metriche test vengono stampate da HfTrainer)
    """
    detector = ADWIN()
    window_buf = deque(maxlen=settings.DRIFT_WINDOW_SIZE)

    cum_correct_toks = 0
    cum_total_toks = 0
    streaming_losses = []
    err_count = 0

    stream_gen = drift_text_stream(stream_file, settings)
    for idx, rec in enumerate(
        tqdm(stream_gen, total=settings.TOTAL_STEPS, desc="🔄 Streaming & Online FT"),
        start=1,
    ):
        raw_inp = rec["input"]
        raw_tgt = rec["target"]

        # — 1) Tokenizzazione + forward per predizione “in‐sample”
        instance = tokenize_for_causal(raw_inp, raw_tgt, settings.MAX_LENGTH)
        input_ids = instance["input_ids"].unsqueeze(0).to(device)
        attention_mask = instance["attention_mask"].unsqueeze(0).to(device)
        labels = instance["labels"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)

        # — 2) Calcolo correttezza sui token di risposta per questo esempio
        mask = (labels != -100)
        total_toks = mask.sum().item()
        correct_toks = ((preds == labels) & mask).sum().item() if total_toks > 0 else 0

        cum_correct_toks += correct_toks
        cum_total_toks += total_toks

        # (opzionale) errore binario per ADWIN
        err = 0 if (correct_toks == total_toks) else 1
        err_count += err

        # — 3) Online‐fine‐tune tramite StreamingTrainer.step()
        loss = trainer.step({
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        })
        if loss is not None:
            streaming_losses.append(loss)

        # — 4) Log periodico ogni PRINT_EVERY passi
        if idx % settings.PRINT_EVERY == 0 and loss is not None:
            mean_loss = float(np.mean(streaming_losses))
            token_acc_cumulata = (cum_correct_toks / cum_total_toks) if cum_total_toks > 0 else 0.0
            tqdm.write(
                f"[Step {idx}] Loss online media: {mean_loss:.4f}  |  "
                f"Token‐Accuracy cumulata: {token_acc_cumulata*100:.2f}%"
            )

        # — 5) Concept drift detection (basato su err binario)
        if detector.update(err):
            tqdm.write(f"\n⚠️ DRIFT rilevato allo step {idx}, salvo checkpoint e resetto buffer…")
            # Decay del buffer
            trainer.buffer = deque(
                (r for r in trainer.buffer if random.random() < settings.DECAY_RATE),
                maxlen=trainer.buffer.maxlen,
            )
            # Salvo checkpoint “drift_{idx}”
            ck_drift = settings.CKPT_ONLINE_DIR / f"drift_{idx}"
            ck_drift.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ck_drift)
            tokenizer.save_pretrained(ck_drift)
            detector.reset()

        # — 6) Sliding‐window evaluation + validazione out‐of‐sample
        window_buf.append((raw_inp, raw_tgt))
        if idx % settings.DRIFT_WINDOW_SIZE == 0:
            # ➊: salva checkpoint “final” temporaneo
            ck_final = settings.CKPT_ONLINE_DIR / "final"
            ck_final.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ck_final)
            tokenizer.save_pretrained(ck_final)

            # (Opzionale) salva finestra corrente “in-sample”
            wf = settings.DATA_DIR / f"window_{idx}.jsonl"
            with wf.open("w", encoding="utf-8") as f_out:
                for a, b in window_buf:
                    f_out.write(json.dumps({"input": a, "target": b}) + "\n")
            window_buf.clear()

            # ➋: valutazione “out‐of‐sample” su val.jsonl con HfTrainer
            from transformers import Trainer as EvalTrainer, TrainingArguments as EvalArgs

            eval_model = AutoModelForCausalLM.from_pretrained(str(ck_final)).to(device)
            eval_args = EvalArgs(
                output_dir="eval_tmp",
                per_device_eval_batch_size=settings.BATCH_SIZE,
                do_train=False,
                do_eval=True,
                logging_strategy="no",
            )
            eval_trainer = EvalTrainer(
                model=eval_model,
                args=eval_args,
                eval_dataset=val_tokenized,
                tokenizer=tokenizer,
                data_collator=val_data_collator,
                compute_metrics=compute_metrics,
            )
            tqdm.write(f"🎯 Valutazione 'val_at_step_{idx}' – modalità finetuned – su {settings.DATA_DIR / 'val.jsonl'}")
            metrics = eval_trainer.evaluate()
            loss_val = metrics.get("eval_loss", None)
            tok_acc_val = metrics.get("eval_token_accuracy", None)
            perplexity_val = math.exp(loss_val) if loss_val is not None else None

            tqdm.write(
                f"▶️  [Step {idx}] Validation (out‐of‐sample) → "
                f"loss = {loss_val:.4f}, token_accuracy = {tok_acc_val*100:.2f}%, perplexity = {perplexity_val:.2f}"
            )

        # — 7) Checkpoint periodico ogni SAVE_EVERY passi
        if idx % settings.SAVE_EVERY == 0:
            ck_step = settings.CKPT_ONLINE_DIR / f"step_{idx}"
            ck_step.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ck_step)
            tokenizer.save_pretrained(ck_step)
            tqdm.write(f"✅ Checkpoint salvato: step_{idx}")

        # Interrompe se raggiunto TOTAL_STEPS
        if idx >= settings.TOTAL_STEPS:
            break

    # — 8) Checkpoint finale
    final_ck = settings.CKPT_ONLINE_DIR / "final"
    final_ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_ck)
    tokenizer.save_pretrained(final_ck)
    tqdm.write("✅ Streaming + Online FT completati.")

    # — 9) Valutazione finale “out‐of‐sample” su test.jsonl
    from transformers import Trainer as EvalTrainer, TrainingArguments as EvalArgs

    eval_model = AutoModelForCausalLM.from_pretrained(str(final_ck)).to(device)
    eval_args = EvalArgs(
        output_dir="eval_tmp",
        per_device_eval_batch_size=settings.BATCH_SIZE,
        do_train=False,
        do_eval=True,
        logging_strategy="no",
    )
    eval_trainer = EvalTrainer(
        model=eval_model,
        args=eval_args,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=test_data_collator,
        compute_metrics=compute_metrics,
    )
    tqdm.write(f"🎯 Valutazione test finale – modalità finetuned – su {settings.DATA_DIR / 'test.jsonl'}")
    test_metrics = eval_trainer.evaluate()
    loss_test = test_metrics.get("eval_loss", None)
    tok_acc_test = test_metrics.get("eval_token_accuracy", None)
    perplexity_test = math.exp(loss_test) if loss_test is not None else None

    tqdm.write(
        f"▶️  Test finale → loss = {loss_test:.4f}, "
        f"token_accuracy = {tok_acc_test*100:.2f}%, perplexity = {perplexity_test:.2f}"
    )

    # Calcolo metriche “in‐sample”
    mean_loss = float(np.mean(streaming_losses)) if streaming_losses else None
    token_acc_cumulata = (cum_correct_toks / cum_total_toks) if cum_total_toks > 0 else None

    return {
        "streaming_mean_loss":      mean_loss,
        "streaming_token_accuracy": token_acc_cumulata,
        "streaming_test_metrics":   test_metrics,
    }


# ---------------------------------------------------------------------
# 8) main: avvio streaming
# ---------------------------------------------------------------------
def main():
    # Assicuriamoci che le directory esistano
    for d in (settings.DATA_DIR, settings.CKPT_ONLINE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    trainer_online = setup_trainer_online()

    # Scegli il file JSONL di streaming (generalmente “train.jsonl”)
    stream_file = settings.DATA_DIR / "train.jsonl"
    print("\n🎯 Avvio streaming + online fine‐tuning (concept drift)…")
    streaming_results = streaming_with_concept_drift(trainer_online, stream_file)

    print("\n✅ Risultati finali (in‐sample e test out‐of‐sample):")
    print(f"   • streaming_mean_loss (in‐sample):      {streaming_results['streaming_mean_loss']:.4f}")
    print(f"   • streaming_token_accuracy (in‐sample): {streaming_results['streaming_token_accuracy']*100:.2f}%")
    # Le metriche “out‐of‐sample” su validation e test sono state già stampate da evaluate
    # streaming_results['streaming_test_metrics'] contiene il dict di test, se serve usarlo ulteriormente.


if __name__ == "__main__":
    set_seed(settings.SEED)
    main()
