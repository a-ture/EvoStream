

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import torch
from Levenshtein import distance as levenshtein_distance
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from src.config.setting import settings

# ─── DEVICE & WARNINGS ──────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", message=".*is_decoder.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")

# ─── 1. SELF‑CONTAINED MASK / PREDICTION UTILITIES ──────────────────────────
K = settings.KMER_SIZE  # normalmente 6


def _mask_kmers(seq: str, tok) -> str:
    """Sostituisce con [MASK] ogni k‑mer che contiene almeno una base ambigua."""
    kmers = [seq[i:i + K] for i in range(0, len(seq) - K + 1)]
    return " ".join(tok.mask_token if any(c in settings.AMBIG for c in km) else km for km in kmers)


def _resolve_batch_mask(seq: str, tok, mdl) -> str:
    """Restituisce la sequenza con le ambiguità riempite a livello char‑level."""
    masked = _mask_kmers(seq, tok)
    enc = tok(masked, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits[0]  # [Ltok,V]
    preds = tok.convert_ids_to_tokens(logits.argmax(-1))

    out = list(seq)
    for tok_idx, (inp_tok, pred_tok) in enumerate(zip(masked.split(), preds)):
        if inp_tok == tok.mask_token:
            for off, ch in enumerate(pred_tok):
                pos = tok_idx + off
                if pos < len(out) and out[pos] in settings.AMBIG:
                    out[pos] = ch
    return "".join(out)


# ─── 2. COLLATE PER DATALOADER ──────────────────────────────────────────────

def _collate(batch: list[dict], tok):
    inputs = [r["input"] for r in batch]
    targets = [r["target"] for r in batch]
    enc = tok(inputs, return_tensors="pt", padding=True,
              truncation=True, max_length=settings.MAX_LENGTH)
    enc["targets"] = targets  # keep plain targets
    return enc


# ─── 3. FUNZIONE PRINCIPALE DI VALUTAZIONE ──────────────────────────────────

def eval_file(run_name: str,
              ckpt_dir: str | Path | None,
              test_file: str | Path,
              mode: str = "finetuned",
              batch_size: int = 64):
    print(f"▶️  Valutazione '{run_name}' – modalità {mode} – su {test_file}")

    # --- carica tokenizer & modello ----------------------------------------
    if mode == "finetuned":
        tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        cfg = AutoConfig.from_pretrained(ckpt_dir, is_decoder=True)
        mdl = AutoModelForMaskedLM.from_pretrained(ckpt_dir, config=cfg).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(settings.MODEL_ID, use_fast=True)
        cfg = AutoConfig.from_pretrained(settings.MODEL_ID, is_decoder=True)
        mdl = AutoModelForMaskedLM.from_pretrained(settings.MODEL_ID, config=cfg).to(device)
    mdl.eval()

    # --- dataset & dataloader ----------------------------------------------
    ds = load_dataset("json", data_files={"test": str(test_file)})["test"]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: _collate(b, tok))

    # --- metriche -----------------------------------------------------------
    amb_tp = amb_fp = amb_fn = amb_total = 0
    seq_exact = edit_sum = total_seqs = 0

    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            inp_str = tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
            tgt_str = batch["targets"]

            for inp, tgt in zip(inp_str, tgt_str):
                total_seqs += 1
                pred = _random_or_predict(inp, tok, mdl, mode)

                amb_idx = [i for i, c in enumerate(inp) if c in settings.AMBIG]
                amb_total += len(amb_idx)
                for i in amb_idx:
                    if i < len(pred) and i < len(tgt) and pred[i] == tgt[i]:
                        amb_tp += 1
                    elif i < len(pred):
                        amb_fp += 1
                    else:
                        amb_fn += 1
                seq_exact += int(pred == tgt)
                edit_sum += levenshtein_distance(pred, tgt)

    _print_metrics(amb_tp, amb_fp, amb_fn, amb_total,
                   seq_exact, total_seqs, edit_sum)


# helper per scegliere la modalità di predizione

def _random_or_predict(inp: str, tok, mdl, mode: str) -> str:
    if mode == "random":
        return "".join(random.choice("ATCG") if c in settings.AMBIG else c for c in inp)
    else:
        return _resolve_batch_mask(inp, tok, mdl)


# ─── 4. METRICHE ────────────────────────────────────────────────────────────

def _print_metrics(tp, fp, fn, total_amb, seq_exact, total_seqs, edit_sum):
    amb_acc = tp / total_amb * 100 if total_amb else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    exact = seq_exact / total_seqs * 100
    avg_edit = edit_sum / total_seqs

    print(f"   • Accuracy ambiguità : {amb_acc:.2f}%")
    print(f"   • Precision ambiguità: {precision:.2f}%")
    print(f"   • Recall ambiguità   : {recall:.2f}%")
    print(f"   • F1 ambiguità       : {f1:.2f}%")
    print(f"   • Exact‑Match Rate   : {exact:.2f}%")
    print(f"   • Edit‑Distance med. : {avg_edit:.2f}\n")


# ─── 5. CLI -----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Valuta un checkpoint DNABERT/GROVER (batch)")
    ap.add_argument("--ckpt", required=True, help="Directory checkpoint")
    ap.add_argument("--test", required=True, help="File JSONL di test")
    ap.add_argument("--mode", choices=["finetuned", "zero-shot", "random"], default="finetuned")
    ap.add_argument("--bs", type=int, default=64, help="Batch size")
    args = ap.parse_args()

    eval_file("cli_run", ckpt_dir=args.ckpt, test_file=args.test,
              mode=args.mode, batch_size=args.bs)
