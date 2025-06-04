#!/usr/bin/env python3
import random
import json
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

from src.config.setting import settings


# --- Funzione per costruire il reverse complement (rimane identica) ---
def reverse_complement(s: str) -> str:
    return str(Seq(s).reverse_complement())


# --- Mappa IUPAC “reale” per ogni base canonica -> codici che la contengono ---
IUPAC_MAP = {
    "A": ["R", "W", "M", "D", "H", "V", "N"],
    "C": ["Y", "S", "M", "B", "H", "V", "N"],
    "G": ["R", "S", "K", "B", "D", "V", "N"],
    "T": ["Y", "W", "K", "B", "D", "H", "N"],
}


def inserisci_ambiguita_realistica(seq: str, p: float) -> str:
    """
    Sostituisce in ciascuna posizione `base` della sequenza con un codice IUPAC
    contenente quella base con probabilità p, altrimenti lascia la base invariata.
    """
    out_chars = []
    for base in seq:
        if base not in IUPAC_MAP:
            # Se troviamo già un codice ambiguo (es. “N” o “R”), lo lasciamo invariato
            out_chars.append(base)
            continue

        if random.random() < p:
            possibili = IUPAC_MAP[base]
            c = random.choice(possibili)
            out_chars.append(c)
        else:
            out_chars.append(base)
    return "".join(out_chars)


def main():
    # --- 1) Caricamento FASTA ---
    print(f"Carico sequenza da {settings.FASTA_FILE}…")
    record = SeqIO.read(settings.FASTA_FILE, "fasta")
    ref_seq = str(record.seq)
    L = len(ref_seq)
    print(f"Sequenza di lunghezza {L} bp.")

    # --- 2) Sliding-window + reverse-complement ---
    windows = [
        ref_seq[i: i + settings.FRAG_LENGTH]
        for i in range(0, L - settings.FRAG_LENGTH + 1, settings.SLIDE_STRIDE)
    ]
    all_frags = []
    for w in windows:
        all_frags.append(w)  # frammento forward
        all_frags.append(reverse_complement(w))  # frammento reverse complement
    print(f"Totale frammenti (incl. RC): {len(all_frags)}")

    # --- 3) Prepara file di output ---
    out_dir = settings.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    f_train = open(out_dir / "train.jsonl", "w")
    f_val = open(out_dir / "val.jsonl", "w")
    f_test = open(out_dir / "test.jsonl", "w")

    # --- 4) Genera K versioni rumorose per ciascun frammento ---
    K = 1  # Numero di versioni con ambiguità diverse per ciascun frammento
    # → len(all_frags) * K esempi totali

    print(f"Generazione di {K} esempi rumorosi per ciascuno dei {len(all_frags)} frammenti…")
    total_examples = 0
    for frag in all_frags:
        for _ in range(K):
            # Scegliamo p uniformemente tra min e max definiti in settings
            p = random.uniform(settings.P_AMBIG_MIN, settings.P_AMBIG_MAX)

            # Generiamo la versione ambigua
            frag_amb = inserisci_ambiguita_realistica(frag, p)

            # Creiamo il record
            rec = {"input": frag_amb, "target": frag}

            # Assegniamo train / val / test in base alla percentuale
            r = random.random()
            if r < settings.TRAIN_RATIO:
                f_train.write(json.dumps(rec) + "\n")
            elif r < settings.TRAIN_RATIO + settings.VAL_RATIO:
                f_val.write(json.dumps(rec) + "\n")
            else:
                f_test.write(json.dumps(rec) + "\n")

            total_examples += 1

    f_train.close()
    f_val.close()
    f_test.close()

    # Numero di esempi totali effettivi (poco meno di len(all_frags)*K a cause delle assignazioni casuali)
    num_train = int(total_examples * settings.TRAIN_RATIO)
    num_val = int(total_examples * settings.VAL_RATIO)
    num_test = total_examples - num_train - num_val

    print("\nDataset creato:")
    print(f"  • train.jsonl: {num_train} esempi")
    print(f"  • val.jsonl:   {num_val} esempi")
    print(f"  • test.jsonl:  {num_test} esempi")
    print(f"  • Totale righe: {total_examples} esempi (≈ {len(all_frags)} frammenti × {K} versioni)")


if __name__ == "__main__":
    main()
