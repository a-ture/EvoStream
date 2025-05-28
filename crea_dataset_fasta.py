#!/usr/bin/env python3
import random
import json
from pathlib import Path
from Bio import SeqIO

# --- PARAMETRI ---
FASTA_FILE      = "ecoli_ref.fasta"  # il tuo file FASTA
OUTPUT_DIR      = Path(".")          # cartella di output
FRAG_LENGTH     = 100                # lunghezza di ciascun frammento
N_SAMPLES       = 1000               # numero totale di esempi da generare
P_AMBIG         = 0.05               # probabilità di sostituire con un'ambiguità

# split ratios
TRAIN_RATIO     = 0.8
VAL_RATIO       = 0.1
TEST_RATIO      = 0.1

# codici IUPAC
BASES = ['A','T','C','G']
AMBIG = ['N','R','Y','S','W','K','M','B','D','H','V']

# --- FUNZIONI ---
def inserisci_ambiguita(seq: str, p: float) -> str:
    """Introduce ambiguità in ciascuna posizione con probabilità p."""
    out = []
    for nuc in seq:
        if random.random() < p:
            out.append(random.choice(AMBIG))
        else:
            out.append(nuc)
    return "".join(out)

# --- CARICAMENTO FASTA ---
print(f"Carico sequenza da {FASTA_FILE}…")
record = SeqIO.read(FASTA_FILE, "fasta")
ref_seq = str(record.seq)
L = len(ref_seq)
print(f"Sequenza di lunghezza {L} bp letta.")

# --- PREPARAZIONE OUTPUT ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_f = open(OUTPUT_DIR / "train.jsonl", "w")
val_f   = open(OUTPUT_DIR / "val.jsonl",   "w")
test_f  = open(OUTPUT_DIR / "test.jsonl",  "w")

# --- GENERAZIONE DATI ---
print(f"Genero {N_SAMPLES} frammenti di lunghezza {FRAG_LENGTH}…")
for i in range(N_SAMPLES):
    # Estrai frammento casuale
    start = random.randint(0, L - FRAG_LENGTH)
    frag  = ref_seq[start:start + FRAG_LENGTH]
    # Introduci ambiguità
    frag_amb = inserisci_ambiguita(frag, P_AMBIG)
    rec = {"input": frag_amb, "target": frag}

    # Assegna alla split corretta
    r = random.random()
    if r < TRAIN_RATIO:
        train_f.write(json.dumps(rec) + "\n")
    elif r < TRAIN_RATIO + VAL_RATIO:
        val_f.write(json.dumps(rec) + "\n")
    else:
        test_f.write(json.dumps(rec) + "\n")

# --- CHIUSURA FILE ---
train_f.close()
val_f.close()
test_f.close()

print("Dataset creato:")
print(f"  • train.jsonl      ({int(N_SAMPLES*TRAIN_RATIO)} esempi ~{TRAIN_RATIO*100:.0f}%)")
print(f"  • val.jsonl        ({int(N_SAMPLES*VAL_RATIO)} esempi ~{VAL_RATIO*100:.0f}%)")
print(f"  • test.jsonl       ({int(N_SAMPLES*TEST_RATIO)} esempi ~{TEST_RATIO*100:.0f}%)")
