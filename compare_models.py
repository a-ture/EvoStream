# compare_both.py: valutazione senza data leakage
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Modelli da confrontare (salvati dopo training)
MODELS = {
    "batch":  "batch_output/final",
    "online": "river_ckpts/final"
}
TEST_FILE = "test.jsonl"
MAX_LEN   = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Controllo di eventuale overlap tra train e test
if __name__ == "__main__":
    # Quick check di overlap
    import json
    train_seqs = {json.loads(line)["input"] for line in open("train.jsonl")}
    test_seqs  = {json.loads(line)["input"] for line in open(TEST_FILE)}
    overlap = train_seqs & test_seqs
    if overlap:
        print(f"⚠️ ATENZIONE: {len(overlap)} sequenze duplicate tra train e test")
    else:
        print("✅ Nessun overlap train/test\n")

    # Funzione di valutazione
    def eval_model(name, path):
        print(f"▶️  Modello '{name}' — valutazione su {TEST_FILE}")
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(path).to(device)
        mdl.eval()

        total_loss = 0.0
        total_toks = 0
        amb_correct = 0
        amb_total = 0

        for rec in load_dataset("json", data_files={"test": TEST_FILE})["test"]:
            inp, tgt = rec["input"], rec["target"]
            # individua posizioni di codice IUPAC ambigui
            amb_idx = [i for i, c in enumerate(inp) if c in "NRYSWKMBVDH"]
            amb_total += len(amb_idx)

            # input + target e tokenizzazione
            full = inp + tgt
            enc = tok(full,
                      return_tensors="pt",
                      truncation=True,
                      padding="max_length",
                      max_length=MAX_LEN).to(device)

            # maschero la parte di input per il calcolo della loss
            labels = enc.input_ids.clone()
            labels[0, :len(inp)] = -100

            with torch.no_grad():
                out = mdl(input_ids=enc.input_ids,
                          attention_mask=enc.attention_mask,
                          labels=labels)
            # accumulo per perplexity
            loss = out.loss.item() * (labels != -100).sum().item()
            total_loss += loss
            total_toks += (labels != -100).sum().item()

            # generazione della sola parte target
            gen_ids = mdl.generate(
                **enc,
                max_new_tokens=len(tgt),
                pad_token_id=tok.eos_token_id
            )[0]
            pred = tok.decode(gen_ids, skip_special_tokens=True)[len(inp):]

            # accuracy sulle posizioni ambigue
            for i in amb_idx:
                if i < len(pred) and pred[i] == tgt[i]:
                    amb_correct += 1

        ppl = math.exp(total_loss / total_toks)
        amb_acc = amb_correct / amb_total * 100 if amb_total > 0 else 0
        print(f"   • Perplexity: {ppl:.2e}")
        print(f"   • Accuracy ambiguità: {amb_acc:.2f}%\n")

    # Esecuzione confronto
    for name, path in MODELS.items():
        eval_model(name, path)
