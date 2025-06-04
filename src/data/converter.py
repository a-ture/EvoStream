import csv
import json

with open("../../data/train.jsonl") as fin, open("../../data/train.csv", "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["input", "target"])
    for line in fin:
        rec = json.loads(line)
        writer.writerow([rec["input"], rec["target"]])
print("✅ train.csv creato.")
