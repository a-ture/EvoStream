import csv
import json

with open("train.jsonl") as fin, open("train.csv", "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["input", "target"])
    for line in fin:
        rec = json.loads(line)
        writer.writerow([rec["input"], rec["target"]])
print("✅ train.csv creato.")
