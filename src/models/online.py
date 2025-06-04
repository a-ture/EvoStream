#!/usr/bin/env python3
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    set_seed, get_linear_schedule_with_warmup
)
from river import stream
import torch

from src.config.setting import settings
from src.streaming.streaming_trainer import StreamingTrainer

def main():
    # 1) Setup
    set_seed(settings.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_ID, use_fast=True
    )
    model     = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID, is_decoder=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=settings.WARMUP_STEPS,
        num_training_steps=settings.TOTAL_STEPS
    )

    trainer = StreamingTrainer(model, tokenizer, optimizer, scheduler, settings)
    trainer.model.train()

    # 2) Stream loop
    stream_iter = stream.iter_csv(
        settings.CSV_FILE,
        target="target",
        converters={"input": str}
    )
    for idx, (x, y) in enumerate(stream_iter, start=1):
        loss = trainer.step((x["input"], y))
        if loss is not None and idx % settings.PRINT_EVERY == 0:
            print(f"[Step {idx}] Loss: {loss:.4f}")
        if idx % settings.SAVE_EVERY == 0:
            trainer.save_checkpoint(idx)

    # 3) Salvataggio finale
    trainer.save_checkpoint("final")
    print("✅ Online fine-tuning completato.")

if __name__ == "__main__":
    main()
