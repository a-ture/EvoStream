import torch
from torch.cuda.amp import autocast, GradScaler
from collections import deque


class StreamingTrainer:
    def __init__(self, model, tokenizer, optimizer, scheduler, settings):
        # Dispositivo e modello
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Tokenizer, ottimizzatore, scheduler e settings
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.settings = settings

        # AMP scaler (per mixed‐precision, se desideri usarlo) e replay buffer
        self.scaler = GradScaler()
        self.buffer = deque(maxlen=settings.REPLAY_BUFFER_SIZE)

    def step(self, features: dict) -> float:
        """
        Riceve un dizionario `features` contenente già:
          - "input_ids":    Tensor di dimensione (1, seq_len) su device
          - "attention_mask": Tensor di dimensione (1, seq_len) su device
          - "labels":       Tensor di dimensione (1, seq_len) su device (–100 sui token di prompt/pad)

        Esegue un passo di training:
         1) forward sul modello causale (GROVER)
         2) backward (+ mixed precision tramite self.scaler)
         3) optimizer.step() + scheduler.step()
         4) zero_grad()
         5) ritorna il valore di loss.item()
        """
        self.model.train()

        # Forward + calcolo loss in autocast (mixed precision)
        with autocast():
            outputs = self.model(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
                labels=features["labels"]
            )
            loss = outputs.loss  # loss già “medio” sulla sequenza

        # Backward e aggiornamento pesi
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # (Opzionale) Salvo nel buffer i tensori per eventuale “replay” in caso di drift
        self.buffer.append(features)

        return loss.item()

    def save_checkpoint(self, idx: int | str) -> None:
        """
        Salva modello e tokenizer in due modalità:
          - se idx == "final", salvo in settings.CKPT_ONLINE_DIR/final
          - altrimenti, salvo in settings.CKPT_ONLINE_DIR/step_{idx}
        """
        if str(idx) == "final":
            ckpt_dir = self.settings.CKPT_ONLINE_DIR / "final"
        else:
            ckpt_dir = self.settings.CKPT_ONLINE_DIR / f"step_{idx}"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
