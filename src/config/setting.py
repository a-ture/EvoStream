from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Percorsi
    DATA_DIR: Path = Path("../../data")
    FASTA_FILE: Path = DATA_DIR / "ecoli_ref.fasta"
    OUTPUT_DIR: Path = Path("../outputs")
    CONFIGS_DIR: Path = Path("../../configs")
    CKPT_BATCH_DIR: Path = Path("../../checkpoints/batch_output")

    AMBIG: list[str] = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']

    # Streaming / training online (se non serve, ignoralo)
    CSV_FILE: Path = Path("../../data/train.csv")
    CKPT_ONLINE_DIR: Path = Path("../../checkpoints/online_ckpts")
    TOTAL_STEPS: int = 10000
    WARMUP_STEPS: int = 500
    REPLAY_BUFFER_SIZE: int = 5000
    GRAD_ACCUM_STEPS: int = 4

    # Dataset params (batch offline)
    FRAG_LENGTH: int = 100  # lunghezza di ciascun frammento
    SLIDE_STRIDE: int = 50  # passo tra finestre (puoi aumentarlo o ridurlo)
    N_SAMPLES: int = 500_000  # NON è più utilizzato nel nuovo approccio “all_frags × K”
    P_AMBIG_MIN: float = 0.30  # min probabilità di sostituire ogni base con IUPAC
    P_AMBIG_MAX: float = 0.70  # max probabilità

    TRAIN_RATIO: float = 0.5
    VAL_RATIO: float = 0.05
    TEST_RATIO: float = 0.005

    # File di split (userai questi Path nel batch script)
    TRAIN_FILE: Path = OUTPUT_DIR / "train.jsonl"
    VAL_FILE: Path = OUTPUT_DIR / "val.jsonl"
    TEST_FILE: Path = OUTPUT_DIR / "test.jsonl"

    # Parametri modello
    MODEL_ID: str = "PoetschLab/GROVER"
    SEED: int = 42
    MAX_LENGTH: int = 128
    NEW_TOKENS: int = 32
    STREAM_DELAY: float = 0.1
    PRINT_EVERY: int = 20
    LR: float = 5e-7
    WEIGHT_DECAY: float = 0.01
    SAVE_TOTAL_LIMIT: int = 5
    SAVE_EVERY: int = 200

    API_URL: str = "http://localhost:8000/resolve"

    # Altri parametri (li puoi ignorare se non usi streaming)
    TOTAL_SAMPLES: int = 1000
    DRIFT_POINT: int = 500
    P_AMBIG_INITIAL: float = 0.05
    P_AMBIG_FINAL: float = 0.20
    DRIFT_WINDOW_SIZE: int = 100

    DECAY_RATE: float = 0.9
    NUM_EPOCHS: int = 5000
    BATCH_SIZE: int = 128
    EVAL_STEPS: int = 10
    LOGGING_STEPS: int = 10
    KMER_SIZE: int = 6
    KMER_STRIDE: int = 1
    BATCH_EPOCHS:int = 3

settings = Settings()
