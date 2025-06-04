import random
from typing import Iterator, Dict

def drift_stream(
    ref_seq: str,
    ref_len: int,
    settings
) -> Iterator[Dict[str,str]]:
    """
    Genera un flusso con concept drift simulato:
     - primi DRIFT_POINT campioni con P_AMBIG_INITIAL
     - dopo con P_AMBIG_FINAL
    """
    for i in range(settings.TOTAL_STEPS):
        p = (
            settings.P_AMBIG_INITIAL
            if i < settings.DRIFT_POINT
            else settings.P_AMBIG_FINAL
        )
        start = random.randint(0, ref_len - settings.FRAG_LENGTH)
        frag = ref_seq[start: start + settings.FRAG_LENGTH]
        frag_amb = "".join(
            random.choice(settings.AMBIG) if random.random() < p else nuc
            for nuc in frag
        )
        yield {"input": frag_amb, "target": frag}



def seq_to_kmers(seq: str, k: int, stride: int = 1) -> str:
    """Sliding window k-mer conversion, restituisce 'KM1 KM2 KM3…'."""
    kmers = [seq[i:i+k] for i in range(0, len(seq)-k+1, stride)]
    return " ".join(kmers)
