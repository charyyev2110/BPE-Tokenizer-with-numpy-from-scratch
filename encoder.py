import numpy as np

from vocab import Vocab
from train import merge_seq


def encode(vocab: Vocab, b: bytes) -> np.ndarray:
    seq = np.frombuffer(b, dtype=np.uint8).astype(np.int64)

    rank = vocab.rank
    if not rank:
        return seq

    while seq.size >= 2:
        best = None
        best_rank = float('inf')
        # best_rank = 10**9
        left, right = seq[:-1], seq[1:]
        for a, c in zip(left.tolist(), right.tolist()):
            r = rank.get((a, c))
            if r is not None and r < best_rank:
                best_rank = r
                best = (a, c)

        if best is None:
            break
        a, c = best
        merged_bytes = vocab.itob[a] + vocab.itob[c]
        new_id = vocab.btoi.get(merged_bytes)
        if new_id is None:
            break

        seq = merge_seq(seq, a, c, new_id)

    return seq


def encode_text(vocab: Vocab, text: str, add_bos=None, add_eos=None, bos_id=None, eos_id=None) -> np.ndarray:
    seq = encode(vocab, text.encode('utf-8'))
    if add_bos and bos_id is not None:
        seq = np.concatenate([np.array([bos_id], dtype=np.int64), seq])
    if add_eos and eos_id is not None:
        seq = np.concatenate([seq, np.array([eos_id], dtype=np.int64)])
    return seq


def decode(vocab, ids) -> str:
    bs = b"".join(vocab.itob[int(i)] for i in ids if int(i) < len(vocab.itob))
    return bs.decode('utf-8', errors='replace')
