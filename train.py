import numpy as np

import sys

from typing import Dict, List, Optional, Iterable, Tuple

from vocab import Vocab
from data import GetData

# text to byte


def ttob(text: Iterable[str]) -> list[np.ndarray]:
    seqs: list[np.ndarray] = [np.frombuffer(
        t.encode('utf-8'), dtype=np.uint8).astype(np.uint64) for t in text]
    # print(f'print(seqs[:10]): {print(seqs[:10])}')
    return seqs


def count_pairs(seqs: list[np.ndarray]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    # print(f"len(seqs): {len(seqs)}")
    for idx, s in enumerate(seqs):
        if s.size < 2:
            continue
        left, right = s[: -1], s[1:]
        pairs = np.stack([left, right], axis=1)
        # print('pairs stacked')
        for a, b in pairs:
            key = (int(a), int(b))
            counts[key] = counts.get(key, 0) + 1
            # print(counts)

    return counts


def best_pairs(counts: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], int]:
    if not counts:
        return None
    # argmax by frequency
    (a, b), f = max(counts.items(), key=lambda kv: kv[1])
    # print(f'(a, b), f: {((a, b), f)[:20]}')
    return (a, b), f


def merge_seq(seq, a, b, new_id):
    out = []
    i = 0
    L = int(seq.size)
    while i < L:
        if i + 1 < L and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(int(seq[i]))
            i += 1
    # print('out', out)
    return np.asarray(out, dtype=np.int64)


def train_bpe(vocab: Vocab, text: Iterable[str], max_vocab_size: int) -> None:
    # max_vocab_size = max(256, int(max_vocab_size))
    # seqs = ttob(text)

    total_merges = max_vocab_size - vocab.size
    seqs = ttob(text if not isinstance(text, str)else [text])
    rank_counter = 0
    while vocab.size < max_vocab_size:
        counts = count_pairs(seqs)
        bp = best_pairs(counts)
        if bp is None:
            break
        (a, b), freq = bp
        if freq < 2:
            break

        new_bytes = vocab.itob[a] + vocab.itob[b]
        new_id = vocab.add_symbol(new_bytes)
        if (a, b) not in vocab.rank:
            vocab.rank[(a, b)] = rank_counter
            rank_counter += 1

        # rewrite all sequences
        for i, s in enumerate(seqs):
            if s.size < 2:
                continue
            # skip if pair is not present
            if not ((s[:-1] == a) & (s[1:] == b)).any():
                continue
            seqs[i] = merge_seq(s, a, b, new_id)

        done = vocab.size - 256
        percent = min(1.0, done / total_merges)
        bar_len = 30
        filled = int(bar_len * percent)
        bar = "🟩" * filled + '-' * (bar_len - filled)
        sys.stdout.write(
            f"\rTraining BPE | {bar} | {percent*100:5.1f}% \nVocab: {vocab.size}/{max_vocab_size}")
        sys.stdout.flush()


# data = GetData('./input.txt').read_data()
# print(data[:100])

# vocab = Vocab()
# train = train_bpe(vocab, data, max_vocab_size=1000)
# raw = data.replace('\r\n', '\n')
# print(raw[:100])
# # print(data[:10])

# seqs = ttob([raw])
# print(f"seqs: {len(seqs[0])}")
# # print(f'len(seqs): {len(seqs)}')

# pairs = count_pairs(seqs)
# # print(f'pairs: {pairs[:10]}')
# best_pairs = best_pairs(pairs)
# print(f'best_pairs: {best_pairs[:10]}')
