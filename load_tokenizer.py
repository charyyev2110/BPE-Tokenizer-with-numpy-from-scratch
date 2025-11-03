import numpy as np

from typing import Tuple

from vocab import Vocab
from specials import Specials


class LoadTokenizer:
    def __init__(self, path: str, specials=Specials, vocab=Vocab()) -> None:
        self.specials = specials
        self.z = np.load(path, allow_pickle=True)
        self.vocab = vocab

    def run(self) -> Tuple[Vocab, Specials]:
        self.vocab.itob = list(self.z['itob'])
        self.vocab.btoi = {b: i for i, b in enumerate(self.vocab.itob)}
        keys = list(self.z['rank_keys'])
        vals = list(self.z['rank_vals'])
        self.vocab.rank = {tuple(k): int(v) for k, v in zip(keys, vals)}
        pad, bos, eos = self.z['specials'].tolist()
        specials = Specials(PAD=pad, BOS=bos, EOS=eos)
        return self.vocab, specials
