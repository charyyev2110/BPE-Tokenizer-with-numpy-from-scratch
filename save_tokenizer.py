import numpy as np


class SaveTokenizer:
    def __init__(self, path, vocab, specials):
        np.savez_compressed(
            path,
            itob=np.array(vocab.itob, dtype=object),
            rank_keys=np.array(list(vocab.rank.keys()), dtype=object),
            rank_vals=np.array(list(vocab.rank.values()), dtype=np.int64),
            specials=np.array([specials.PAD, specials.BOS,
                              specials.EOS], dtype=object)
        )
