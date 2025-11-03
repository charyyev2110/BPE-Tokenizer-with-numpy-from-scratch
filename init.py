import os
import pickle
import json
from typing import Iterable, Optional
from specials import Specials
from vocab import Vocab
from train import train_bpe
from encoder import encode_text
from encoder import decode


class BPETokenizer:
    def __init__(self, vocab_size: int, specials: Optional[Specials] = None):
        if vocab_size < 256:
            vocab_size = 256
        self.vocab_size = int(vocab_size)
        self.specials = specials or Specials()
        self.vocab = Vocab()
        self._frozen = False

    def train(self, texts: Iterable[str], merges: Optional[int] = None):
        target = max(self.vocab_size, 256)
        train_bpe(self.vocab, texts, max_vocab_size=target)
        self._frozen = True   # freeze merges

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        if not self._frozen:
            raise RuntimeError("Tokenizer not trained/frozen yet")
        bos_id = self.specials.BOS if self.specials.BOS is not None else None
        eos_id = self.specials.EOS if self.specials.EOS is not None else None
        return encode_text(self.vocab, text, add_bos=add_bos, add_eos=add_eos,
                           bos_id=bos_id, eos_id=eos_id)

        # print(encoded)

    def decode(self, ids):
        print('decode', decode(self.vocab, ids))
        return decode(self.vocab, ids)

    def save(self, path: str = './saved'):
        """Save vocab and merges to the file"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'vocab_itob.pkl'), 'wb') as f:
            pickle.dump(self.vocab.itob, f)

        with open(os.path.join(path, 'vocab_btoi.pkl'), 'wb') as f:
            pickle.dump(self.vocab.btoi, f)

        with open(os.path.join(path, 'vocab_rank.json'), 'w') as f:
            json.dump({str(k): v for k, v in self.vocab.rank.items()}, f)

        print(f'✅ Tokenizer saved to: {os.path.abspath(path)}')

    @classmethod
    def load(cls, vocab_size: int, specials, path: str = './saved'):
        """Load vocab and merges from a saved files"""
        tok = cls(vocab_size=vocab_size, specials=specials)

        with open(os.path.join(path, 'vocab_itob.pkl'), 'rb') as f:
            tok.vocab.itob = pickle.load(f)

        with open(os.path.join(path, 'vocab_btoi.pkl'), 'rb') as f:
            tok.vocab.btoi = pickle.load(f)

        with open(os.path.join(path, 'vocab_rank.json'), 'r') as f:
            data = json.load(f)
            tok.vocab.rank = ({eval(k): v for k, v in data.items()})

        tok._frozen = True
        print(f'✅ Tokenizer loaded from: {os.path.abspath(path)}')
        return tok
