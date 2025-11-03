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

    def encode(self, text: str, add_bos=False, add_eos=False):
        bas_id = None if self.specials.BOS is None else (
            len(self.vocab.itob) if self.specials.BOS is None else self.specials.BOS)
        eos_id = None if self.specials.EOS is None else (
            len(self.vocab.itob)+1 if self.specials.EOS is None else self.specials.EOS)
        return encode_text(self.vocab, text, add_bos=add_bos, add_eos=add_eos, bos_id=self.specials.BOS, eos_id=self.specials.EOS)

    def decode(self, ids):
        return decode(self.vocab, ids)
