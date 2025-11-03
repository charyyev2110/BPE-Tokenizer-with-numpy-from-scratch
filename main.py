import numpy as np

from init import BPETokenizer, Specials
from data import GetData

VOCAB = 32000

tok = BPETokenizer(vocab_size=VOCAB, specials=Specials(
    PAD=256, BOS=257, EOS=258))

data = GetData('./input.txt').read_data()
tok.train(data)

# encode -> batch -> decode
ids = tok.encode('testing bpe tokenizer', add_bos=True, add_eos=True)
print(f'ids: {ids[:24]}')
