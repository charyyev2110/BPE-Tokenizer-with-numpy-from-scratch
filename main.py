import json
import pickle

from init import BPETokenizer, Specials
from data import GetData

VOCAB = 1000


# when you train
# tok = BPETokenizer(vocab_size=VOCAB, specials=Specials(
#     PAD=256, BOS=257, EOS=258))

# data = GetData('./input.txt').read_data()
# tok.train([data])
# tok.save('./saved')


# when you test, loads from saved vocab and merges
tok = BPETokenizer.load(vocab_size=1000, specials=Specials(
    PAD=256, BOS=257, EOS=258), path='./saved')

# encode -> batch -> decode
ids = tok.encode('testing bpe tokenizer', add_bos=True, add_eos=True)
print(f'\nids: {ids[:24]}')
