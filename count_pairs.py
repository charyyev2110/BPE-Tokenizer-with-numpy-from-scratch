from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from collections import Counter


@dataclass(frozen=True)
class SpecialTokens:
    PAD: Optional[int] = None
    BOS: Optional[int] = None
    EOS: Optional[int] = None


class CountPairs:
    def __init__(self, vocab_size: int, specials: SpecialTokens = SpecialTokens()):
        if vocab_size < 256:
            vocab_size = 256
        self.vocab_size = int(vocab_size)
        self.specials = specials

        # converts ids to bytes
        self.itob: List[int] = [bytes([i]) for i in range(256)]
        # converts bytes back to ids
        self.btoi: Dict[bytes, int] = {bytes([i]): i for i in range(256)}

        self.rank: Dict[Tuple[int, int], int] = {}
        self._frozen = False
