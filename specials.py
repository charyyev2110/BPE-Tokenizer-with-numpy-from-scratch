from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Specials:
    PAD: Optional[int] = None
    BOS: Optional[int] = None
    EOS: Optional[int] = None
