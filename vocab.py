from typing import Dict, List, Tuple

from data import GetData


class Vocab:
    """Stores the tokens and bytes mapping and the learned ranks"""

    def __init__(self):
        # id -> bytes
        self.itob: List[bytes] = [bytes([i]) for i in range(256)]
        # bytes -> ids
        self.btoi: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
        # merge rankings (a, b) -> priority (lower to higher)
        self.rank: Dict[Tuple[int, int], int] = {}

    @property
    def size(self) -> int:
        # print(f"len(self.itob): {len(self.itob)}")
        return len(self.itob)

    def add_symbol(self, b: bytes) -> int:
        if b in self.btoi:
            return self.btoi[b]
        idx = len(self.itob)
        self.itob.append(b)
        self.btoi[b] = idx
        return idx


# get_data = GetData('./input.txt')
# data = get_data.read_data()
# vocab = Vocab()


# print(data[:10])
