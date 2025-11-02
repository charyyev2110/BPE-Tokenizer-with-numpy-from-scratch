import numpy as np

from pathlib import Path


class GetData:
    def __init__(self, path: str):
        """
        Loads data
        Args:
            path: string,

        Returs:
            text data: string
        """
        self.p: str = Path(path)
        if self.p is None:
            raise FileNotFoundError(
                f"There is not file in proved file path: {self.p}")

    def read_data(self):
        """Reads the data from file"""
        with open(self.p, 'r', encoding='utf-8') as f:
            data = f.read()
            return data


# test = GetData('./input.txt')
# data = test.read_data()
# print(data[:100])
