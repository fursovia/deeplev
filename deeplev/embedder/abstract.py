import abc
import enum
from typing import Union

import numpy as np

Embedding = np.ndarray


class IEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, text: str) -> Embedding:
        ...


class EmbedderName(str, enum.Enum):
    ZERO = 'zero'


EmbedderLike = Union[IEmbedder, EmbedderName]
