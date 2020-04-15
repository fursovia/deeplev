from typing import Sequence
from abc import ABC, abstractmethod
import random

from deeplev.utils import clean_sequence
from deeplev.typo_generator import generate_default_typo


class BaseSampler(ABC):
    def __init__(self, texts: Sequence[str]) -> None:
        self._texts = texts

    @abstractmethod
    def sample_positive(self, text: str) -> str:
        pass

    @abstractmethod
    def sample_negative(self, text: str) -> str:
        pass


class TypoSampler(BaseSampler):
    def __init__(self, texts: Sequence[str]) -> None:
        texts = [clean_sequence(x) for x in texts]
        super().__init__(texts)

        self._vocab = list(set("".join(texts)))

    def sample_positive(self, text: str) -> str:
        return generate_default_typo(text, self._vocab)

    def sample_negative(self, text: str) -> str:
        return random.sample(self._texts, 1)[0]
