from typing import Sequence, Optional
from abc import ABC, abstractmethod
import random

from deeplev.utils import clean_sequence
from deeplev.typo_generator import generate_default_typo


class BaseSampler(ABC):
    def __init__(self, texts: Sequence[str]) -> None:
        self._texts = texts

    @abstractmethod
    def sample_positive(self, text: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def sample_negative(self, text: Optional[str] = None) -> str:
        pass


class TypoSampler(BaseSampler):
    def __init__(self, texts: Sequence[str]) -> None:
        texts = [clean_sequence(x) for x in texts]
        super().__init__(texts)

        self._vocab = list(set(''.join(texts)))

    def sample_positive(self, text: Optional[str] = None) -> str:
        return generate_default_typo(text, self._vocab)

    def sample_negative(self, text: Optional[str] = None) -> str:
        return random.sample(self._texts, 1)[0]
