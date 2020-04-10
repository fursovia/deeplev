import abc
from typing import Sequence


class IPredictor(abc.ABC):
    @abc.abstractmethod
    def fit(self, texts: Sequence[str]):
        ...

    @abc.abstractmethod
    def get_similar_texts(self, text: str) -> Sequence[str]:
        ...
