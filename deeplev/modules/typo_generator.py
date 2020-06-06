import random
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class TypoGenerator(ABC):
    @abstractmethod
    def generate_typo(self, sequence: str) -> str:
        pass


class Sequential(TypoGenerator):
    def __init__(self, *typo_generators: TypoGenerator) -> None:
        self.typo_generators = typo_generators

    def generate_typo(self, sequence: str) -> str:
        for typo_generator in self.typo_generators:
            sequence = typo_generator.generate_typo(sequence)
        return sequence


class SymbolRemover(TypoGenerator):
    def generate_typo(self, sequence: str) -> str:
        if not sequence:
            return sequence
        remove_idx = random.randint(0, len(sequence) - 1)
        return sequence[:remove_idx] + sequence[remove_idx + 1 :]


class SymbolReplacer(TypoGenerator):
    def __init__(self, vocab: Sequence[str]) -> None:
        self._vocab = vocab

    def generate_typo(self, sequence: str) -> str:
        if not sequence:
            return sequence
        replace_idx = random.randint(0, len(sequence) - 1)
        replce_with = random.sample(self._vocab, k=1)[0]
        return sequence[:replace_idx] + replce_with + sequence[replace_idx + 1 :]


class SymbolInserter(TypoGenerator):
    def __init__(self, vocab: Sequence[str]) -> None:
        self._vocab = vocab

    def generate_typo(self, sequence: str) -> str:
        if not sequence:
            return sequence
        insert_idx = random.randint(0, len(sequence) - 1)
        insert_symbol = random.sample(self._vocab, k=1)[0]
        return sequence[:insert_idx] + insert_symbol + sequence[insert_idx:]


def generate_default_typo(sequence: str, vocab: Sequence[str]) -> str:
    sequence_length = len(sequence)
    typo_generators = np.random.choice(
        [SymbolRemover(), SymbolReplacer(vocab), SymbolInserter(vocab)],
        size=random.randint(1, sequence_length * 2),
    )
    typo_generator = Sequential(*typo_generators)
    generated_sequence = typo_generator.generate_typo(sequence)
    # TODO: can be blank
    return generated_sequence
