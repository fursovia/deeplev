from abc import ABC, abstractmethod
import numpy as np
from typing import List


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
        remove_idx = np.random.randint(0, len(sequence))
        return sequence[:remove_idx] + sequence[remove_idx + 1:]


class SymbolReplacer(TypoGenerator):
    def __init__(self, vocab: List[str]) -> None:
        self._vocab = vocab

    def generate_typo(self, sequence: str) -> str:
        if not sequence:
            return sequence
        replace_idx = np.random.randint(0, len(sequence))
        replce_with = np.random.choice(self._vocab)
        return sequence[:replace_idx] + replce_with + sequence[replace_idx + 1:]


class SymbolInserter(TypoGenerator):
    def __init__(self, vocab: List[str]) -> None:
        self._vocab = vocab

    def generate_typo(self, sequence: str) -> str:
        if not sequence:
            return sequence
        insert_idx = np.random.randint(0, len(sequence))
        insert_symbol = np.random.choice(self._vocab)
        return sequence[:insert_idx] + insert_symbol + sequence[insert_idx:]


def generate_default_typo(sequence: str, vocab: List[str]) -> str:
    sequence_length = len(sequence)
    typo_generators = np.random.choice(
        [SymbolRemover(), SymbolReplacer(vocab), SymbolInserter(vocab)],
        size=np.random.randint(1, sequence_length * 2)
    )
    typo_generator = Sequential(*typo_generators)
    generated_sequence = typo_generator.generate_typo(sequence)
    # TODO: can be blank
    return generated_sequence
