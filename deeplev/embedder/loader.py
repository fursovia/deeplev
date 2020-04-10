from typing import Dict, Callable

from deeplev.embedder.abstract import IEmbedder, EmbedderLike, EmbedderName
from deeplev.embedder.zero import ZeroEmbedder


def load_embedder(embedder_like: EmbedderLike) -> IEmbedder:
    if isinstance(embedder_like, IEmbedder):
        return embedder_like

    if not isinstance(embedder_like, str):
        raise ValueError(f'embedder_like must be str or IEmbedder.\nActual: {type(embedder_like)}')

    if embedder_like not in _EMBEDDER_MAPPING:
        raise ValueError(f'unknown embedder {embedder_like}')

    return _EMBEDDER_MAPPING[embedder_like]()


def _load_zero() -> ZeroEmbedder:
    return ZeroEmbedder(dimension=100)


_EMBEDDER_MAPPING: Dict[EmbedderName, Callable[[], IEmbedder]] = {
    EmbedderName.ZERO: _load_zero
}
