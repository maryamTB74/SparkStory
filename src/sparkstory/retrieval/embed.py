"""The single seam through which every embedding model is created.

The sibling of ``models/get_model.py``, and a separate seam rather than a widened
one. A chat model takes messages and binds an output schema; an embedder takes
strings and returns vectors. Nothing is shared but the idea of a registry, so
sharing a factory would only mean branching on which kind an entry is.

Three things live behind this seam, and they are the reason it exists:

* **Lazy, cached weight loading.** Importing this module reaches no network.
  Weights load on first use and stay loaded, because a per-request load would
  cost seconds on the cheapest path in the system.
* **Normalisation.** Every vector returned here is unit length, so a dot product
  *is* a cosine and no caller has to remember whether it already normalised.
* **The test seam.** ``FakeEmbedder`` is what keeps "no network required" true
  for the whole retrieval layer.
"""

import hashlib
import re
from typing import Any, Protocol

import numpy as np

from sparkstory.config import settings
from sparkstory.retrieval.exceptions import UnknownEmbeddingModelError
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Loaded models, keyed by registry name. Populated on first use, never cleared:
#: weights are immutable and a reload costs seconds.
_CACHE: dict[str, Any] = {}


def _unit(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone.

    An all-zero row is what an empty or punctuation-only chunk produces. Dividing
    it by its norm yields NaN, which then silently poisons every similarity score
    it touches -- so the zero row is preserved instead, and simply never matches.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class Embedder(Protocol):
    """What retrieval needs from an embedding model.

    A ``Protocol`` rather than an ABC, and unusually for this codebase it is
    justified: there are genuinely two implementations from day one -- the real
    model and the fake -- and the fake must not inherit weight-loading it does not
    have.

    Both methods return unit-length vectors. ``embed_query`` exists separately
    from ``embed_texts`` because some models embed a question differently from a
    document; for a static model the two are identical, and keeping them separate
    means swapping in an asymmetric model later touches nothing else.
    """

    dimensions: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed documents. Returns one unit-length row per text."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query. Returns a single unit-length vector."""
        ...


class StaticEmbedder:
    """A model2vec static model: distilled embeddings, numpy at inference time."""

    def __init__(self, model: Any, dimensions: int) -> None:
        self._model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        matrix = np.asarray(self._model.encode(texts), dtype=np.float32)
        return _unit(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class FakeEmbedder:
    """Deterministic bag-of-words vectors. No weights, no network.

    Words are hashed into buckets and counted, so texts sharing words score higher
    than texts that do not. That lexical structure is the point: it lets the store
    and fusion tests assert on *ranking* offline. Hashing the whole string instead
    would make every vector orthogonal, and those tests could then only check
    exact matches -- which is the shape of test that passes while retrieval is
    quietly broken.

    It is not a semantic model and must not be used to judge retrieval quality.
    That is what the ``corpus``-marked eval set is for.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            # sha256 rather than `hash()`, which is salted per process: a salted
            # bucket would make an index unsearchable by the next process.
            #
            # And rather than the word's own bytes, which was the first attempt
            # and was quietly broken -- `int.from_bytes` is big-endian, so any
            # word under 8 bytes carried trailing zeros and `% dimensions`
            # collapsed it to bucket 0. Almost every word hashed to the same
            # place, and a ranking test in test_store.py is what caught it.
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:8]) % self.dimensions] += 1.0
        return vector

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return _unit(np.vstack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _unit(self._vector(text))


def get_embedder(model_id: str) -> Embedder:
    """Build an embedder from the registry, or return the cached one.

    Args:
        model_id: A key of ``settings.embedding_configs``. Callers pass
            ``settings.embedding_model`` rather than a literal, so swapping the
            model stays a config change.

    Returns:
        An :class:`Embedder`.

    Raises:
        UnknownEmbeddingModelError: ``model_id`` is not in the registry. The
            message lists the ids that are, because this is almost always a typo
            in ``EMBEDDING_MODEL``.
    """
    if model_id in _CACHE:
        return _CACHE[model_id]

    try:
        config = settings.embedding_configs[model_id]
    except KeyError:
        known = ", ".join(sorted(settings.embedding_configs))
        raise UnknownEmbeddingModelError(
            f"Unknown embedding model {model_id!r}. Known models: {known}. "
            "Add an entry to Settings.embedding_configs, or fix EMBEDDING_MODEL "
            "in your .env."
        ) from None

    # Imported here, not at module scope: model2vec pulls in tokenizers and
    # huggingface-hub, and importing the package must stay cheap enough that the
    # MCP server starts instantly and the offline test suite never touches them.
    from model2vec import StaticModel

    logger.info("Loading embedding model %r", model_id)
    model = StaticModel.from_pretrained(config["identifier"])
    embedder = StaticEmbedder(model=model, dimensions=int(config["dimensions"]))
    _CACHE[model_id] = embedder
    return embedder
