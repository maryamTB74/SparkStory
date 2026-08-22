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
from sparkstory.models.exceptions import MissingAPIKeyError
from sparkstory.retrieval.exceptions import (
    EmbeddingDimensionError,
    UnknownEmbeddingModelError,
)
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


#: How many texts one ``embed_content`` request may carry.
#:
#: Conservative on purpose. Ingest embeds the whole corpus, so the choice is
#: between one round trip per chunk (58 requests for the current corpus) and a
#: handful of batches. The endpoint does cap batch size, and a request rejected
#: for being too large costs a retry of everything in it -- so this sits well
#: under any documented limit rather than at it. Raising it is safe; the split is
#: order-preserving and covered by a test.
_MAX_BATCH = 32


class GeminiEmbedder:
    """A hosted Google embedding model, reached through ``google-genai``.

    The second real implementation of :class:`Embedder`, and the first that
    reaches the network. Three things it does that ``StaticEmbedder`` does not
    need to, each of which exists because a hosted model can fail in ways local
    weights cannot:

    * **Batches.** One request per chunk would be 58 round trips on ingest.
    * **Checks the returned width** against what the registry declared. See
      :class:`EmbeddingDimensionError` -- a wrong width is a configuration
      problem that would otherwise surface as a database error.
    * **Normalises anyway.** Google documents that ``gemini-embedding-2``
      auto-normalises truncated widths, and the older ``-001`` does not below
      3072. Relying on that would make this seam's guarantee a property of their
      model rather than of this module, and the cost of doing it ourselves is one
      division on an already-unit vector.

    The client is injected rather than constructed here, which is what lets every
    test above run offline against a stub. ``build_gemini_embedder`` is the seam
    that builds a real one.
    """

    def __init__(self, client: Any, identifier: str, dimensions: int) -> None:
        self._client = client
        self._identifier = identifier
        self.dimensions = dimensions
        # Query cache, added after a live run hit the free tier's 100
        # requests/minute. `make test-corpus` embeds the same 19 labelled queries
        # once per retriever per top_k -- 114 calls for 19 distinct strings, which
        # cannot pass in one quota window however long you wait.
        #
        # Safe because the input is a string and the output is a fixed vector for
        # a pinned model: re-embedding the same query is a paid no-op. Only
        # `embed_query` is cached, not `embed_texts` -- ingest sees each chunk
        # once, so caching there would grow without ever being read.
        #
        # Unbounded on purpose. The keys are search queries in one process, and a
        # server that has embedded enough distinct queries to matter has a much
        # larger bill than this dict has a memory cost.
        self._query_cache: dict[str, np.ndarray] = {}

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        # Imported here rather than at module scope for the reason model2vec is:
        # importing this module must reach no network and cost nothing, because
        # the MCP server imports it at startup and the offline suite imports it
        # in every retrieval test.
        from google.genai import types

        result = self._client.models.embed_content(
            model=self._identifier,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        matrix = np.asarray(
            [embedding.values for embedding in result.embeddings], dtype=np.float32
        )
        # Row count before width, because a short response is the more dangerous
        # of the two and it is not hypothetical: `gemini-embedding-2` returns one
        # embedding however many texts it is given, verified live. Stacking that
        # would attach vectors to the wrong chunks from the second batch onward
        # and produce an index that looks healthy and retrieves nonsense --
        # rule 22's failure, one layer out.
        if matrix.shape[0] != len(texts):
            raise EmbeddingDimensionError(
                f"{self._identifier!r} returned {matrix.shape[0]} embeddings for "
                f"{len(texts)} texts. Some models (notably gemini-embedding-2) "
                "embed only the first input regardless of how many are sent. Use "
                "a batching model such as 'gemini-embedding-001', or set "
                "_MAX_BATCH to 1 for this provider."
            )
        if matrix.shape[1] != self.dimensions:
            raise EmbeddingDimensionError(
                f"{self._identifier!r} returned {matrix.shape[1]}-dimensional "
                f"vectors, but the registry declares {self.dimensions}. The "
                "table name and pgvector column are both built from the declared "
                "width, so this cannot be stored. Fix `dimensions` in "
                "Settings.embedding_configs and re-ingest into the table that "
                "name produces."
            )
        return matrix

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            # Returned rather than requested: an empty batch is a legitimate
            # input (a file of only headings ingests to no chunks) and paying for
            # a request that embeds nothing is avoidable.
            return np.zeros((0, self.dimensions), dtype=np.float32)

        batches = [
            self._embed_batch(texts[start : start + _MAX_BATCH])
            for start in range(0, len(texts), _MAX_BATCH)
        ]
        # vstack in slice order, so row i is texts[i] however many batches it
        # took. A reordering here would attach every vector to the wrong chunk
        # and produce an index that looks healthy and retrieves nonsense.
        return _unit(np.vstack(batches))

    def embed_query(self, text: str) -> np.ndarray:
        cached = self._query_cache.get(text)
        if cached is not None:
            # Copied out, so a caller mutating the returned array in place cannot
            # corrupt every later search for the same query.
            return cached.copy()
        vector = self.embed_texts([text])[0]
        self._query_cache[text] = vector
        return vector.copy()


def build_gemini_embedder(
    config: dict[str, Any], model_id: str, api_key: str | None
) -> GeminiEmbedder:
    """Construct a :class:`GeminiEmbedder`, or explain what is missing.

    Split from ``get_embedder`` so the missing-key path is testable without a
    registry entry or a real client.

    Raises:
        MissingAPIKeyError: ``api_key`` is ``None``. The message names both the
            environment variable and the model that wanted it -- rule 21's
            lesson, now on its fifth stage. The memory extractor's version of
            this failed *open*, stored nothing, and the run looked normal; an
            embedder that failed open would produce books with no grounding and
            no complaint.
    """
    if api_key is None:
        env_var = config.get("api_key_env_var", "GOOGLE_API_KEY")
        raise MissingAPIKeyError(
            f"{env_var} is not set, and the embedding model {model_id!r} needs "
            f"it. Set {env_var} in your .env, or set EMBEDDING_MODEL to a local "
            "entry such as 'potion-base-8M', which needs no credential."
        )

    from google import genai

    return GeminiEmbedder(
        client=genai.Client(api_key=api_key),
        identifier=str(config["identifier"]),
        dimensions=int(config["dimensions"]),
    )


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

    # Dispatch on the registry's `provider` rather than on the shape of the
    # entry. Two construction paths now exist and they share nothing: local
    # weights load once into numpy, a hosted client needs a credential and makes
    # a request per batch. Reading it from the registry is what keeps adding a
    # third provider a config change.
    #
    # `.get` with a default rather than `[...]`, so an entry written before this
    # key existed still resolves to the local path it was written for.
    provider = str(config.get("provider", "model2vec"))

    embedder: Embedder
    if provider == "google":
        logger.info("Building hosted embedder %r", model_id)
        embedder = build_gemini_embedder(
            config=config,
            model_id=model_id,
            api_key=settings.api_key_for(str(config["api_key_env_var"])),
        )
    elif provider == "model2vec":
        # Imported here, not at module scope: model2vec pulls in tokenizers and
        # huggingface-hub, and importing the package must stay cheap enough that
        # the MCP server starts instantly and the offline test suite never
        # touches them.
        from model2vec import StaticModel

        logger.info("Loading embedding model %r", model_id)
        model = StaticModel.from_pretrained(config["identifier"])
        embedder = StaticEmbedder(model=model, dimensions=int(config["dimensions"]))
    else:
        raise UnknownEmbeddingModelError(
            f"Embedding model {model_id!r} declares provider {provider!r}, which "
            "has no builder. Known providers: google, model2vec."
        )

    _CACHE[model_id] = embedder
    return embedder
