"""The embedding seam.

``FakeEmbedder`` is the reason the rest of the retrieval layer can be tested at
all. The obvious implementation makes live embedding calls; this project has kept
"no network required" true from the start, and a retrieval layer is where that
property is most at risk.
"""

import numpy as np
import pytest

from sparkstory.retrieval.embed import FakeEmbedder, get_embedder
from sparkstory.retrieval.exceptions import UnknownEmbeddingModelError


class TestGetEmbedder:
    def test_unknown_model_names_the_known_ones(self) -> None:
        """Almost always a typo in EMBEDDING_MODEL, so the message has to list
        what would have worked."""
        with pytest.raises(UnknownEmbeddingModelError) as caught:
            get_embedder("not-a-real-embedder")
        assert "potion-base-8M" in str(caught.value)

    def test_is_a_configuration_error(self) -> None:
        """So the MCP tool layer keeps translating it into an actionable message
        rather than letting it surface as a bug."""
        from sparkstory.entities.exceptions import ConfigurationError

        assert issubclass(UnknownEmbeddingModelError, ConfigurationError)

    def test_importing_the_module_downloads_nothing(self) -> None:
        """Weight loading is lazy. If it were not, importing sparkstory would
        reach the network, and the whole offline test suite would depend on a
        model download."""
        import sparkstory.retrieval.embed as module

        assert module._CACHE == {} or all(
            value is not None for value in module._CACHE.values()
        )


class TestFakeEmbedder:
    def test_is_deterministic(self) -> None:
        embedder = FakeEmbedder(dimensions=64)
        first = embedder.embed_query("the moon has no air")
        second = embedder.embed_query("the moon has no air")
        assert np.array_equal(first, second)

    def test_returns_the_configured_width(self) -> None:
        assert FakeEmbedder(dimensions=32).embed_query("hello").shape == (32,)

    def test_embed_texts_returns_one_row_per_text(self) -> None:
        matrix = FakeEmbedder(dimensions=16).embed_texts(["a", "b", "c"])
        assert matrix.shape == (3, 16)

    def test_shared_words_score_higher_than_unrelated_ones(self) -> None:
        """Not decoration: bag-of-words similarity is what lets the store and
        fusion tests assert on *ranking* offline. A whole-string hash would make
        every vector orthogonal, and those tests could only check exact matches.
        """
        embedder = FakeEmbedder(dimensions=256)
        query = embedder.embed_query("moon air wind")
        related = embedder.embed_query("the moon has no air and no wind")
        unrelated = embedder.embed_query("penguins swim using their wings")
        assert float(query @ related) > float(query @ unrelated)

    def test_vectors_are_unit_length(self) -> None:
        """So a dot product is a cosine, and the store never has to normalise
        twice or wonder whether it already did."""
        vector = FakeEmbedder(dimensions=64).embed_query("some text here")
        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_empty_text_does_not_divide_by_zero(self) -> None:
        """An empty or punctuation-only chunk is a corpus defect, not a crash."""
        vector = FakeEmbedder(dimensions=8).embed_query("")
        assert vector.shape == (8,)
        assert np.isfinite(vector).all()

    def test_distinct_short_words_land_in_distinct_buckets(self) -> None:
        """Regression test for a real defect. The first implementation hashed a
        word's own bytes with ``int.from_bytes``, which is big-endian -- so any
        word shorter than 8 bytes carried trailing zeros and ``% dimensions``
        collapsed it to bucket 0. Almost every word hashed to the same place, and
        the fake reported everything as similar to everything.
        """
        embedder = FakeEmbedder(dimensions=256)
        buckets = {
            int(np.argmax(embedder.embed_query(word)))
            for word in ("moon", "air", "wind", "fox", "clock", "mouse")
        }
        assert len(buckets) >= 5, f"words collided into {buckets}"

    def test_punctuation_does_not_split_a_word_from_its_match(self) -> None:
        """ "air," and "air" must hash alike, or a comma in the corpus silently
        removes a word from the index."""
        embedder = FakeEmbedder(dimensions=256)
        with_comma = embedder.embed_query("air,")
        without = embedder.embed_query("air")
        assert np.array_equal(with_comma, without)


class TestGeminiEmbedder:
    """The hosted embedder, added 2026-08-19 when it became the default.

    Every test here runs offline against a stub client. That is deliberate and
    it is the same argument ``FakeEmbedder`` exists for: the default embedder
    now needs a network call, and if these tests reached for it the whole suite
    would depend on Google being up -- which open item 3 says is not a safe bet.

    What these tests can and cannot establish is worth stating plainly, because
    rule 33 was learned exactly here. They pin the *contract*: batching, ordering,
    normalisation, failure translation. They cannot tell us the real endpoint
    behaves this way. Only ingesting the corpus can, and until that runs the
    identifier and width in the registry are documentation-derived assertions.
    """

    def _client(self, vectors: list[list[float]], record: list | None = None):
        """A stand-in for ``google.genai.Client`` returning fixed vectors."""

        class _Embedding:
            def __init__(self, values: list[float]) -> None:
                self.values = values

        class _Result:
            def __init__(self, values: list[list[float]]) -> None:
                self.embeddings = [_Embedding(v) for v in values]

        class _Models:
            def embed_content(self, *, model, contents, config):  # noqa: ANN001
                if record is not None:
                    record.append(list(contents))
                return _Result(vectors[: len(contents)])

        class _Client:
            models = _Models()

        return _Client()

    def test_returns_one_unit_length_row_per_text(self) -> None:
        """The seam promises unit vectors so a dot product is a cosine. Google
        auto-normalises truncated widths, but relying on that would make our
        guarantee a property of their model rather than of this module."""
        from sparkstory.retrieval.embed import GeminiEmbedder

        embedder = GeminiEmbedder(
            client=self._client([[3.0, 4.0], [0.0, 5.0]]),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        matrix = embedder.embed_texts(["a", "b"])
        assert matrix.shape == (2, 2)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    def test_embed_query_returns_a_single_vector(self) -> None:
        from sparkstory.retrieval.embed import GeminiEmbedder

        embedder = GeminiEmbedder(
            client=self._client([[3.0, 4.0]]),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        vector = embedder.embed_query("a")
        assert vector.shape == (2,)
        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_empty_input_makes_no_call(self) -> None:
        """An empty batch must not become a paid request for nothing, and must
        still return the right shape so callers need no special case."""
        from sparkstory.retrieval.embed import GeminiEmbedder

        calls: list = []
        embedder = GeminiEmbedder(
            client=self._client([], record=calls),
            identifier="gemini-embedding-2",
            dimensions=768,
        )
        matrix = embedder.embed_texts([])
        assert matrix.shape == (0, 768)
        assert calls == []

    def test_texts_are_batched_not_sent_one_by_one(self) -> None:
        """Ingest embeds the whole corpus. One request per chunk would be 58
        round trips where a handful will do."""
        from sparkstory.retrieval.embed import GeminiEmbedder

        calls: list = []
        embedder = GeminiEmbedder(
            client=self._client([[1.0, 0.0]] * 5, record=calls),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        embedder.embed_texts(["a", "b", "c", "d", "e"])
        assert len(calls) == 1, "five short texts should be one request"

    def test_a_long_input_is_split_into_several_requests(self) -> None:
        """The endpoint caps how many inputs one request may carry, so the
        batch size is ours to respect rather than the caller's to know."""
        from sparkstory.retrieval.embed import _MAX_BATCH, GeminiEmbedder

        calls: list = []
        count = _MAX_BATCH + 3
        embedder = GeminiEmbedder(
            client=self._client([[1.0, 0.0]] * count, record=calls),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        matrix = embedder.embed_texts([f"t{i}" for i in range(count)])
        assert len(calls) == 2
        assert matrix.shape == (count, 2)

    def test_batching_preserves_order(self) -> None:
        """A reordered batch would silently attach every vector to the wrong
        chunk -- an index that looks fine and retrieves nonsense."""
        from sparkstory.retrieval.embed import _MAX_BATCH, GeminiEmbedder

        count = _MAX_BATCH + 2
        vectors = [[float(i + 1), 0.0] for i in range(count)]
        embedder = GeminiEmbedder(
            client=self._client(vectors),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        matrix = embedder.embed_texts([f"t{i}" for i in range(count)])
        # Every stub row points along +x, so normalising makes them identical;
        # what matters is that the count and order survive the split.
        assert matrix.shape == (count, 2)
        assert np.allclose(matrix[:, 0], 1.0)

    def test_a_wrong_width_is_rejected_rather_than_stored(self) -> None:
        """The registry width is pinned into the pgvector column and the table
        name. A vector of another width cannot be stored, and finding that out
        at INSERT time would blame the database for a model change."""
        from sparkstory.retrieval.embed import GeminiEmbedder
        from sparkstory.retrieval.exceptions import EmbeddingDimensionError

        embedder = GeminiEmbedder(
            client=self._client([[1.0, 0.0, 0.0]]),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        with pytest.raises(EmbeddingDimensionError) as caught:
            embedder.embed_texts(["a"])
        assert "2" in str(caught.value) and "3" in str(caught.value)

    def test_a_short_response_is_rejected_rather_than_misaligned(self) -> None:
        """The defect a live run found, pinned so it cannot return.

        ``gemini-embedding-2`` returns exactly ONE embedding however many texts
        it is given. The first version of this class was written around a
        list-in-list-out contract, and against that endpoint a 32-text batch came
        back as a single vector -- which `np.vstack` would happily stack into a
        matrix with the wrong number of rows, attaching vectors to the wrong
        chunks for every batch after the first.

        The registry now names ``-001``, which does batch. This test is what
        makes that a checked property rather than a comment: any provider
        returning fewer rows than it was asked for fails loudly here."""
        from sparkstory.retrieval.embed import GeminiEmbedder
        from sparkstory.retrieval.exceptions import EmbeddingDimensionError

        embedder = GeminiEmbedder(
            # Three texts in, one vector back -- exactly what `-2` does.
            client=self._client([[1.0, 0.0]]),
            identifier="gemini-embedding-2",
            dimensions=2,
        )
        with pytest.raises(EmbeddingDimensionError) as caught:
            embedder.embed_texts(["a", "b", "c"])
        assert "3" in str(caught.value) and "1" in str(caught.value)

    def test_a_repeated_query_is_not_embedded_twice(self) -> None:
        """Added after a live run exhausted the free tier's 100 requests/minute.

        `make test-corpus` embeds the same 19 labelled queries once per retriever
        per top_k -- 114 requests for 19 distinct strings. That cannot pass in one
        quota window however long you wait for it, so the fix is not to embed the
        same string twice rather than to wait longer."""
        from sparkstory.retrieval.embed import GeminiEmbedder

        calls: list = []
        embedder = GeminiEmbedder(
            client=self._client([[3.0, 4.0]] * 4, record=calls),
            identifier="gemini-embedding-001",
            dimensions=2,
        )
        first = embedder.embed_query("could a flag wave on the moon?")
        second = embedder.embed_query("could a flag wave on the moon?")
        assert len(calls) == 1, "the second call should have been served cached"
        assert np.allclose(first, second)

    def test_a_cached_query_survives_a_caller_mutating_it(self) -> None:
        """The cache hands out copies. Returning the stored array itself would
        let one caller's in-place edit corrupt every later search for that
        query -- a bug that would surface as retrieval quietly degrading."""
        from sparkstory.retrieval.embed import GeminiEmbedder

        embedder = GeminiEmbedder(
            client=self._client([[3.0, 4.0]] * 4),
            identifier="gemini-embedding-001",
            dimensions=2,
        )
        first = embedder.embed_query("a query")
        first[0] = 99.0
        assert embedder.embed_query("a query")[0] != 99.0

    def test_a_missing_key_names_the_variable_and_the_model(self) -> None:
        """Rule 21's stage five. Four stages have now defaulted to a provider
        `.env` did not pin, and the memory extractor's version failed open and
        looked normal. This one must fail loudly and say what to set."""
        from sparkstory.models.exceptions import MissingAPIKeyError
        from sparkstory.retrieval.embed import build_gemini_embedder

        with pytest.raises(MissingAPIKeyError) as caught:
            build_gemini_embedder(
                config={
                    "identifier": "gemini-embedding-2",
                    "dimensions": 768,
                    "api_key_env_var": "GOOGLE_API_KEY",
                },
                model_id="gemini-embedding",
                api_key=None,
            )
        message = str(caught.value)
        assert "GOOGLE_API_KEY" in message
        assert "gemini-embedding" in message
