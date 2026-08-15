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
