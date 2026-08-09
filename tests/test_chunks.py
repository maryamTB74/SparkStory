"""What a stored chunk is.

Two properties here are load-bearing beyond this module. ``chunk_id`` must be
stable across re-ingestion, because a `GroundedFact` records one and a later
session proves provenance by looking it up -- an id that changed when the index
was rebuilt would invalidate every fact ever recorded. And ``embed_text`` must
differ from ``text``: the embedded form carries its source title for context,
while the form handed to the agent stays bare.
"""

import hashlib

from sparkstory.retrieval.chunks import Chunk, SourceKind, chunk_id_for


class TestChunkId:
    def test_is_the_file_stem_and_a_one_based_ordinal(self) -> None:
        assert chunk_id_for("moon", 0) == "moon#1"
        assert chunk_id_for("moon", 2) == "moon#3"

    def test_is_stable_for_the_same_position(self) -> None:
        """Re-ingesting an unchanged corpus must produce identical ids, or every
        recorded provenance reference breaks."""
        assert chunk_id_for("mother-goose", 4) == chunk_id_for("mother-goose", 4)

    def test_distinguishes_files_with_the_same_ordinal(self) -> None:
        assert chunk_id_for("moon", 0) != chunk_id_for("penguins", 0)


def a_chunk(text: str = "The Moon has no air.", **overrides: object) -> Chunk:
    payload: dict = {
        "chunk_id": "moon#1",
        "text": text,
        "title": "The Moon",
        "source": "NASA -- Moon Facts",
        "licence": "public domain",
        "url": None,
        "source_kind": SourceKind.FACT,
    }
    payload.update(overrides)
    return Chunk(**payload)


class TestChunk:
    def test_round_trips_through_json(self) -> None:
        """The store writes chunks as JSON, so a field that cannot round-trip is
        a field that silently disappears from an index."""
        original = a_chunk()
        restored = Chunk.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_embed_text_carries_the_source_title(self) -> None:
        """Context-enriched chunking, lesson 9: a bare fact embeds worse than one
        that says what it is about."""
        chunk = a_chunk(text="It has no air.", title="The Moon")
        assert "The Moon" in chunk.embed_text
        assert "It has no air." in chunk.embed_text

    def test_text_stays_bare(self) -> None:
        """What the agent reads is the fact, not the fact plus a heading. The
        title is retrieval scaffolding and would otherwise end up quoted in a
        `GroundedFact.claim`."""
        assert a_chunk().text == "The Moon has no air."

    def test_content_hash_tracks_the_text(self) -> None:
        """Ids are stable by design, which means a rewritten chunk keeps its id.
        The hash is how a changed chunk under an unchanged id is detectable."""
        assert a_chunk().content_sha256 == a_chunk().content_sha256
        other = a_chunk(text="Something else.")
        assert a_chunk().content_sha256 != other.content_sha256

    def test_hash_is_stable_across_processes(self) -> None:
        """Computed independently here rather than pinned to a literal: a salted
        hash (Python's built-in ``hash``) would differ between the run that wrote
        an index and the run that reads it, making every stored hash useless."""
        expected = hashlib.sha256(b"The Moon has no air.").hexdigest()
        assert a_chunk().content_sha256 == expected

    def test_url_is_optional(self) -> None:
        """Fabricating a plausible citation is worse than omitting one, so `url`
        may be absent while `source` may not."""
        assert a_chunk(url=None).url is None
        assert a_chunk(url="https://example.org/moon").url is not None


class TestSourceKind:
    def test_has_exactly_the_two_kinds_the_tools_pin(self) -> None:
        """Each retrieval tool pins one kind. A third value would be reachable by
        neither tool, which is a chunk nothing can ever find."""
        assert {kind.value for kind in SourceKind} == {"fact"}
