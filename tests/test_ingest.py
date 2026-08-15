"""Loading the corpus into chunks.

These test the *loader*, not the corpus. Whether the facts are true is a review
question and belongs to a human -- there is no assertion that can check it, and one
that pretended to would be worse than none.

The one corpus-wide test at the end is a structural smell check: it asserts the
committed corpus parses and carries a licence on every file, which is the failure
that would otherwise appear as "retrieval returns nothing" much later.
"""

from pathlib import Path

import pytest

from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.ingest import CorpusError, load_corpus, parse_corpus_file

WELL_FORMED = """\
---
title: The Moon
source: NASA -- Earth's Moon
licence: public domain
---

The Moon has no air.

---

The Moon has no sound.
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestParseCorpusFile:
    def test_splits_on_the_separator(self, tmp_path: Path) -> None:
        chunks = parse_corpus_file(
            write(tmp_path, "moon.md", WELL_FORMED), SourceKind.FACT
        )
        assert [chunk.text for chunk in chunks] == [
            "The Moon has no air.",
            "The Moon has no sound.",
        ]

    def test_ids_are_positional_and_one_based(self, tmp_path: Path) -> None:
        chunks = parse_corpus_file(
            write(tmp_path, "moon.md", WELL_FORMED), SourceKind.FACT
        )
        assert [chunk.chunk_id for chunk in chunks] == ["moon#1", "moon#2"]

    def test_front_matter_reaches_every_chunk(self, tmp_path: Path) -> None:
        """Attribution is per chunk because a `GroundedFact` carries one chunk's
        source, not one file's."""
        chunks = parse_corpus_file(
            write(tmp_path, "moon.md", WELL_FORMED), SourceKind.FACT
        )
        assert all(chunk.source == "NASA -- Earth's Moon" for chunk in chunks)
        assert all(chunk.licence == "public domain" for chunk in chunks)
        assert all(chunk.title == "The Moon" for chunk in chunks)

    def test_source_kind_comes_from_the_caller(self, tmp_path: Path) -> None:
        """The directory decides the kind, not a per-file field that could
        contradict it. One kind exists today; the argument is what matters."""
        chunks = parse_corpus_file(
            write(tmp_path, "moon.md", WELL_FORMED), SourceKind.FACT
        )
        assert all(chunk.source_kind is SourceKind.FACT for chunk in chunks)

    def test_url_is_absent_when_not_given(self, tmp_path: Path) -> None:
        chunks = parse_corpus_file(
            write(tmp_path, "moon.md", WELL_FORMED), SourceKind.FACT
        )
        assert all(chunk.url is None for chunk in chunks)

    def test_url_is_kept_when_given(self, tmp_path: Path) -> None:
        text = WELL_FORMED.replace(
            "licence: public domain",
            "licence: public domain\nurl: https://science.nasa.gov/moon/",
        )
        chunks = parse_corpus_file(write(tmp_path, "moon.md", text), SourceKind.FACT)
        assert chunks[0].url == "https://science.nasa.gov/moon/"

    def test_blank_chunks_are_dropped(self, tmp_path: Path) -> None:
        """A trailing separator or a stray blank paragraph must not become an
        empty chunk: an empty chunk embeds to an all-zero vector, which silently
        matches nothing while inflating the corpus count."""
        text = WELL_FORMED + "\n---\n\n\n---\n\nA real one.\n"
        chunks = parse_corpus_file(write(tmp_path, "moon.md", text), SourceKind.FACT)
        assert [chunk.text for chunk in chunks] == [
            "The Moon has no air.",
            "The Moon has no sound.",
            "A real one.",
        ]

    def test_multi_line_chunks_keep_their_text(self, tmp_path: Path) -> None:
        """Chunks are wrapped in the source files, so a chunk is a paragraph, not
        a line."""
        text = WELL_FORMED.replace(
            "The Moon has no air.", "The Moon has no air,\nso nothing flutters."
        )
        chunks = parse_corpus_file(write(tmp_path, "moon.md", text), SourceKind.FACT)
        assert chunks[0].text == "The Moon has no air,\nso nothing flutters."


class TestMalformedFiles:
    @pytest.mark.parametrize("missing", ["title", "source", "licence"])
    def test_a_missing_required_field_names_the_file(
        self, tmp_path: Path, missing: str
    ) -> None:
        """Silently ingesting an unattributed chunk is the failure mode that
        matters here: it would surface later as a fact with no source, in the one
        feature whose purpose is being able to say where a claim came from."""
        lines = [
            line for line in WELL_FORMED.splitlines() if not line.startswith(missing)
        ]
        path = write(tmp_path, "moon.md", "\n".join(lines) + "\n")
        with pytest.raises(CorpusError, match="moon.md"):
            parse_corpus_file(path, SourceKind.FACT)

    def test_no_front_matter_at_all_is_an_error(self, tmp_path: Path) -> None:
        path = write(tmp_path, "moon.md", "Just some text with no header.\n")
        with pytest.raises(CorpusError, match="front-matter"):
            parse_corpus_file(path, SourceKind.FACT)

    def test_a_file_with_no_chunks_is_an_error(self, tmp_path: Path) -> None:
        header = "---\ntitle: T\nsource: S\nlicence: L\n---\n\n"
        with pytest.raises(CorpusError, match="no chunks"):
            parse_corpus_file(write(tmp_path, "empty.md", header), SourceKind.FACT)


class TestLoadCorpus:
    def test_a_missing_directory_is_not_fatal(self, tmp_path: Path) -> None:
        """A corpus with facts but no craft files is legitimate."""
        (tmp_path / "facts").mkdir()
        write(tmp_path / "facts", "moon.md", WELL_FORMED)
        assert len(load_corpus(tmp_path)) == 2

    # `test_duplicate_chunk_ids_are_rejected` stood here. It built a collision by
    # writing `same.md` into both facts/ and craft/, since a chunk id is
    # `<file stem>#<n>` and the stem ignores the directory.
    #
    # With one kind directory that collision is **unreachable through the public
    # API**: filesystem stems are unique within a directory, so `load_corpus`
    # cannot be made to produce one. The guard in load_corpus is kept -- it costs
    # nothing and a second kind would make it reachable again -- but it is now
    # untested rather than tested, and pretending otherwise by reaching into a
    # private helper would be a test of the test rather than of the guarantee.

    def test_ignores_non_markdown_files(self, tmp_path: Path) -> None:
        (tmp_path / "facts").mkdir()
        write(tmp_path / "facts", "moon.md", WELL_FORMED)
        write(tmp_path / "facts", "notes.txt", "not part of the corpus")
        assert len(load_corpus(tmp_path)) == 2


class TestTheCommittedCorpus:
    """One structural check over the real corpus.

    Not a quality check -- whether a fact is true is Maryam's review. This catches
    a malformed file, which would otherwise show up much later as "the researcher
    found nothing" with no obvious cause.
    """

    def test_it_parses_and_is_attributed(self) -> None:
        chunks = load_corpus(Path(__file__).resolve().parents[1] / "corpus")
        assert len(chunks) >= 40, f"only {len(chunks)} chunks"
        assert all(chunk.source for chunk in chunks)
        assert all(chunk.licence for chunk in chunks)

    def test_no_chunk_is_too_long_to_be_one_idea(self) -> None:
        """One logical unit per chunk. A chunk that has grown into three facts
        retrieves for all three and grounds none of them well."""
        chunks = load_corpus(Path(__file__).resolve().parents[1] / "corpus")
        overlong = [c.chunk_id for c in chunks if len(c.text) > 400]
        assert not overlong, f"chunks too long: {overlong}"
