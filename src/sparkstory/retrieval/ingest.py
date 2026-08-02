"""Turning ``corpus/`` into chunks, and chunks into an index.

Runs offline, from a script, never on a request path.

The file format is deliberately dumb -- front-matter, then paragraphs separated by
``---`` -- because the corpus is hand-written and the format's only job is to stay
out of the way. No YAML parser: the front-matter is four known keys, and a
dependency to read four keys would be a dependency to audit.

**Missing attribution is a hard error, not a warning.** Ingesting an unattributed
chunk would surface much later as a fact with no source, in the one feature whose
purpose is being able to say where a claim came from.
"""

from collections import Counter
from pathlib import Path

from sparkstory.entities.exceptions import SparkStoryError
from sparkstory.retrieval.chunks import Chunk, SourceKind, chunk_id_for
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SEPARATOR = "\n---\n"
_REQUIRED_FIELDS = ("title", "source", "licence")

#: Directory name -> the kind of every chunk inside it. The directory *is* the
#: metadata, so there is no per-file field that can contradict it.
_KIND_DIRECTORIES = {"facts": SourceKind.FACT, "craft": SourceKind.CRAFT}


class CorpusError(SparkStoryError):
    """A corpus file is malformed.

    Not a ``ConfigurationError``: nobody fixes this by editing ``.env``, and it is
    raised only by ingestion, which is a developer running a script rather than a
    client making a request. Deliberately never raised on a request path -- a
    missing index is handled by returning nothing, not by raising.
    """


def _split_front_matter(text: str, path: Path) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise CorpusError(
            f"{path.name} has no front-matter. It must begin with a --- block "
            "carrying title, source and licence."
        )

    _, header, body = text.split("---\n", 2)
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()

    missing = [field for field in _REQUIRED_FIELDS if not fields.get(field)]
    if missing:
        raise CorpusError(
            f"{path.name} is missing required front-matter: {', '.join(missing)}. "
            "Every chunk carries its own attribution, so none of these is optional."
        )
    return fields, body


def parse_corpus_file(path: Path, source_kind: SourceKind) -> list[Chunk]:
    """Read one corpus file into chunks.

    Chunk ids come from the file stem and the chunk's position, so re-reading an
    unchanged file reproduces every id exactly -- which is what makes a recorded
    ``chunk_id`` still resolvable a session later.

    Raises:
        CorpusError: the file has no front-matter, is missing a required field, or
            contains no chunks.
    """
    text = path.read_text(encoding="utf-8")
    fields, body = _split_front_matter(text, path)

    # Blank fragments are dropped rather than becoming empty chunks: an empty
    # chunk embeds to an all-zero vector, which matches nothing while still
    # inflating the corpus count and shifting no ids -- a silent dead entry.
    pieces = [piece.strip() for piece in body.split(_SEPARATOR)]
    texts = [piece for piece in pieces if piece]
    if not texts:
        raise CorpusError(f"{path.name} has no chunks after its front-matter.")

    return [
        Chunk(
            chunk_id=chunk_id_for(path.stem, ordinal),
            text=chunk_text,
            title=fields["title"],
            source=fields["source"],
            licence=fields["licence"],
            url=fields.get("url") or None,
            source_kind=source_kind,
        )
        for ordinal, chunk_text in enumerate(texts)
    ]


def load_corpus(corpus_root: Path) -> list[Chunk]:
    """Read every corpus file under ``corpus_root`` into chunks.

    A missing ``facts/`` or ``craft/`` directory is fine -- a corpus with only one
    kind is legitimate, and the tool for the other kind simply finds nothing.

    Raises:
        CorpusError: a file is malformed, or two chunks share an id. Ids must be
            unique across the whole corpus, because provenance lookup takes an id
            and nothing else: two chunks sharing one makes a recorded fact
            ambiguous.
    """
    chunks: list[Chunk] = []
    for directory, kind in sorted(_KIND_DIRECTORIES.items()):
        folder = corpus_root / directory
        if not folder.is_dir():
            logger.debug("No %s/ directory under %s", directory, corpus_root)
            continue
        for path in sorted(folder.glob("*.md")):
            chunks.extend(parse_corpus_file(path, kind))

    counts = Counter(chunk.chunk_id for chunk in chunks)
    duplicates = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    if duplicates:
        raise CorpusError(
            f"duplicate chunk ids: {', '.join(duplicates)}. Two corpus files in "
            "different directories share a filename -- rename one."
        )
    return chunks
