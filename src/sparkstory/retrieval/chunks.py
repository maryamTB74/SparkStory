"""One retrievable unit of the corpus.

**Nothing in this module is prompt text**, which makes it the exception in a
codebase where docstrings usually are. A ``Chunk`` is never returned by a model
and never bound as an output schema -- it is written by the ingestion script and
read by the store -- so its docstrings and field names are for us, and no
``Field(description=...)`` is needed. What *does* reach a model is
``Chunk.text`` and ``Chunk.source``, handed over by the retrieval tools.

Two decisions worth knowing before editing.

**Ids are positional, not content-derived.** ``moon#3`` is the third chunk of
``corpus/facts/moon.md``, so re-ingesting an unchanged corpus reproduces every id
exactly. That stability is a requirement rather than a nicety: a ``GroundedFact``
records a ``chunk_id``, and proving a fact came from us means looking that id up
later. A content-hash id would change whenever a typo was fixed, silently
invalidating every fact ever recorded.

The cost of positional ids is that editing a chunk keeps its id while changing its
meaning, so ``content_sha256`` is stored to make that detectable. Inserting a
chunk mid-file renumbers everything after it, which is a real hazard -- append
rather than insert.
"""

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceKind(StrEnum):
    """Which index a chunk belongs to.

    Exactly two, because each retrieval tool pins one. A third value would be
    reachable by no tool, i.e. a chunk nothing could ever find.
    """

    FACT = "fact"
    CRAFT = "craft"


def chunk_id_for(file_stem: str, ordinal: int) -> str:
    """Build a stable chunk id from a corpus filename and a 0-based position.

    One-based in the rendered form, because the id is shown to a model and to a
    human reading an artifact, and `moon#0` reads as a mistake.
    """
    return f"{file_stem}#{ordinal + 1}"


class Chunk(BaseModel):
    """A single fact or craft example, with everything needed to attribute it."""

    chunk_id: str = Field(description="Stable id, `<file stem>#<1-based ordinal>`.")
    text: str = Field(description="The chunk itself, as the agent will read it.")
    title: str = Field(description="What the source document is about.")
    source: str = Field(description="Human-readable attribution, always present.")
    licence: str = Field(description="The licence the source text is under.")
    # Optional because a fabricated citation is worse than an absent one in a
    # feature whose whole purpose is factual accuracy. `source` is mandatory;
    # `url` is filled in only where the address is known to be right.
    url: str | None = Field(default=None, description="Source URL, where known.")
    source_kind: SourceKind = Field(description="Which index this belongs to.")

    @property
    def embed_text(self) -> str:
        """The text that gets embedded, prefixed with its source title.

        Lesson 9's context-enriched chunking: *"It has no air"* embeds poorly on
        its own and well as *"The Moon: It has no air"*. Deliberately different
        from ``text``, which is what the agent is shown -- if the title travelled
        with the fact it would end up quoted inside a ``GroundedFact.claim``.
        """
        return f"{self.title}: {self.text}"

    @property
    def content_sha256(self) -> str:
        """Hash of ``text``, so a rewritten chunk under a reused id is detectable.

        ``hashlib`` rather than the built-in ``hash``, which is salted per process
        -- a salted value would differ between the run that wrote an index and the
        run that reads it, which makes a stored hash worse than none.
        """
        return hashlib.sha256(self.text.encode()).hexdigest()
