"""The corpus format rules, enforced rather than documented.

**Deliberately not marked ``corpus``.** These compare committed files against a
committed manifest and need neither a database nor model weights, so they belong in
the default suite where they will actually run when somebody edits a fact file. A
guard that only fires under ``make test-corpus`` would not have caught the edit that
motivated it.

The load-bearing rule is id stability, and it is the one nothing enforced.
"""

import hashlib
import json
from pathlib import Path

from sparkstory.retrieval.ingest import load_corpus

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _ROOT / "corpus"
_MANIFEST = _CORPUS / "chunk_manifest.json"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_chunk_ids_still_point_at_the_same_text() -> None:
    """Chunk ids are positional and permanent: the third chunk of ``moon.md`` is
    ``moon#3`` for as long as anything cites it.

    So inserting a chunk renumbers every chunk after it, and every recorded
    citation to those ids silently starts pointing at different text. Nothing
    enforced that, and the symptom would not look like an insertion -- it would
    look like labelled queries suddenly missing their targets, which reads as a
    retrieval regression and would be debugged as one, in the wrong file.

    **Appending is fine and does not fail this.** Only changing what an existing id
    points at does. When an edit is deliberate, regenerate the manifest:
    ``make corpus-manifest``.
    """
    manifest = json.loads(_MANIFEST.read_text())
    current = {chunk.chunk_id: _digest(chunk.text) for chunk in load_corpus(_CORPUS)}

    moved = sorted(
        chunk_id
        for chunk_id, digest in manifest.items()
        if chunk_id in current and current[chunk_id] != digest
    )
    assert not moved, (
        f"these ids now point at different text: {moved}. An insertion renumbers "
        "every chunk after it, which invalidates any recorded citation to them. "
        "Append instead -- or run `make corpus-manifest` if the change is meant."
    )


def test_no_chunk_the_manifest_knows_about_has_vanished() -> None:
    """A deletion is as damaging as an insertion and looks different: the id stops
    resolving rather than resolving to the wrong thing. Both break a citation, so
    both are caught, but they need separate messages or the fix is guesswork.
    """
    manifest = json.loads(_MANIFEST.read_text())
    current = {chunk.chunk_id for chunk in load_corpus(_CORPUS)}

    gone = sorted(set(manifest) - current)
    assert not gone, (
        f"these ids no longer exist: {gone}. Anything that cited them now cites "
        "nothing. Run `make corpus-manifest` if the removal is deliberate."
    )


def test_every_fact_file_has_complete_front_matter() -> None:
    """``title``, ``source`` and ``licence`` are all required.

    Without source and licence a chunk cannot be attributed, and attribution is the
    whole of what a factual claim in a children's book rests on -- there is no
    other check that the fact is true. ``url`` is deliberately not required: a
    plausible fabricated address is worse than no address in a feature whose
    purpose is accuracy.
    """
    missing = []
    for path in sorted((_CORPUS / "facts").glob("*.md")):
        text = path.read_text()
        front_matter = text.split("---")[1] if text.startswith("---") else ""
        for key in ("title:", "source:", "licence:"):
            if key not in front_matter:
                missing.append(f"{path.name} has no {key}")
    assert not missing, missing


def test_no_chunk_is_long_enough_to_hold_two_facts() -> None:
    """One fact per chunk, in a sentence a five-year-old could hear.

    A chunk holding two facts blurs its own embedding: it matches both subjects
    weakly instead of one strongly, so it ranks below single-subject chunks for
    both -- and when it does win, the text handed over carries a second fact
    nobody asked about, which the Researcher then has to decide about.

    Sixty words is generous against a corpus whose chunks average well under
    thirty; it catches a paragraph pasted in, not a slightly long sentence.
    """
    too_long = [
        f"{chunk.chunk_id} ({len(chunk.text.split())} words)"
        for chunk in load_corpus(_CORPUS)
        if len(chunk.text.split()) > 60
    ]
    assert not too_long, (
        f"these chunks are long enough to hold more than one fact: {too_long}"
    )


def test_the_manifest_covers_every_chunk() -> None:
    """A chunk absent from the manifest is a chunk the stability guard is blind
    to, which is the guard failing quietly rather than the corpus being wrong.
    """
    manifest = json.loads(_MANIFEST.read_text())
    unmanifested = sorted(
        {chunk.chunk_id for chunk in load_corpus(_CORPUS)} - set(manifest)
    )
    assert not unmanifested, (
        f"these chunks are not in the manifest, so nothing guards their ids: "
        f"{unmanifested}. Run `make corpus-manifest`."
    )
