"""The memory store: append-only writes, exact reads.

Runs on the SQLite `memory_engine` fixture, so these execute under a plain
`make test` with no database. That is deliberate -- see the fixture's docstring
for why copying the `corpus`-marked pgvector seam would have left every
guarantee here unchecked.
"""

from sqlalchemy import Engine

from sparkstory.memory.store import PgMemoryStore
from sparkstory.memory.types import MemoryKind, MemoryRecord


def _record(child_id: str, text: str, subject: str = "Kit") -> MemoryRecord:
    return MemoryRecord(
        child_id=child_id,
        kind=MemoryKind.SEMANTIC,
        text=text,
        subject=subject,
        source_request_id="req-1",
    )


def test_fetch_is_byte_identical_across_calls(memory_engine: Engine) -> None:
    """The Finn/Kit guard, and the reason reads are SQL rather than similarity.

    A near-miss on a character fact IS the defect this package exists to fix, so
    two fetches must agree exactly -- not merely overlap.
    """
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    store.save([_record("maryam", "Kit is a fox with a white-tipped tail.")])

    first = store.fetch("maryam")
    second = store.fetch("maryam")

    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]


def test_one_child_never_sees_another(memory_engine: Engine) -> None:
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    store.save([_record("maryam", "Kit is a fox.")])
    store.save([_record("sam", "Ted is a bear.")])

    assert [r.text for r in store.fetch("maryam")] == ["Kit is a fox."]
    assert [r.text for r in store.fetch("sam")] == ["Ted is a bear."]


def test_a_contradiction_leaves_the_original_row_intact(memory_engine: Engine) -> None:
    """`keep both` is a schema property, not a convention."""
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    original = _record("maryam", "Kit has a white-tipped tail.")
    store.save([original])
    store.save([_record("maryam", "Kit has a bushy red tail.")])

    texts = sorted(r.text for r in store.fetch("maryam"))
    assert texts == ["Kit has a bushy red tail.", "Kit has a white-tipped tail."]

    kept = next(r for r in store.fetch("maryam") if r.memory_id == original.memory_id)
    assert kept.text == original.text, "the original must not be rewritten"


def test_superseded_records_are_not_returned(memory_engine: Engine) -> None:
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    old = _record("maryam", "Kit has a white-tipped tail.")
    new = _record("maryam", "Kit has a bushy red tail.")
    store.save([old, new])

    store.supersede(old.memory_id, by=new.memory_id)

    remaining = store.fetch("maryam")
    assert [r.text for r in remaining] == ["Kit has a bushy red tail."]
    assert len(remaining) == 1


def test_fetch_filters_by_kind(memory_engine: Engine) -> None:
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    store.save(
        [
            _record("maryam", "Kit is a fox."),
            MemoryRecord(
                child_id="maryam",
                kind=MemoryKind.EPISODIC,
                text="Story 1: Kit reached the moon.",
                source_request_id="req-1",
            ),
        ]
    )

    semantic = store.fetch("maryam", kind=MemoryKind.SEMANTIC)
    assert [r.text for r in semantic] == ["Kit is a fox."]


def test_fetch_on_an_unknown_child_is_empty_not_an_error(memory_engine: Engine) -> None:
    """A child's first book must work, so absence is normal."""
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    assert store.fetch("nobody") == []


def test_saving_nothing_is_a_no_op(memory_engine: Engine) -> None:
    """A book that established nothing must not open a transaction to say so."""
    store = PgMemoryStore(database_url=None, engine=memory_engine)
    store.save([])
    assert store.fetch("maryam") == []
