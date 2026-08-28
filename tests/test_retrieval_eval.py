"""Retrieval quality, as a number.

**Marked ``corpus``, so it is excluded from ``make test``** -- it needs the built
index and real embedding weights. Run it with ``make test-corpus`` after
``scripts/ingest_knowledge.py``.

This is the only measurement anywhere in this project. Every other quality question
here is answered by reading two runs and forming an opinion. Chunking should be
tuned on a labelled set against hit rate and recall instead, which is what this
file supplies. Because the embedder is local and deterministic, the same query
produces the same vector forever -- so this is a genuine regression test rather than
a flaky judge, and a prompt or chunking change that hurts retrieval shows up as a
falling number instead of a feeling.

The queries are deliberately *paraphrases*. Asking "does 'the Moon has no air'
retrieve the chunk containing 'the Moon has no air'" measures nothing; asking
whether "could a flag wave on the moon?" finds it measures the thing we care about.

**The two thresholds are different in kind, and the difference matters.** @1 is a
floor (0.75, against 0.81 measured): it has headroom, so a floor beneath it can
still catch a regression. @3 is an equality (1.00): it saturates, and a floor
twenty points beneath a number that cannot move is a check with no room to fail.
Neither is a target -- do not read 0.75 as "retrieval is 75% good".
"""

from pathlib import Path

import pytest

from sparkstory.config import settings
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.embed import get_embedder
from sparkstory.retrieval.pg_store import PgVectorStore, build_store

pytestmark = pytest.mark.corpus

#: (query, expected chunk id, which index). Paraphrases, not restatements.
LABELLED: list[tuple[str, str, SourceKind]] = [
    ("could a flag wave on the moon?", "moon#1", SourceKind.FACT),
    (
        "could you shout to someone standing next to you up there?",
        "moon#2",
        SourceKind.FACT,
    ),
    ("why would you bounce if you walked on the moon?", "moon#3", SourceKind.FACT),
    ("do footprints last a long time up there?", "moon#7", SourceKind.FACT),
    ("why can't a bird fly?", "animals#1", SourceKind.FACT),
    ("can a snail take its house off?", "animals#3", SourceKind.FACT),
    ("how does an animal find its way in the dark?", "animals#4", SourceKind.FACT),
    ("how can you tell how old a tree is?", "plants-and-garden#3", SourceKind.FACT),
    ("which way does a sunflower face?", "plants-and-garden#5", SourceKind.FACT),
    ("what makes a rainbow appear?", "weather#3", SourceKind.FACT),
    ("why does the flash come before the bang?", "weather#5", SourceKind.FACT),
    ("could you stand on a cloud?", "weather#6", SourceKind.FACT),
    ("when are foxes awake?", "foxes#1", SourceKind.FACT),
    # stars-and-sky had seven chunks and no query at all until 2026-08-17, so
    # nothing had ever measured whether any of them could be retrieved. Found by
    # the file-coverage test below on its first run, which is the case that test
    # was written for.
    ("why do stars flicker?", "stars-and-sky#4", SourceKind.FACT),
    ("can you catch a falling star?", "stars-and-sky#5", SourceKind.FACT),
    ("where do the stars go in the daytime?", "stars-and-sky#6", SourceKind.FACT),
    # Exact-term queries. Added after the first measurement showed fusion merely
    # *tying* vector-only -- because every query above is a paraphrase, which is
    # where keyword search is weakest by construction, so the set could not see
    # the half of the design BM25 exists for. These are the shape the Researcher
    # actually issues: the live runs produced "moon no atmosphere",
    # "repetition for early readers", "fox physical features".
    ("moon no atmosphere", "moon#1", SourceKind.FACT),
    ("snow crystals six sides", "weather#4", SourceKind.FACT),
    ("tadpole grows legs", "animals#6", SourceKind.FACT),
]

#: @3 saturates on this corpus: every labelled query finds its target there, on
#: both the fused and the vector-only retriever (16/16 and 16/16, measured
#: 2026-08-17). A floor of 0.8 therefore sat twenty points below a number that is
#: structurally 1.00, so it could not fail and guarded nothing.
#:
#: Asserted at saturation instead, so any query that stops finding its target
#: fails. Verified by pointing a labelled query at a chunk id that does not exist
#: and watching this fire.
#:
#: Do NOT convert this back to a floor. A floor beneath a saturated metric is a
#: check with no room to fail, which is the failure this file exists to detect
#: everywhere else.
EXPECTED_HIT_RATE_AT_3 = 1.0

#: The @1 floor, and unlike @3 this one has genuine headroom: it measured 0.81
#: (13/16) on 2026-08-17, and @1 is where fusion can be told apart from
#: vector-only at all. A floor, not a target.
MINIMUM_HIT_RATE_AT_1 = 0.75


@pytest.fixture(scope="module")
def index() -> PgVectorStore:
    """The store under test, skipped rather than failed when no database is up.

    Skipping is the right behaviour here for the same reason the `corpus` marker
    exists: these tests need a running Postgres holding an ingested corpus, which
    a plain `make test` has not got. A hard failure would make the default suite
    depend on a service.
    """
    if not settings.database_url:
        pytest.skip(
            "DATABASE_URL is not set; start one with `docker compose up -d postgres`"
        )
    store = build_store(
        settings.database_url,
        get_embedder(settings.embedding_model),
        settings.embedding_model,
    )
    if not store.is_built:
        pytest.skip(
            f"No corpus in {store.table.name}. Build it: make migrate && make ingest"
        )
    return store


def _hits_at(index: PgVectorStore, top_k: int = 3) -> tuple[int, list[str]]:
    """Return how many queries found their target in the top ``top_k``, and misses."""
    found = 0
    missed: list[str] = []
    for query, expected, kind in LABELLED:
        results = index.search(query, source_kind=kind, top_k=top_k)
        ids = [hit.chunk.chunk_id for hit in results]
        if expected in ids:
            found += 1
        else:
            missed.append(f"{query!r} wanted {expected}, got {ids}")
    return found, missed


class TestHitRate:
    def test_hit_rate_at_3_is_still_perfect(self, index: PgVectorStore) -> None:
        """Asserted at 1.00 rather than against a floor -- see
        ``EXPECTED_HIT_RATE_AT_3`` for why a floor here guarded nothing.
        """
        found, missed = _hits_at(index, top_k=3)
        rate = found / len(LABELLED)
        # Printed rather than only asserted: the number is the point, and a run
        # that passes still tells you where the corpus is weak.
        print(f"\nhit-rate@3 = {rate:.2f} ({found}/{len(LABELLED)})")
        for line in missed:
            print(f"  MISS {line}")
        assert rate == EXPECTED_HIT_RATE_AT_3, (
            f"hit-rate@3 fell to {rate:.2f}; it has been 1.00. Misses: {missed}"
        )

    def test_hit_rate_at_5_is_reported(self, index: PgVectorStore) -> None:
        """Reported, never asserted, and the omission is deliberate: @5 is looser
        than @3, which already saturates, so an assertion here could not fail
        either.

        It is printed because a query that misses even at @5 says the corpus lacks
        the fact, which is a different problem from ranking it badly and has a
        different fix.
        """
        found, missed = _hits_at(index, top_k=5)
        print(f"\nhit-rate@5 = {found / len(LABELLED):.2f} ({found}/{len(LABELLED)})")
        for line in missed:
            print(f"  MISS AT 5 {line}")

    def test_hit_rate_at_1_holds(self, index: PgVectorStore) -> None:
        """Reported as well as @3 because **@3 saturates on this corpus.** Both
        retrievers score 16/16 there, so the number cannot move far and a floor
        beneath it cannot detect a regression. @1 has room: it is where the first
        measurement was able to tell fusion from vector-only at all.
        """
        found, missed = _hits_at(index, top_k=1)
        rate = found / len(LABELLED)
        print(f"\nhit-rate@1 = {rate:.2f} ({found}/{len(LABELLED)})")
        for line in missed:
            print(f"  MISS {line}")
        assert rate >= MINIMUM_HIT_RATE_AT_1, f"hit-rate@1 fell to {rate:.2f}"

    def test_the_labelled_set_is_big_enough_to_mean_something(self) -> None:
        """A five-query set moves 20 points per query, so it cannot distinguish a
        regression from noise."""
        assert len(LABELLED) >= 15

    def test_every_expected_chunk_actually_exists(self, index: PgVectorStore) -> None:
        """A typo in an expected id would make a query permanently unsatisfiable,
        and the suite would read that as a retrieval problem forever."""
        missing = [
            expected
            for _query, expected, _kind in LABELLED
            if index.get(expected) is None
        ]
        assert not missing, f"labelled ids not in the corpus: {missing}"

    def test_expected_chunks_are_in_the_index_they_are_labelled_for(
        self, index: PgVectorStore
    ) -> None:
        wrong = [
            expected
            for _query, expected, kind in LABELLED
            if (chunk := index.get(expected)) and chunk.source_kind is not kind
        ]
        assert not wrong, f"labelled with the wrong kind: {wrong}"


class TestHybridBeatsEitherHalf:
    """The justification for running two retrievers instead of one.

    If fusion does not beat vector-only on this set, the BM25 half is complexity
    with no payoff and should be argued for again or removed. Recorded as a
    comparison rather than an assertion on the margin, because a small corpus can
    legitimately leave them tied.
    """

    def test_fusion_is_at_least_as_good_as_vectors_alone(
        self, index: PgVectorStore
    ) -> None:
        """Compared **at top-1**, and that is the whole point of this test's
        history. Measured at top-3 first, where both retrievers score 20/20 and the
        comparison is vacuous -- it cannot fail, so it proves nothing. At top-1
        there is room, and fusion wins by one (17 vs 16). That single query is the
        entire measured justification for the keyword half existing; if it goes,
        hybrid search should be argued for again or removed.
        """
        for top_k in (1, 2, 3):
            fused, _ = _hits_at(index, top_k=top_k)
            vector_only = sum(
                expected
                in [
                    hit.chunk.chunk_id
                    for hit in index.search_vectors_only(
                        query, source_kind=kind, top_k=top_k
                    )
                ]
                for query, expected, kind in LABELLED
            )
            print(
                f"\ntop_k={top_k}: fused {fused}/{len(LABELLED)} "
                f"vs vector-only {vector_only}/{len(LABELLED)}"
            )
            assert fused >= vector_only, (
                f"at top_k={top_k}, fusion ({fused}) is worse than vectors alone "
                f"({vector_only}) -- the keyword half is hurting, not helping"
            )


class TestTheCorpusIsReachable:
    def test_no_fact_chunk_is_unreachable_by_its_own_text(
        self, index: PgVectorStore
    ) -> None:
        """A chunk that cannot be retrieved even by its own words is dead weight:
        it costs tokens at build time and can never ground anything. Uses each
        chunk's own text as the query, which is the weakest possible test and
        therefore the one worth having.
        """
        unreachable = []
        for chunk in index.chunks:
            hits = index.search(chunk.text, source_kind=chunk.source_kind, top_k=3)
            if chunk.chunk_id not in [hit.chunk.chunk_id for hit in hits]:
                unreachable.append(chunk.chunk_id)
        assert not unreachable, f"unreachable chunks: {unreachable}"


def test_the_index_matches_the_committed_corpus(index: PgVectorStore) -> None:
    """Catches a stale index -- the corpus changed and nobody re-ingested.

    Without this, `make test-corpus` would measure yesterday's index and pass while
    the corpus it is supposed to be testing has moved on.
    """
    from sparkstory.retrieval.ingest import load_corpus

    corpus = load_corpus(Path(__file__).resolve().parents[1] / "corpus")
    assert len(corpus) == len(index.chunks), (
        "index is stale -- re-run: uv run python scripts/ingest_knowledge.py"
    )


def test_every_fact_file_has_labelled_queries() -> None:
    """A fact file nothing asks about is coverage that no measurement can see.

    LABELLED covers the subjects it does because those are the files that existed
    when it was written, and nothing tied the two together. So adding a seventh
    fact file would raise what the corpus *holds* while every number here stayed
    flat -- they are computed over queries that never mention the new subject. A
    growing corpus and a static hit-rate reads as stability and is the opposite.

    The friction is deliberate and worth stating: adding facts now also costs
    writing queries for them. Growth and measurement move together, or the number
    stops describing the thing it is named after.
    """
    corpus_root = Path(__file__).resolve().parents[1] / "corpus" / "facts"
    files = {path.stem for path in corpus_root.glob("*.md")}
    labelled_files = {expected.split("#")[0] for _query, expected, _kind in LABELLED}

    unmeasured = sorted(files - labelled_files)
    assert not unmeasured, (
        f"these fact files have no labelled query: {unmeasured}. Add at least one "
        f"(query, '{unmeasured[0]}#N', SourceKind.FACT) entry to LABELLED so the "
        "new facts are measured rather than merely stored."
    )


# `rerank` in addition to the file-wide `corpus`, so this class is excluded from
# `make test-corpus` and from CI's corpus job. The distinction is what each
# marker actually promises: `corpus` means "needs the index and real embedding
# weights", which is free and offline once ingest has run; these two tests put an
# LLM in front of every labelled query and bill for it. Sharing one marker is how
# a paid comparison got run unasked, and how it ended up inside a CI job that
# deliberately holds no credentials.
#
# Run it deliberately, having decided to pay:  uv run pytest -m rerank
@pytest.mark.rerank
class TestRerankingAgainstFusion:
    """Does reranking beat plain fusion? Measured, not argued.

    Reported rather than asserted on the margin, for the reason the fusion
    comparison above gives: a corpus this small can leave two rankers legitimately
    tied, and an assertion on a one-query margin would fail on noise.

    **If reranking does not beat fusion at a depth with room to fail, keep fusion
    and write that down.** A mechanism kept because it ought to work is how a
    feature survives without evidence, and a recorded negative result is worth more
    than an unmeasured stage.

    Costs model calls, so it is `corpus`-marked along with the rest of this file
    and runs only under `make test-corpus`.
    """

    async def test_report_hit_rate_with_and_without_reranking(
        self, index: PgVectorStore
    ) -> None:
        """Both rankers see the same ten candidates and return the same count.

        **Read the @1 row carefully: the two rows are not symmetric, and cannot
        be.** Fusion at @1 means "take fusion's own top 1". The reranker at @1
        means "choose 1 out of fusion's top 10". The reranker therefore has
        strictly more information, and no fair version of this comparison exists,
        because fusion *is* the ordering being compared against -- giving it the
        pool would just be asking it the same question twice.

        So the claim the number supports is narrower than "reranking scores 1.00":
        it is *choosing one from a pool of ten beats taking the first of ten, on
        this labelled set, by four queries*. That is the job a reranker does, and
        it is worth knowing -- but measured 2026-08-17, every labelled target
        already sits within fusion's top 3 (15 at rank 1, one at rank 2, three at
        rank 3), so the ceiling here is low and a reranker choosing from ten is
        close to guaranteed to find it. Do not read 1.00 as headroom discovered.
        """
        from sparkstory.models.get_model import get_chat_model
        from sparkstory.retrieval.rerank import identity_reranker
        from sparkstory.retrieval.rerankers.llm import RankedIds, build_llm_reranker

        candidate_pool = 10
        rerankers = {
            "fusion only": identity_reranker,
            "llm rerank": build_llm_reranker(
                get_chat_model(settings.reranker_model).with_structured_output(
                    RankedIds
                )
            ),
        }

        for name, rerank in rerankers.items():
            for top_k in (1, 3):
                found = 0
                for query, expected, kind in LABELLED:
                    hits = index.search(query, source_kind=kind, top_k=candidate_pool)
                    ranked = await rerank(query, hits, top_k)
                    if expected in [hit.chunk.chunk_id for hit in ranked]:
                        found += 1
                print(
                    f"\n{name:<12} @{top_k}: {found}/{len(LABELLED)} "
                    f"= {found / len(LABELLED):.2f}"
                )

    async def test_the_reranker_answers_the_same_way_twice(
        self, index: PgVectorStore
    ) -> None:
        """Run twice on identical input; the ranking must not move.

        A gate rather than a nicety. Retrieval here is deterministic -- a local
        embedder, the same vector for the same query forever -- and that is what
        makes a falling hit-rate mean something. This project already has a
        temperature-zero judge that moved by two pages of an eight-page book across
        identical input, measured three separate times, so temperature alone is not
        evidence of repeatability.

        If this fails, the reranker's hit-rate above is not readable: a number that
        changes between runs cannot say whether reranking helped.
        """
        from sparkstory.models.get_model import get_chat_model
        from sparkstory.retrieval.rerankers.llm import RankedIds, build_llm_reranker

        rerank = build_llm_reranker(
            get_chat_model(settings.reranker_model).with_structured_output(RankedIds)
        )

        unstable = []
        for query, _expected, kind in LABELLED:
            hits = index.search(query, source_kind=kind, top_k=10)
            first = [h.chunk.chunk_id for h in await rerank(query, hits, 3)]
            second = [h.chunk.chunk_id for h in await rerank(query, hits, 3)]
            if first != second:
                unstable.append(f"{query!r}: {first} then {second}")

        stable = len(LABELLED) - len(unstable)
        print(f"\nreranker stability: {stable}/{len(LABELLED)} queries identical")
        for line in unstable:
            print(f"  MOVED {line}")
        assert not unstable, (
            "the reranker reordered identical input, so any hit-rate measured with "
            f"it is unreadable: {unstable}"
        )
