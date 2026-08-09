"""Shared fixtures.

Fixtures build objects explicitly rather than reading from ``.env`` so tests do
not depend on whatever happens to be configured on the machine running them.
"""

from collections.abc import Callable

import pytest

from sparkstory.entities.stories import (
    CharacterSketch,
    ChildProfile,
    NarrativeFunction,
    PagePlan,
    Pronouns,
    ReadingLevel,
    ScenePlan,
    Story,
    StoryBeat,
    StoryBrief,
    StoryOutline,
    StoryPage,
    StoryProse,
    Tone,
)
from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.types import SearchHit


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Opik off for every test, whatever the developer's .env says.

    Autouse and unconditional, because the failure it prevents is invisible from
    inside the suite. The workflow tests invoke a pipeline 37 times, each
    minting a fresh ``request_id``; with ``OPIK_ENABLED=true`` in a .env that is
    37 threads uploaded per suite run, and it produced hundreds in a real
    project before anyone connected the two.

    This is non-obvious rule 25 again -- a test that fakes only the model still
    reaches a real service, because the seam it forgot is a different one. The
    tests that need tracing on set it themselves, after this fixture runs.
    """
    monkeypatch.setattr("sparkstory.config.settings.opik_enabled", False)


@pytest.fixture
def child() -> ChildProfile:
    return ChildProfile(
        name="Maryam",
        age=5,
        pronouns=Pronouns.SHE_HER,
        reading_level=ReadingLevel.EARLY_READER,
        interests=["foxes", "astronomy"],
    )


@pytest.fixture
def brief(child: ChildProfile) -> StoryBrief:
    return StoryBrief(
        child=child,
        premise="a fox who wants to visit the moon",
        tone=Tone.MAGICAL,
        page_count=10,
        must_include=["a paper rocket"],
        avoid=["spiders", "the dark"],
    )


@pytest.fixture
def outline() -> StoryOutline:
    """A minimal valid outline: exactly the four-beat floor."""
    return StoryOutline(
        title="Maryam and the Paper Rocket",
        logline="A girl and a fox build a rocket to say goodnight to the moon.",
        theme="trying something new even when it feels too big",
        characters=[
            CharacterSketch(
                name="Maryam",
                role="main character",
                description="A curious girl who loves looking at the stars.",
            ),
            CharacterSketch(
                name="Pip",
                role="loyal companion",
                description="A small fox who is braver than he feels.",
            ),
        ],
        beats=[
            StoryBeat(
                position=1,
                function=NarrativeFunction.SETUP,
                title="Goodnight, moon",
                summary=(
                    "Maryam waves to the moon each night and wishes it were closer."
                ),
                characters_present=["Maryam"],
            ),
            StoryBeat(
                position=2,
                function=NarrativeFunction.INCITING_INCIDENT,
                title="The paper rocket",
                summary=(
                    "Pip brings her a paper rocket and suggests they visit instead."
                ),
                characters_present=["Maryam", "Pip"],
            ),
            StoryBeat(
                position=3,
                function=NarrativeFunction.CLIMAX,
                title="Too high",
                summary="The rocket will not fly and Maryam nearly gives up trying.",
                characters_present=["Maryam", "Pip"],
            ),
            StoryBeat(
                position=4,
                function=NarrativeFunction.RESOLUTION,
                title="A closer moon",
                summary="They find the moon reflected in a puddle and say goodnight.",
                characters_present=["Maryam", "Pip"],
            ),
        ],
    )


#: Beat each page draws from, in order: ten pages covering all four beats, with the
#: climax given the most room. Matches `brief.page_count` and stays non-decreasing,
#: so this plan passes validation and a test that wants a failure must break it
#: deliberately.
_PAGE_BEATS = (1, 1, 2, 2, 3, 3, 3, 3, 4, 4)


@pytest.fixture
def page_plan() -> PagePlan:
    """A valid ten-page plan for the `outline` fixture and `brief.page_count`."""
    return PagePlan(
        pages=[
            ScenePlan(
                page_number=number,
                beat_position=beat,
                setting="the garden at night",
                visual_action=f"a single quiet moment, drawn on page {number}",
                emotional_shift="something small shifts",
                # None on the last page only, so the fixture exercises both
                # branches of the optional field rather than just one.
                page_turn_hook=None if number == len(_PAGE_BEATS) else "and then?",
                characters_present=["Maryam", "Pip"],
            )
            for number, beat in enumerate(_PAGE_BEATS, start=1)
        ]
    )


#: One distinct opening word per page. Deliberately varied: the deterministic
#: read-aloud check in `workflows/reviews.py` flags pages that share an opening,
#: so prose reading "Page 1.", "Page 2." would produce a finding in every test
#: that runs the prose loop and drown out whatever that test was measuring.
_PAGE_OPENINGS = (
    "Maryam waited by the window.",
    "Softly, the wind turned over.",
    "Up above, something glittered.",
    "Pip pressed his nose to the glass.",
    "Down came a scattering of light.",
    "Away it went, over the fence.",
    "In the garden nothing moved at all.",
    "Nobody spoke for a long moment.",
    "Then the whole sky leaned closer.",
    "So they said goodnight to the moon.",
)


@pytest.fixture
def prose(page_plan: PagePlan) -> StoryProse:
    """Prose matching the `page_plan` fixture page for page."""
    return StoryProse(
        pages=[
            StoryPage(
                page_number=page.page_number,
                text=_PAGE_OPENINGS[page.page_number - 1],
            )
            for page in page_plan.pages
        ]
    )


@pytest.fixture
def story(outline: StoryOutline, page_plan: PagePlan, prose: StoryProse) -> Story:
    """A finished ten-page story, assembled from the fixtures above.

    Composed rather than hand-written so it cannot drift from the plan and
    prose fixtures the rest of the suite uses.
    """
    return Story(outline=outline, page_plan=page_plan, pages=prose.pages)


@pytest.fixture
def book_factory() -> Callable[..., Story]:
    """Builds a ``Story`` from page texts, for tests that measure one.

    A factory rather than a fixture because the eval metrics are *about* the exact
    words on the page, so each test needs its own book. The ``story`` fixture
    above is the opposite: one fixed book shared by tests that only need a valid
    one.
    """

    def build(page_texts: list[str], beat_summaries: list[str] | None = None) -> Story:
        summaries = beat_summaries or [
            f"Something happens in beat {i + 1}, at some length." for i in range(4)
        ]
        built_outline = StoryOutline(
            title="A Title",
            logline="One sentence that captures the whole story here.",
            theme="a theme worth exploring",
            characters=[
                CharacterSketch(
                    name="Kit",
                    role="main character",
                    description="A child who wonders.",
                )
            ],
            beats=[
                StoryBeat(
                    position=i + 1,
                    function=NarrativeFunction.SETUP,
                    title=f"Beat {i + 1}",
                    summary=summary,
                )
                for i, summary in enumerate(summaries)
            ],
        )
        # Padded to `PagePlan`'s floor of 4 regardless of how many pages of prose
        # a test needs. Nothing being measured reads the plan -- `beats_per_page`
        # divides by the prose pages -- so the padding cannot affect a result.
        built_plan = PagePlan(
            pages=[
                ScenePlan(
                    page_number=i + 1,
                    beat_position=1,
                    setting="a garden",
                    visual_action="Kit looks up, eyes wide",
                    emotional_shift="curiosity",
                )
                for i in range(max(4, len(page_texts)))
            ]
        )
        return Story(
            outline=built_outline,
            page_plan=built_plan,
            pages=[
                StoryPage(page_number=i + 1, text=text)
                for i, text in enumerate(page_texts)
            ],
        )

    return build


# --- A store for tests that are not about retrieval -------------------------
#
# `LocalVectorStore` used to serve this purpose by accident: file-backed, so a
# test could build one under tmp_path and get a working store with no service.
# Postgres removed that, and most tests that used it -- the provenance filter,
# the tool-rendering layer -- do not care how ranking works. They care that a
# store holds chunks, resolves an id, and returns hits.
#
# Same seam as FakeModel and FakeEmbedder, one level out: a real implementation
# of the Protocol with the interesting part removed. Ranking is lexical overlap,
# deliberately not a claim about relevance -- anything measuring retrieval
# quality belongs in test_retrieval_eval.py, against the real store.
class FakeChunkStore:
    """Chunks in a list, searched by counting shared words."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = list(chunks or [])

    def save(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)

    @property
    def is_built(self) -> bool:
        return bool(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self._chunks if c.chunk_id == chunk_id), None)

    def search(
        self,
        query: str,
        source_kind: SourceKind | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Rank by how many query words a chunk contains.

        Filtering happens before scoring, matching the real store -- a test that
        asserts a kind filter works must exercise the same ordering of operations.
        """
        wanted = set(query.lower().split())
        candidates = [
            chunk
            for chunk in self._chunks
            if source_kind is None or chunk.source_kind is source_kind
        ]
        scored = [
            (len(wanted & set(chunk.embed_text.lower().split())), chunk)
            for chunk in candidates
        ]
        # Ties break on insertion order, so a test's expectations stay stable.
        ranked = sorted(scored, key=lambda pair: -pair[0])
        # Every candidate is returned up to top_k, including zero-overlap ones.
        # The real store does the same -- a vector search always ranks the whole
        # filtered set and returns top_k of it, however weak the match. Dropping
        # zero-scoring chunks here made the fake *stricter* than the thing it
        # stands in for, which surfaced as a tool returning one candidate where
        # the payload tests reasonably expect several.
        return [
            SearchHit(chunk=chunk, similarity=float(score))
            for score, chunk in ranked[:top_k]
        ]
