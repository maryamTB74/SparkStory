"""What this run fetched from the web, and what it is allowed to vouch for.

The web counterpart of ``LocalVectorStore``: the thing a ``web:<n>`` id resolves
against when ``drop_unprovenanced`` decides whether to keep a fact. Pure data --
no network, no model, no key -- which is what lets everything downstream of it be
tested offline.

**Ids are run-scoped, and that is a deliberate difference from corpus ids.**
``moon#3`` is stable across runs because the corpus is committed text, and that
stability is load-bearing: a fact recorded months ago can still be looked up. A
web source has no such identity. The same query tomorrow returns different pages,
so ``web:1`` means "the first source *this run* accepted" and nothing more. A
module-level counter would make a later run's ``web:1`` resolve against an earlier
run's page -- wrong in the most dangerous way, which is plausibly.

**A source is unverified until proven otherwise.** ``verified`` defaults to
``False`` so that a construction site which forgets the flag produces something
provenance will *reject*, rather than something it will accept on trust.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# `web:` rather than a bare number so an id is self-describing wherever it turns
# up -- in a model's output, in a run artifact, in a log line. It also makes the
# store/ledger split checkable by inspection: `moon#1` and `web:1` cannot be
# confused for one another by a human or by a prefix test.
WEB_ID_PREFIX = "web:"


def web_id_for(ordinal: int) -> str:
    """Build a ledger id from a 0-based position.

    One-based in the rendered form, matching ``chunk_id_for``: the id is shown to
    a model and to a human reading an artifact, and ``web:0`` reads as a mistake.
    """
    return f"{WEB_ID_PREFIX}{ordinal + 1}"


# Not prompt text, and it is the exception that proves the rule in this codebase.
class Evidence(StrEnum):
    """How a source came to be trusted, which is not always the same way.

    A boolean cannot carry this. ``verified`` answers "may this ground a book?"
    and two different checks can both answer yes while meaning different things,
    so the *kind* is recorded beside the verdict. Without this, a run during a
    Firecrawl outage would be indistinguishable afterwards from a healthy one.
    """

    #: Nothing checked it. Never grounds a book.
    NONE = "none"
    #: A page was fetched and a judge quoted a sentence from it supporting the
    #: claim, with the quote confirmed present in code. The full guarantee.
    FETCHED = "fetched"
    #: The URL came structurally from a search API's response rather than from a
    #: model, so it cannot be fabricated -- but no page was fetched, so nothing
    #: confirmed it *says* what was claimed. Accepted only when the fetcher is
    #: unavailable. Weaker, deliberately, and recorded as such.
    SEARCH_API = "search_api"


# A WebSource is never returned by a model and never bound as an output schema --
# it is built by us from what a provider returned. What *does* reach a model is
# `text` and the id, handed over by the search tool.
class WebSource(BaseModel):
    """One page this run consulted, and whether it earned its citation."""

    url: str = Field(description="Where the claim came from.")
    title: str = Field(description="What the page is called.")
    text: str = Field(description="The passage the claim was drawn from.")
    query: str = Field(description="The search that surfaced it.")
    # False until something has cleared it. Recorded rather than implied, so a
    # run with verification switched off is distinguishable afterwards from one
    # where every source genuinely passed.
    verified: bool = Field(
        default=False,
        description="Whether this source may ground a story.",
    )
    # *How* it was cleared, which `verified` alone cannot say. A reader asking
    # "was this book grounded in checked pages?" needs this, not the boolean.
    evidence: Evidence = Field(
        default=Evidence.NONE,
        description="What kind of check cleared this source.",
    )


class WebLedger:
    """The web sources one planning run accepted, in the order it accepted them.

    Deliberately a plain class rather than a Protocol or an injected interface.
    There is one implementation and no second caller, and this codebase's rule is
    that an abstraction arrives when a second implementation does -- the same
    argument that kept ``FileCanonStore`` concrete.
    """

    def __init__(self) -> None:
        self._sources: list[WebSource] = []

    def add(self, source: WebSource) -> str:
        """Record a source and return the id that now resolves to it."""
        self._sources.append(source)
        return web_id_for(len(self._sources) - 1)

    def get(self, source_id: str) -> WebSource | None:
        """Resolve an id, or ``None`` if this ledger did not mint it.

        Returns ``None`` for a corpus id too. The ledger must not answer for the
        store: ``drop_unprovenanced`` consults both, and an id resolving against
        the wrong one would attribute a corpus fact to a web page or the reverse.
        """
        if not source_id.startswith(WEB_ID_PREFIX):
            return None
        try:
            ordinal = int(source_id.removeprefix(WEB_ID_PREFIX))
        except ValueError:
            return None
        if not 1 <= ordinal <= len(self._sources):
            return None
        return self._sources[ordinal - 1]

    @property
    def sources(self) -> list[WebSource]:
        """Everything this run recorded, verified or not.

        A copy, so a caller writing the run artifact cannot mutate the ledger the
        provenance filter is about to consult.
        """
        return list(self._sources)

    def __bool__(self) -> bool:
        """False when nothing was consulted, which is the usual case."""
        return bool(self._sources)

    def __len__(self) -> int:
        return len(self._sources)
