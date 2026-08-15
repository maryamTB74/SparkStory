"""Turning an asserted URL into a checked one.

**The load-bearing module of this feature.** ``providers`` returns a URL the
*model* wrote; everything downstream treats a ledger entry as provenance. This is
what stands between the two, and with Tavily dropped it is the only thing that
does.

Two gates, cheapest first.

**The page must exist.** A fabricated URL cannot survive being fetched, which is
what makes it safe to receive one at all. A dead link settles the question before
any model call -- cheaper, and it means an invented URL never reaches a judge
that might be talked into accepting it.

**The page must say what was claimed**, and this half is a model judging a model.
The question to ask of any rubric was asked before the prompt was written: *what
is the laziest thing that satisfies this?* For "is this claim supported?" the
laziest answer is **yes**. So a verdict alone is not accepted -- the judge must
quote the supporting sentence, and **code checks the quote is in the page**.
A model that agrees but cannot point at the sentence has verified nothing. That
is the same move ``drop_unprovenanced`` makes by overwriting ``source`` from the
store: convert a claim we would have to trust into one we can check.

Quote matching is deliberately forgiving on formatting and strict on content. A
page's line wrapping is not a reason to reject a real quote; an invented sentence
is the thing being caught.

**Failure returns ``None``, never raises.** A fact that cannot be verified costs
the fact, not the book -- the same split ``provenance.py`` already draws.
"""

import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from sparkstory.config import settings
from sparkstory.retrieval.web.ledger import Evidence
from sparkstory.retrieval.web.providers import WebResult
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

PageFetcher = Callable[[str], Awaitable[str | None]]
ClaimJudge = Callable[[str, str], Awaitable["ClaimVerdict"]]


# This one *is* prompt text -- it is bound as the judge's output schema, so the
# docstring and both descriptions reach the model.
class ClaimVerdict(BaseModel):
    """Whether a page supports a claim, and the sentence that shows it."""

    supported: bool = Field(
        description=(
            "True only if the page plainly says this. If you are unsure, or the "
            "page only touches on the subject, answer false."
        )
    )
    quote: str = Field(
        default="",
        description=(
            "The sentence from the page that shows it, copied exactly, word for "
            "word. Leave empty if there is none. Never write a sentence that is "
            "not in the page."
        ),
    )


CLAIM_JUDGE_PROMPT = """\
Below is a claim and the text of a web page.

Decide one thing: does this page plainly say the claim is true?

Answer false if you are unsure, if the page only touches on the subject, or if \
the claim needs several parts of the page put together. Only a claim the page \
states plainly counts. Being cautious costs one fact; being agreeable puts \
something unchecked into a book for a small child.

If it is true, copy out the one sentence from the page that shows it, word for \
word. The sentence is checked against the page afterwards, so a sentence you \
compose rather than copy will be caught and the claim thrown away.

CLAIM:
{claim}

PAGE:
{page}"""


def _normalise(text: str) -> str:
    """Collapse whitespace and case so formatting cannot fail a real quote.

    Strict on content, forgiving on shape: the failure being caught is an
    invented sentence, not a reflowed one. Punctuation is kept, because dropping
    it would let "submarines sink" match "submarines sink?" -- a question and a
    statement are different claims.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


async def fetch_page(url: str, fetcher: PageFetcher | None = None) -> str | None:
    """Fetch a page as markdown, or ``None`` if it cannot be read.

    Args:
        url: The address to fetch.
        fetcher: How to fetch. Defaults to Firecrawl; tests inject a fake, so
            nothing here constructs a client or reads a key unless asked to.

    Returns:
        The page text, or ``None`` for a 404, a timeout, or an empty page. A
        dead link is an ordinary outcome here rather than an error -- it is
        precisely how a fabricated URL is meant to be caught.
    """
    fetch = fetcher or _firecrawl_fetcher()
    if fetch is None:
        logger.warning("no fetcher configured, cannot read %s", url)
        return None
    page = await fetch(url)
    if page is None or not page.strip():
        logger.warning("could not read %s -- dropping the claim that cited it", url)
        return None
    return page


async def verify_result(
    result: WebResult,
    fetcher: PageFetcher | None = None,
    judge: ClaimJudge | None = None,
    verify: bool | None = None,
) -> WebResult | None:
    """Check a candidate source, returning it verified or ``None``.

    Args:
        result: What search returned. Its ``url`` is model-asserted at this point.
        fetcher: Injected in tests.
        judge: Injected in tests.
        verify: Overrides ``settings.verify_web_claims``. When falsy the whole
            check is skipped and the result comes back **still unverified** --
            never silently marked otherwise, because ``drop_unprovenanced`` drops
            unverified sources and that is what makes the skip safe.

    Returns:
        The result with ``verified=True``, or ``None`` if it failed either gate.
    """
    should_verify = settings.verify_web_claims if verify is None else verify
    if not should_verify:
        # Kept, but honestly labelled. The record says the check did not run, so
        # a later reader can tell this apart from a source that genuinely passed.
        logger.warning("verification is off: %s is unchecked", result.url)
        return result

    fetch = fetcher or _firecrawl_fetcher()
    if fetch is None:
        # Firecrawl is the thing that is unavailable, and no search provider can
        # stand in for it -- searching and fetching are different jobs. So the
        # only question left is whether the URL can be trusted on its own, and
        # that depends entirely on where it came from.
        return _accept_on_structural_url(result)

    page = await fetch_page(result.url, fetcher=fetch)
    if page is None:
        return None

    ask = judge or _claim_judge()
    verdict = await ask(result.text, page)

    if not verdict.supported:
        logger.info("page does not support the claim, dropping: %s", result.url)
        return None

    if not verdict.quote.strip():
        logger.warning("supported with no quote, dropping: %s", result.url)
        return None

    if _normalise(verdict.quote) not in _normalise(page):
        # The check the judge cannot talk its way past. A quote that is not in
        # the page means the verdict was composed rather than read.
        logger.warning(
            "quoted a sentence that is not on the page, dropping: %s", result.url
        )
        return None

    logger.info("verified against %s", result.url)
    return result.model_copy(update={"verified": True, "evidence": Evidence.FETCHED})


def _accept_on_structural_url(result: WebResult) -> WebResult | None:
    """Decide a source when no page can be fetched.

    A **structural** URL came out of a search API's response, so it cannot be a
    fabrication -- accepted, and recorded as ``SEARCH_API`` rather than
    ``FETCHED`` because nothing confirmed the page *says* what was claimed. The
    text is the snippet that engine returned for the query, so the claim is not
    unmoored; it is just not checked against the full page. A genuinely
    on-topic page that says something slightly different would pass.

    A **model-asserted** URL has nothing behind it at all without a fetch, so it
    is dropped. An outage may degrade the guarantee; it may not remove it.
    """
    if not result.url_is_structural:
        logger.warning(
            "no fetcher and the url is model-asserted, dropping: %s", result.url
        )
        return None

    logger.warning(
        "no fetcher available: accepting %s on its source alone, unchecked "
        "against the page",
        result.url,
    )
    return result.model_copy(update={"verified": True, "evidence": Evidence.SEARCH_API})


def _firecrawl_fetcher() -> PageFetcher | None:
    """Build the real fetcher, or ``None`` when no key is configured.

    Returning ``None`` rather than raising is what makes the degraded path
    possible: a missing key means "cannot fetch", which is a state the caller
    can reason about, whereas an exception would take the run down over a
    service that is only needed for enrichment.

    Imported lazily and constructed on call, so at ``MAX_WEB_SEARCHES=0`` no
    client exists, no key is read, and the dependency is never imported.
    """
    api_key = settings.api_key_for("FIRECRAWL_API_KEY")
    if not api_key:
        return None

    async def fetch(url: str) -> str | None:
        from firecrawl import AsyncFirecrawl

        try:
            response = await AsyncFirecrawl(api_key=api_key).scrape(
                url, formats=["markdown"], timeout=30_000
            )
        except Exception as exc:
            # Deliberately broad, and the only place in this codebase that is.
            # Every failure mode here -- 404, timeout, TLS error, robots block,
            # a paywall -- means the same thing to us: the page cannot be read,
            # so the claim goes. Letting any of them propagate would turn one bad
            # URL into a dead run.
            logger.warning("fetch failed for %s: %s", url, exc)
            return None
        return getattr(response, "markdown", None)

    return fetch


def _claim_judge() -> ClaimJudge:
    """Build the real judge.

    Uses the researcher's own model rather than a new setting: this is part of
    research, and a setting only arrives once something needs it to differ. A
    dedicated ``CLAIM_JUDGE_MODEL`` would be config for a distinction nobody has
    asked for yet.
    """

    async def judge(claim: str, page: str) -> ClaimVerdict:
        from sparkstory.models.get_model import get_chat_model

        model = get_chat_model(settings.researcher_model, schema=ClaimVerdict)
        return await model.ainvoke(
            CLAIM_JUDGE_PROMPT.format(claim=claim, page=page[:20_000])
        )

    return judge
