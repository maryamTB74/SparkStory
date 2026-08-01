"""Reviews the workflow computes for itself, and grooming of the merged list.

Two jobs, one topic. Some findings do not need a model: counting how many pages a
beat received, or how many pages open with the same word, is arithmetic, and
spending a critic call on it would be an agent doing a ``for`` loop's job. Those
findings are produced here. The list a critic returns is also groomed here before
it reaches a generator.

**How this differs from ``validation.py``, which sits beside it.** The split is
the return type, and it is the whole distinction. ``validation.py`` raises on the
*impossible* -- a page count that disagrees with the brief, a beat with no page,
pages that run backwards. This module returns findings for the *improvable*,
which a generator can act on and which must never kill a run.
"""

from collections import Counter

from sparkstory.entities.reviews import (
    OutlineReview,
    OutlineReviews,
    ProseReview,
    ProseReviews,
    ProseRubric,
)
from sparkstory.entities.stories import PagePlan, StoryOutline, StoryProse
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


def pages_per_beat(outline: StoryOutline, plan: PagePlan) -> dict[int, int]:
    """Count the pages each beat received, including beats that received none.

    Seeded from the outline rather than returned as a bare ``Counter``: a beat
    with no pages must appear as ``0`` rather than vanish, since that is the case
    most worth seeing. This also runs on plans ``validate_page_plan`` has not
    approved yet, so it must not raise on the way there.
    """
    counts = Counter(page.beat_position for page in plan.pages)
    return {beat.position: counts.get(beat.position, 0) for beat in outline.beats}


def format_pacing_report(outline: StoryOutline, plan: PagePlan) -> str:
    """Render pages-per-beat as one log line, keyed by narrative function.

    Keyed by function rather than position because "beat 3 got 4 pages" is data
    and "climax=4" is a finding. Session 2's book gave the climax one page and
    the setup two, which is the imbalance this makes visible.

    **Reported, never enforced.** There is no loop over the page plan to route a
    finding to, and an unbalanced climax is bad rather than impossible -- raising
    ``StoryStructureError`` over a soft preference would kill an otherwise usable
    book. How often this comes out lopsided is what tells a later session whether
    a plan loop earns its keep, which is the same reasoning that keeps
    ``_retry_on`` from retrying structural errors.
    """
    counts = pages_per_beat(outline, plan)
    by_function = [
        f"{beat.function.value}={counts[beat.position]}" for beat in outline.beats
    ]
    return "pages per beat: " + " ".join(by_function)


def drop_unroutable_outline_reviews(
    reviews: OutlineReviews, outline: StoryOutline
) -> OutlineReviews:
    """Discard reviews citing a beat the outline does not have.

    A critic pointing at beat 9 of a four-beat outline has slipped, and there are
    three things one could do about it:

    Raise
        Kills a run over a critic's mistake. Disproportionate -- the other
        findings in the same pass are still perfectly usable.
    Demote it to a story-level finding
        Relocates a finding without saying so, which is worse than losing it: the
        planner then acts on it in the wrong place.
    Pass it through
        Invites the planner to invent a ninth beat so the note makes sense.

    Dropping it and saying so is the only one of the three that is honest, and
    the ``WARNING`` is the point rather than a detail: a critic that has stopped
    making sense must be visible in a log, not quietly filtered on every run.
    """
    positions = {beat.position for beat in outline.beats}
    kept, dropped = [], []
    for review in reviews.reviews:
        # None is "the story as a whole", not a missing beat.
        if review.beat_position is None or review.beat_position in positions:
            kept.append(review)
        else:
            dropped.append(review.beat_position)

    if dropped:
        logger.warning(
            "Dropped %d outline review(s) citing beats %s; the outline has %s",
            len(dropped),
            sorted(dropped),
            sorted(positions),
        )

    return OutlineReviews(outline=reviews.outline, reviews=kept)


#: Below this share of pages, a repeated opening word is not worth a revision
#: pass. Proportional rather than a flat count: three of twenty-four pages is
#: unremarkable, while Session 2's book opened six of eight with a character's
#: name. The floor of 3 stops a four-page book tripping on two.
_REPEATED_OPENING_SHARE = 3


def _opening_word(text: str) -> str:
    """The first word of a page, stripped of punctuation and lowercased.

    Normalised because a listener hears no difference between ``"Maryam,`` and
    ``Maryam!`` -- the drone is identical, and a raw comparison would score them
    as variation.
    """
    words = text.split()
    if not words:
        return ""
    return words[0].strip("\"'“”‘’.,!?;:—-").lower()


def deterministic_prose_reviews(
    prose: StoryProse, page_plan: PagePlan
) -> list[ProseReview]:
    """Findings that can be counted rather than judged.

    Counting these with a model call would be an agent doing a ``for`` loop's
    job. But a check that only raises cannot fix anything, so each finding is
    returned as a review and merges into the same list the Writer edits from.

    ``page_plan`` is accepted but not yet read. It is in the signature because
    that is what the workflow calls, and the next counted finding -- how a page's
    length compares with what its plan asked of it -- needs it. Naming the
    argument now is cheaper than changing every caller later.
    """
    reviews: list[ProseReview] = []

    # Counted on the normalised form so punctuation and case cannot hide a
    # repeat, but reported as first written: quoting 'maryam' at the Writer
    # reads as a typo and invites it to "fix" the capitalisation.
    openings: Counter[str] = Counter()
    as_written: dict[str, str] = {}
    for page in prose.pages:
        key = _opening_word(page.text)
        if not key:
            continue
        openings[key] += 1
        as_written.setdefault(key, page.text.split()[0].strip("\"'“”‘’.,!?;:—-"))

    if openings:
        key, count = openings.most_common(1)[0]
        word = as_written[key]
        threshold = max(_REPEATED_OPENING_SHARE, -(-len(prose.pages) // 3))
        if count >= threshold:
            reviews.append(
                ProseReview(
                    rubric=ProseRubric.READ_ALOUD,
                    # Book-level: no single page is at fault, the pattern is.
                    page_number=None,
                    comment=(
                        f"{count} of the {len(prose.pages)} pages begin with the "
                        f"same word, {word!r}. Read aloud that becomes a drone. "
                        "Vary how pages open: a sound, a question, a line of "
                        "speech, the middle of an action."
                    ),
                )
            )

    return reviews


def drop_unroutable_prose_reviews(
    reviews: ProseReviews, page_plan: PagePlan
) -> ProseReviews:
    """Discard reviews citing a page the book does not have.

    Same three options and the same reasoning as
    :func:`drop_unroutable_outline_reviews`: raising kills a run over a critic's
    slip, demoting to book level relocates the finding silently, and passing it
    through invites the Writer to add a page so the note makes sense.
    """
    numbers = {page.page_number for page in page_plan.pages}
    kept, dropped = [], []
    for review in reviews.reviews:
        # None is "the book as a whole", not a missing page.
        if review.page_number is None or review.page_number in numbers:
            kept.append(review)
        else:
            dropped.append(review.page_number)

    if dropped:
        logger.warning(
            "Dropped %d prose review(s) citing pages %s; the book has %d pages",
            len(dropped),
            sorted(dropped),
            len(numbers),
        )

    return ProseReviews(prose=reviews.prose, reviews=kept)


def draft_score(reviews: list[OutlineReview] | list[ProseReview]) -> tuple[int, int]:
    """Rank a draft by its reviews. Lower is better.

    Two keys, and safety has to dominate: ranking on count alone would let a
    draft carrying a safety finding win a tie against a safe one, which for a
    guardrail is exactly backwards.

    Used to return the *best* draft a loop saw rather than the last. Earned from
    a live run (``outputs/20260730-232426-*``) where the prose loop oscillated
    5 -> 3 -> 3 findings and hit its cap: pages 3 and 6 were better in draft two
    than in draft three, and returning the last draft shipped the worse one. The
    critic cannot tell "shown subtly" from "absent", so it flags a good page as
    missing its feeling and the Writer names it again. No rubric wording fully
    solves that; keeping the best draft caps the damage.
    """
    unsafe = any(r.rubric is ProseRubric.SAFETY for r in reviews)
    return (int(unsafe), len(reviews))
