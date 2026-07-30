"""Checks a schema cannot express.

Pydantic validates a model's output against its own shape. It cannot compare that
output to the *request*: nothing in ``PagePlan`` knows how many pages the brief
asked for, or which beats existed to be covered. These functions close that gap.

**Why the workflow calls these and the nodes do not.** A node's job is to report
what the model said. Deciding what to do about a malformed answer is
orchestration. Keeping the check in the task is what lets a later session wrap it
in a retry that carries the message back as feedback, instead of unpicking a
raise buried inside an agent.

Every failure raises ``StoryStructureError`` rather than returning a report,
because there is nothing sensible to do with a broken plan today. Loud failure is
the design: a story quietly missing its climax is far worse than a visible error.
"""

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.stories import PagePlan, StoryBrief, StoryOutline, StoryProse


def validate_outline(brief: StoryBrief, outline: StoryOutline) -> None:
    """Check an outline can actually be laid out across the requested pages.

    One rule, and it exists because the alternative is an impossible request. A
    page shows one moment, so a beat needs at least one page; therefore an outline
    with more beats than the book has pages cannot be paged at all. ``StoryBrief``
    permits 4-24 pages and ``StoryOutline`` permits 4-8 beats, so this state is
    reachable with entirely valid inputs -- a 5-page book with a 6-beat outline.

    Caught here rather than at the page plan, where it surfaced first as "beat 6
    has no page". That message blamed the Plot Planner for failing an impossible
    instruction and sent debugging one stage too late.

    Raises:
        StoryStructureError: the outline has more beats than the brief has pages.
    """
    if len(outline.beats) > brief.page_count:
        raise StoryStructureError(
            f"Outline has {len(outline.beats)} beats but the book has only "
            f"{brief.page_count} pages. A beat needs at least one page, so this "
            "outline cannot be laid out. Ask for more pages, or plan fewer beats."
        )


def validate_page_plan(
    brief: StoryBrief, outline: StoryOutline, plan: PagePlan
) -> None:
    """Check a page plan against the brief that asked for it.

    Raises:
        StoryStructureError: on any of four failures -- a page count that
            disagrees with the brief, a page citing a beat that does not exist, a
            beat that received no page at all, or pages that move backwards
            through the story.
    """
    pages = plan.pages

    if len(pages) != brief.page_count:
        raise StoryStructureError(
            f"Page plan has {len(pages)} pages but the brief asked for "
            f"{brief.page_count}."
        )

    beat_positions = {beat.position for beat in outline.beats}

    unknown = sorted({p.beat_position for p in pages} - beat_positions)
    if unknown:
        raise StoryStructureError(
            f"Page plan cites beats {unknown}, which are not in the outline. "
            f"Valid beat positions: {sorted(beat_positions)}."
        )

    # A dropped beat is invisible in the output: the plan still reads plausibly,
    # and the finished book is simply missing part of its story.
    uncovered = sorted(beat_positions - {p.beat_position for p in pages})
    if uncovered:
        raise StoryStructureError(
            f"Beats {uncovered} have no page. Every beat must appear on at least "
            "one page."
        )

    # Catches a model that shuffled the structure while producing pages that each
    # look fine on their own.
    cited = [p.beat_position for p in pages]
    if cited != sorted(cited):
        raise StoryStructureError(
            f"Pages move backwards through the story: beat order is {cited}. "
            "A picture book must stay in story order."
        )


def validate_prose(page_plan: PagePlan, prose: StoryProse) -> None:
    """Check the written pages line up with the plan.

    Raises:
        StoryStructureError: if the page numbers do not match the plan's exactly,
            including duplicates and omissions, or a page's text is blank.
    """
    planned = [p.page_number for p in page_plan.pages]
    written = [p.page_number for p in prose.pages]

    if written != planned:
        raise StoryStructureError(
            f"Prose covers pages {written} but the plan has {planned}."
        )

    # `min_length=1` on the field rejects "" but not "   ", and a page of spaces
    # renders as a blank page in a finished book.
    blank = [p.page_number for p in prose.pages if not p.text.strip()]
    if blank:
        raise StoryStructureError(f"Pages {blank} have no text.")
