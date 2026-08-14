"""How a book looks, and what was actually drawn.

Same three rules as ``entities/stories.py``: docstrings and field descriptions
here are **prompt text** (they become the JSON schema the Illustration Director
is bound to), enums constrain the model rather than only the code, and validation
constraints on model output are load-bearing.

Two halves, and the split matters. ``IllustrationPlan`` is *model output* -- an
agent decides how the book will look. ``StoryArt`` is *our record* of what came
back from the image provider, so nothing in it is prompt text and nothing in it is
ever bound as an output schema. Mixing the two would mean a model writing into the
fields we use to decide whether the feature worked.

**``ConsistencyVerdict`` is the one exception, and it is a real one.** It is bound
as an output schema, so its descriptions are prompt text, and it is *stored on an
``ArtItem``* -- a model writing into our record, which the paragraph above says
not to do. The reason it is allowed here is that the record needs to carry a
judgement nothing else can produce: whether a picture matches its reference is not
observable from a provider response. The guard is that it goes in its own field
which defaults to ``None``, so a model can only ever add a verdict beside our
facts and never overwrite ``status``, ``path`` or ``detail``. Keep that boundary:
if a future judge wants to change what ``status`` says, it is doing our job.

**Three field descriptions here exist to close a rule 13 trap.** Each was written
by asking what the laziest thing that satisfies the field is:

* ``appearance`` -- the lazy answer is "a majestic fox with soulful amber eyes":
  evaluative, unmatchable by a second artist. The description demands
  distinguishing, drawable specifics and rules out judgement words.
* ``style_bible`` -- the lazy answer is naming a genre, "watercolour children's
  book illustration", which constrains nothing across eight pages. It has to
  commit to palette, line and light, because that specificity is the only thing
  holding settings consistent; the portraits only cover characters.
* ``prompt`` -- the lazy answer is restating the page's prose.
  ``ScenePlan.visual_action`` already carries "write notes, never sentences" for
  this reason. Image prompts need one guard more: no story text, or the model
  renders words into the picture, which cannot be fixed afterwards.

**Paths, not bytes.** ``StoryArt`` holds file paths. Base64 image data in a
Pydantic model would reach ``story.json``, every log line and every run artifact
-- finding O already records the web ledger storing whole page snippets as
something to truncate. A path is cheap and checkable.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

#: xAI's ``/v1/images/edits`` requires **exactly two** source images. Measured, not
#: read: one reference returns 422 ("invalid length 1, expected struct ImageUrl
#: with 3 elements") and three returns 400 ("cannot set both 'url' and 'file_id'").
#: Two distinct portraits do genuinely compose -- verified by generating a fox and
#: a girl separately and getting both, recognisably, into one picture.
#:
#: Documentation said "up to 3 source images" and a web search asserted a model id
#: this account cannot even see, so both numbers here come from probing the live
#: endpoint. `ScenePlan` allows six characters, which makes this a real constraint
#: rather than a tidy limit: a three-character page cannot be fully
#: reference-conditioned, and the workflow has to choose what to do about it
#: rather than discover it at runtime.
MAX_REFERENCE_IMAGES = 2


class CharacterAppearance(BaseModel):
    """What one character looks like, every time they are drawn."""

    name: str = Field(
        min_length=1,
        max_length=60,
        description=(
            "The character's name, copied exactly from the story's cast. Do not "
            "rename anyone and do not invent a character who is not in it."
        ),
    )
    appearance: str = Field(
        min_length=20,
        max_length=300,
        description=(
            "Two or three specific, drawable features another artist could match "
            "without seeing your picture: build, colouring, markings, and one "
            "thing worn or carried that never changes. Write only what is "
            "visible. Do not write what the character feels, wants or is like, "
            "and do not use words that judge rather than describe -- 'majestic', "
            "'soulful', 'adorable' tell an artist nothing to draw."
        ),
    )
    portrait_prompt: str = Field(
        min_length=20,
        max_length=400,
        description=(
            "A prompt for one reference portrait of this character alone: the "
            "whole body, standing, facing forward, against a plain background, "
            "no scenery and no other characters. This image is what every page "
            "is drawn from, so describe the character and nothing else."
        ),
    )


class PageArt(BaseModel):
    """The picture on one page."""

    page_number: int = Field(
        ge=1,
        description="Which page of the book this picture belongs to.",
    )
    prompt: str = Field(
        min_length=20,
        max_length=400,
        description=(
            "What this one picture shows: the setting, who is in it, what they "
            "are doing, and where the light comes from. Describe the image as "
            "notes, never as sentences from the story. Never include any words "
            "that should appear as text, a title, a caption or a letter in the "
            "picture -- this picture carries no writing of any kind."
        ),
    )
    # `max_length` is the provider's own limit, and a schema cap rather than a
    # runtime truncation on purpose: silently dropping the fourth character would
    # produce a picture missing someone the page needs, with nothing to read
    # afterwards explaining why. Better that the model chooses.
    characters_present: list[str] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_IMAGES,
        description=(
            "Names of at most three characters this picture centres on, copied "
            "exactly. If the page involves more, choose the three the picture is "
            "really about. Leave empty for a picture with nobody in it."
        ),
    )


class IllustrationPlan(BaseModel):
    """How this book will look, decided once for all of its pages."""

    style_bible: str = Field(
        min_length=50,
        max_length=500,
        description=(
            "The look every page shares, specific enough that two artists "
            "working apart would produce matching pictures. Commit to a named "
            "palette of a few colours, how lines and edges are drawn, where "
            "light comes from, and how full or empty backgrounds are. Naming a "
            "style or a medium alone is not enough -- 'watercolour picture book' "
            "describes thousands of different-looking books."
        ),
    )
    characters: list[CharacterAppearance] = Field(
        min_length=1,
        max_length=6,
        description=(
            "Every character who appears in a picture, each described once. "
            "Decide here what they look like, because these descriptions are "
            "what keeps them the same person on every page."
        ),
    )
    pages: list[PageArt] = Field(
        min_length=1,
        max_length=24,
        description=(
            "One entry per page of the book, in order, with no page left out "
            "and none repeated."
        ),
    )


class ConsistencyAttribute(StrEnum):
    """What differs between a picture and the reference it should match.

    A closed list rather than free text, for the reason ``ReviewRubric`` is an enum:
    it lets code branch on the kind of difference, and it stops "is this
    consistent?" from being answered as a general impression.

    The order is not arbitrary and it is the opposite of what this rubric would
    have been written to check. Across three illustrated runs, faces, body plans,
    ears, tails and props all carried across untouched and **only colour moved** --
    a fox's white paws came back black, a green ant came back black. So colour is
    named first here and first in the prompt, and ``identity`` is last because it
    has never once happened.
    """

    COLOUR = "colour"  # the most common drift by far; see the class docstring
    MARKINGS = "markings"  # a white chest, a black tail tip, patterned pyjamas
    PROP = "prop"  # a collar, a charm, a scarf, a basket
    BODY_PLAN = "body_plan"  # leg count, segments, proportions
    IDENTITY = "identity"  # a different character altogether


class ConsistencyVerdict(BaseModel):
    """Whether one picture shows the same character as its reference.

    Bound as an output schema, so everything visible here is prompt text -- which
    makes this the one exception to the module docstring's rule that our records
    carry no prompt text. It is stored on an ``ArtItem`` and written by a model.

    ``difference`` exists because a verdict alone is worthless: asked "does this
    match?", the cheapest defensible answer is always *yes*, and a judge that
    agrees costs a call and buys nothing. Requiring the difference to be named
    makes an agreeable answer falsifiable -- the same move the web-claim design
    makes by requiring a supporting quote, and the reason attribution is
    overwritten from the store rather than trusted.
    """

    matches: bool = Field(
        description=(
            "True only if the picture shows the same character as the reference. "
            "Judge what the image actually shows, not what the description says "
            "it should show."
        )
    )
    attribute: ConsistencyAttribute | None = Field(
        default=None,
        description=(
            "Which aspect differs. Null only when the picture matches. Check "
            "colour first: it is the difference that occurs most often."
        ),
    )
    difference: str = Field(
        default="",
        description=(
            "One sentence naming what the picture shows against what the "
            "reference asked for -- for example 'the collar is gold in the "
            "picture and silver in the reference'. Do not restate the reference "
            "as though it were what you see. Empty only when the picture matches."
        ),
    )


# --- Our record of what happened, never model output -------------------------


class ArtStatus(StrEnum):
    """Whether one image exists, and how much of the design produced it.

    Three states rather than a boolean, for the reason ``Evidence`` earned itself
    on its first live run: a boolean would collapse "drawn from reference
    portraits" and "drawn from a text prompt alone" into the same ``true`` and
    lose the distinction that says whether this feature worked.
    """

    CONDITIONED = "conditioned"  # generated against reference portraits
    UNCONDITIONED = "unconditioned"  # generated, but with no reference available
    FAILED = "failed"  # not generated; this frame is blank


class ArtItem(BaseModel):
    """One image we asked for, and what became of it."""

    # Not prompt text: an ArtItem is built by us from a provider response and is
    # never bound as an output schema.
    key: str  # a page number as a string, or a character's name
    status: ArtStatus
    path: Path | None = None
    detail: str | None = None  # why it failed, or which references were used
    #: The judge's verdict, or None when this image was never judged -- which is
    #: honest for an UNCONDITIONED or FAILED item (there is no reference to match,
    #: or no image at all) and for any run with judging switched off.
    #:
    #: A separate field rather than a fourth `ArtStatus`, and the distinction is
    #: load-bearing. `status` says what the pipeline *did*; this says whether it
    #: *worked*. Collapsing them would drag `fully_conditioned` -- which is
    #: `all(status is CONDITIONED)`, and which four call sites read, one of them a
    #: prompt that tells a client to report it to the parent -- from "the
    #: consistency mechanism ran" to "the mechanism ran and produced a match".
    #: Those are different claims and the property's docstring commits to the first.
    consistency: ConsistencyVerdict | None = None


class StoryArt(BaseModel):
    """Every image for one book, and an honest account of the run.

    The point of this object is answering *"was this book actually
    reference-conditioned?"* by reading a file rather than by looking at the
    pictures. Finding N is why: the degraded web path produced plausible output
    and nothing in it could have flagged the degradation.
    """

    style_bible: str
    portraits: list[ArtItem] = Field(default_factory=list)
    pages: list[ArtItem] = Field(default_factory=list)

    def page_image(self, page_number: int) -> Path | None:
        """The image to draw on ``page_number``, or None if there is none.

        The renderer's whole interface to this object. Returning None for both
        "failed" and "never asked for" is deliberate: an absent image and a failed
        one produce the same page, so giving the renderer two ways to say the same
        thing would only invite it to branch on a distinction with no consequence.
        """
        for item in self.pages:
            if item.key == str(page_number):
                return item.path
        return None

    @property
    def fully_conditioned(self) -> bool:
        """True when every image exists *and* was drawn from reference portraits.

        The check the acceptance criteria run. Deliberately strict -- one
        unconditioned page makes this False -- because the failure this guards
        against is a book that looks finished while the consistency mechanism
        silently did not run.
        """
        every = self.portraits + self.pages
        return bool(every) and all(i.status is ArtStatus.CONDITIONED for i in every)

    @property
    def fully_consistent(self) -> bool:
        """True when nothing that was judged came back as a mismatch.

        The companion to ``fully_conditioned``, and deliberately a *second*
        property rather than a stricter version of the first. That one answers
        "did the consistency mechanism run?"; this answers "did it work?". A book
        can honestly be `fully_conditioned` and not `fully_consistent` -- that is
        precisely the case three live runs produced, where every item recorded
        `conditioned` while a fox's paws changed colour between its portrait and
        its pages.

        Unjudged items do not count against it: ``None`` means nobody looked, and
        reporting "inconsistent" for a book nobody judged would be the same
        overclaim in the other direction. So this is True for a run with judging
        off, which is why it must be read alongside how many verdicts exist rather
        than alone -- rule 24, a check that cannot fail proves nothing.
        """
        judged = [i.consistency for i in self.portraits + self.pages if i.consistency]
        return all(v.matches for v in judged)
