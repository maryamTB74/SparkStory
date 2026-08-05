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
