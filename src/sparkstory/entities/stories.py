"""Domain schemas for storybook generation.

These models are the contract the rest of the system is built on. Three rules
govern edits here.

**1. Docstrings and field descriptions are prompt text.** When a model is bound
with ``.with_structured_output(StoryOutline)``, Pydantic's JSON schema is sent
to the LLM -- and a class docstring becomes the schema's ``description``, while
each ``Field(description=...)`` becomes a per-property instruction. The same is
true of the input models, whose schema becomes the MCP tool signature that a
client agent reads. So docstrings here are written as concise, model-facing
directives, and **engineering rationale goes in ``#`` comments**, which never
reach the model. (This module docstring is the exception: it is never part of a
schema.)

**2. Enums constrain the model, not just the code.** A free-text ``tone`` field
invites "whimsically melancholic"; an enum gives a fixed vocabulary and lets
downstream code branch deterministically.

**3. Validation constraints on LLM output are load-bearing.** A model returning
three beats when four is the minimum raises ``ValidationError`` rather than
quietly producing a thin story. In this session that is desired: fail loudly.
The evaluator-optimizer loop in a later session converts such failures into a
retry carrying feedback.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# Safe in this direction only: `entities/grounding.py` imports nothing from
# sparkstory, so there is no cycle. Checked rather than assumed -- a cycle here
# would surface as an ImportError at server start, not at edit time.
from sparkstory.entities.grounding import StoryGrounding


# Defaulting to they/them is a correctness decision, not a stylistic one: a name
# does not indicate someone's pronouns, and the story text refers to the child
# throughout. Guessing would misgender a real child in a book written
# specifically for them.
class Pronouns(StrEnum):
    """Pronouns to use for the child in the story."""

    SHE_HER = "she/her"
    HE_HIM = "he/him"
    THEY_THEM = "they/them"


# Kept separate from `age` because the two diverge often -- a precocious
# four-year-old and a reluctant seven-year-old need different books than their
# ages alone suggest. The age hints below are guidance for the caller, not rules.
class ReadingLevel(StrEnum):
    """Target vocabulary and sentence complexity."""

    PRE_READER = "pre_reader"  # read aloud to the child; ages ~2-4
    EARLY_READER = "early_reader"  # short simple sentences; ages ~4-6
    DEVELOPING = "developing"  # longer sentences, some new words; ages ~6-8
    CONFIDENT = "confident"  # paragraphs, richer vocabulary; ages ~8-10


class Tone(StrEnum):
    """Emotional register of the story."""

    GENTLE = "gentle"  # calm, reassuring -- suits bedtime
    FUNNY = "funny"  # silly, playful
    ADVENTUROUS = "adventurous"  # brave, energetic
    MAGICAL = "magical"  # wondrous, dreamlike
    HEARTWARMING = "heartwarming"  # kind, friendship-focused


# A different axis from `Tone`, and the pair is easy to conflate. Tone is
# register -- how the story feels. This is physics -- whether the world may be
# broken. A gentle story can break physics; an adventurous one can be strictly
# real, so neither implies the other.
class WorldRules(StrEnum):
    """How far the story's world must obey the real one."""

    REALISTIC = "realistic"  # every retrieved fact holds
    IMAGINATIVE = "imaginative"  # facts are detail; the premise may break them


# Naming the structural roles explicitly gives the planner a vocabulary for
# story shape, rather than leaving it to emit an undifferentiated list of events.
class NarrativeFunction(StrEnum):
    """The structural job a beat performs in the story."""

    SETUP = "setup"
    INCITING_INCIDENT = "inciting_incident"
    RISING_ACTION = "rising_action"
    MIDPOINT = "midpoint"
    CLIMAX = "climax"
    RESOLUTION = "resolution"


class ChildProfile(BaseModel):
    """The child the story is written for."""

    name: str = Field(
        min_length=1,
        max_length=40,
        description="The child's first name, used as the main character's name.",
    )
    age: int = Field(
        ge=2,
        le=12,
        description="The child's age in years.",
    )
    pronouns: Pronouns = Field(
        default=Pronouns.THEY_THEM,
        description="Pronouns to use for the child throughout the story.",
    )
    reading_level: ReadingLevel = Field(
        default=ReadingLevel.EARLY_READER,
        description="Target vocabulary and sentence complexity.",
    )
    interests: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Things this child loves, for example 'foxes', 'astronomy', "
            "'digging in the garden'. Weave these into the story naturally "
            "rather than listing them."
        ),
    )


# Separate from StoryOutline on purpose: user intent and model output must never
# accumulate in one mutable object, or it stops being possible to tell which is
# which.
class StoryBrief(BaseModel):
    """A request to generate a story."""

    child: ChildProfile = Field(
        description="The child this story is written for.",
    )
    premise: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "The story idea in the user's own words, for example 'a fox who "
            "wants to visit the moon'."
        ),
    )
    tone: Tone = Field(
        default=Tone.GENTLE,
        description="The emotional register the story should hold.",
    )
    # The default is IMAGINATIVE, and that is a behaviour change: a caller who
    # supplies nothing gets different planning than before this field existed.
    # Chosen from evidence rather than compatibility -- on the standing "fox who
    # wants to visit the moon" premise with tone=magical, the realistic
    # rendering planned three failed launches and resolved with the child
    # holding a paper tube up to the Moon, while the ungrounded control let the
    # rocket fly. Anything wanting the old behaviour asks for realistic.
    world_rules: WorldRules = Field(
        default=WorldRules.IMAGINATIVE,
        description=(
            "How far this story's world must obey the real one. Choose "
            "'realistic' when getting the real world right is part of the "
            "point, and the story should never contradict it. Choose "
            "'imaginative' when the idea itself is impossible -- a fox who "
            "flies to the Moon -- and real-world detail is there to make the "
            "impossible parts feel believable."
        ),
    )
    # page_count belongs to the finished book, not to the outline: beats are
    # mapped onto pages by a later stage. Keeping them uncoupled means changing
    # the page count does not invalidate an otherwise good outline.
    page_count: int = Field(
        default=12,
        ge=4,
        le=24,
        description="How many pages the finished book should have.",
    )
    must_include: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Details the user explicitly wants present in the story.",
    )
    # Captured at intake rather than filtered out of generated output: it is far
    # cheaper, and it gives the safety critic concrete per-child criteria instead
    # of only generic rules.
    avoid: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Subjects to keep out of the story entirely, for example 'spiders', "
            "'the dark', 'losing a pet'. Treat these as hard constraints."
        ),
    )


# Deliberately thin. A later stage expands these into full character sheets with
# visual references; the planner only needs enough to write a coherent outline.
# Asking it for appearance details here produces confident invention that the
# illustration stage then contradicts -- hence the explicit instruction not to.
class CharacterSketch(BaseModel):
    """A character in the story."""

    name: str = Field(
        min_length=1,
        max_length=40,
        description="The character's name.",
    )
    role: str = Field(
        min_length=1,
        max_length=60,
        description=(
            "This character's job in the story, for example 'main character', "
            "'loyal companion', 'gentle guide'."
        ),
    )
    description: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "One or two sentences on who they are and what they want. Do not "
            "describe their appearance, clothing or colours."
        ),
    )


class StoryBeat(BaseModel):
    """One structural unit of the story."""

    position: int = Field(
        ge=1,
        description="This beat's 1-based order in the story.",
    )
    function: NarrativeFunction = Field(
        description="The structural job this beat performs.",
    )
    title: str = Field(
        min_length=1,
        max_length=80,
        description="A short label for this beat, for example 'The paper rocket'.",
    )
    summary: str = Field(
        min_length=10,
        max_length=600,
        description=(
            "What happens in this beat, in two or three sentences. Describe "
            "events and feelings, not finished prose."
        ),
    )
    characters_present: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Names of characters appearing in this beat.",
    )


# `beats` is bounded at both ends: too few and the story has no shape, too many
# and later stages spend tokens on structure a picture book cannot carry.
class StoryOutline(BaseModel):
    """A complete plan for a story, before any prose is written."""

    title: str = Field(
        min_length=1,
        max_length=80,
        description=(
            "The book's title. Make it appealing to the child and readable at "
            "their reading level."
        ),
    )
    logline: str = Field(
        min_length=10,
        max_length=250,
        description="One sentence capturing the whole story, as on a back cover.",
    )
    theme: str = Field(
        min_length=3,
        max_length=120,
        description=(
            "The idea the story explores, for example 'being brave enough to "
            "try something new'. State it plainly; do not moralise."
        ),
    )
    characters: list[CharacterSketch] = Field(
        min_length=1,
        max_length=6,
        description=(
            "Every character in the story. The child is always the main "
            "character and must appear here."
        ),
    )
    beats: list[StoryBeat] = Field(
        min_length=4,
        max_length=8,
        description=(
            "The story's structure in order. Include a setup, an inciting "
            "incident, a climax and a resolution at minimum."
        ),
    )
    # Research that shaped this plan, carried so the Writer can obey the same
    # constraints the planner did. Before this the grounding died at the end of
    # `plan_story`: it was computed, planned from, and dropped when the pipeline
    # returned a bare outline -- which is why a craft device could only ever be
    # *described* in a beat summary (findings J and Q). A refrain lives in prose.
    #
    # Nested here rather than passed beside the outline as a third `write_story`
    # argument. Three independently-assembled arguments let a client pair a genuine
    # outline with a genuine grounding from a *different* `plan_story` call, and
    # nothing would compare them. That matters more than it sounds because
    # `world_rules` lives on the brief: changing it re-runs retrieval and re-frames
    # the result, so the same chunk means "the eagle cannot fly" under `realistic`
    # and "empty space is what the wings push against" under `imaginative`. A
    # grounding paired with the wrong brief is not stale, it is wrong. Nesting makes
    # that pairing unrepresentable rather than merely discouraged.
    #
    # Optional because `MAX_RESEARCH_STEPS=0` is a supported configuration and an
    # ungrounded run is the control arm of the A/B this feature is judged by.
    #
    # NOT filled by the planner, and the description below says so because a field
    # description is prompt text (non-obvious rule 1). The planner is the one node
    # that both sees this schema and could invent a `chunk_id` -- which
    # `drop_unprovenanced` would then silently drop, losing a real fact with no
    # error anywhere. Its revision replay excludes this field for the same reason;
    # see `nodes/story_planner.py`.
    grounding: StoryGrounding | None = Field(
        default=None,
        description="Leave this out. It is filled in by the research step, not by you.",
    )


# `beat_position` is what makes this plan checkable rather than merely plausible:
# code can assert every beat received a page, that no page cites a beat that does
# not exist, and that the pages do not wander back and forth through the
# structure. Without it, "did it drop the climax?" can only be answered by reading.
#
# The three note fields replace a single `scene_summary`, whose description asked
# for "one or two sentences ... not finished prose" -- a self-contradiction the
# model resolved by writing prose, which the Writer then paraphrased on four of
# eight pages. Three orthogonal notes cannot be concatenated into a page, so the
# Writer has to write. Splitting them also makes interiority a required field
# rather than something that can be silently dropped, and gives the page turn a
# slot of its own.
class ScenePlan(BaseModel):
    """One page of the book."""

    page_number: int = Field(
        ge=1,
        description="This page's 1-based position in the book.",
    )
    beat_position: int = Field(
        ge=1,
        description="The position of the story beat this page draws from.",
    )
    setting: str = Field(
        min_length=1,
        max_length=120,
        description="Where this page happens, in a few words.",
    )
    visual_action: str = Field(
        min_length=5,
        max_length=300,
        description=(
            "What the picture shows: the one action or image on this page, "
            "drawable as a single image. Write notes, never sentences from the "
            "story. 'rocket tips over, Pip's ears flatten' -- not 'The rocket "
            "tipped over and Pip's ears flattened.'"
        ),
    )
    emotional_shift: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "What changes inside the main character on this page: what they "
            "feel, notice or decide. A few words, not a sentence from the story. "
            "Every page changes something, even if only a little."
        ),
    )
    page_turn_hook: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "The question this page leaves unanswered, answered after the page "
            "turn. A few words. Leave empty on the final page, which answers "
            "rather than asks."
        ),
    )
    characters_present: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Names of the characters who appear on this page.",
    )


class PagePlan(BaseModel):
    """A story laid out page by page, before any prose is written."""

    pages: list[ScenePlan] = Field(
        min_length=4,
        max_length=24,
        description=(
            "Every page of the book, in order. Give a beat more pages when it "
            "needs room to breathe and fewer when it does not; turning the page "
            "is itself part of the drama."
        ),
    )


class StoryPage(BaseModel):
    """The words printed on one page."""

    page_number: int = Field(
        ge=1,
        description="The page these words belong to.",
    )
    # Bounds are deliberately generous rather than reading-level-specific: a Field
    # constraint is static, while the right length varies by ReadingLevel. Judging
    # length against the level belongs to the reading-level rubric in a later
    # session, which can give feedback instead of only rejecting.
    text: str = Field(
        min_length=1,
        max_length=1200,
        description="The words on this page, written to be read aloud.",
    )


class StoryProse(BaseModel):
    """The finished words of a story."""

    pages: list[StoryPage] = Field(
        min_length=4,
        max_length=24,
        description="One entry per page of the plan, in order, none left out.",
    )


# The workflow assembles this; no LLM ever returns it. So unlike every other model
# in this module, its docstring and field names are NOT prompt text -- which is
# also why it needs no Field descriptions.
class Story(BaseModel):
    """A finished story: the plan it was built from and the words it became."""

    outline: StoryOutline
    page_plan: PagePlan
    # Kept as the pages themselves rather than the StoryProse wrapper: the wrapper
    # exists only because structured output needs a top-level model.
    pages: list[StoryPage]
