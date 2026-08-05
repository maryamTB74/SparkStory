"""The illustration workflow, its degradation, and the art it puts in a PDF.

Offline throughout: the chat model is a `FakeModel` and the image provider is a
`FakeImageProvider`, so nothing here reaches a network.

The tests that matter most are the *failure* ones. Illustration is the only stage
in this project that fails soft, so "a broken image leaves a blank frame and the
book still renders" is a behaviour rather than an accident, and finding N is the
worked example of why a degraded path nobody exercised is a path nobody can trust.
"""

from pathlib import Path

import pytest

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.illustration import (
    ArtItem,
    ArtStatus,
    CharacterAppearance,
    IllustrationPlan,
    PageArt,
    StoryArt,
)
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.models.fake_image_model import FakeImageProvider
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.illustration_director import (
    IllustrationDirectorNode,
    render_illustration_request,
)
from sparkstory.renderers import render_pdf
from sparkstory.workflows import illustrate as illustrate_module
from sparkstory.workflows.illustrate import run_illustration_pipeline
from sparkstory.workflows.validation import validate_illustration_plan


def _plan(pages: int = 10, characters: tuple[str, ...] = ("Kit",)) -> IllustrationPlan:
    """A minimal plan whose fields all satisfy their length constraints."""
    return IllustrationPlan(
        style_bible=(
            "Flat shapes, no outlines. Palette: burnt orange, cream, deep navy. "
            "Light always from the upper left. Backgrounds nearly empty."
        ),
        characters=[
            CharacterAppearance(
                name=name,
                appearance=(
                    f"{name} is a small orange fox with one white ear tip and a "
                    "blue knitted scarf worn on every page."
                ),
                portrait_prompt=(
                    f"{name}, a small orange fox with a blue scarf, full body, "
                    "standing, facing forward, plain background"
                ),
            )
            for name in characters
        ],
        pages=[
            PageArt(
                page_number=n,
                prompt=f"page {n}: the fox beside a paper rocket in tall grass",
                characters_present=list(characters),
            )
            for n in range(1, pages + 1)
        ],
    )


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> FakeImageProvider:
    """Install a fake image provider and return it for assertions."""
    fake = FakeImageProvider()
    monkeypatch.setattr(
        illustrate_module, "get_image_model", lambda _model_id: fake.as_model()
    )
    return fake


@pytest.fixture
def directed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Director's chat model, so no LLM is needed."""
    monkeypatch.setattr(
        illustrate_module,
        "get_chat_model",
        lambda _model_id: FakeModel(_plan()),
    )


class TestTheDirectorPrompt:
    def test_includes_the_words_on_each_page(
        self, brief: StoryBrief, story: Story
    ) -> None:
        """A picture contradicting the words on its own page is the one error a
        reader cannot miss, so the prose is in the prompt, not only the plan."""
        rendered = render_illustration_request(brief, story)

        for page in story.pages:
            assert page.text in rendered

    def test_includes_the_avoid_list(self, brief: StoryBrief) -> None:
        """A parent's exclusions govern pictures too -- an illustration is the part
        of a picture book a pre-reader consumes unaided."""
        assert brief.avoid, "fixture must have exclusions for this to mean anything"
        rendered = render_illustration_request(brief, _story_for(brief))

        for avoided in brief.avoid:
            assert avoided in rendered

    def test_binds_the_illustration_plan_schema(
        self, brief: StoryBrief, story: Story
    ) -> None:
        model = FakeModel(_plan())
        IllustrationDirectorNode(model=model, brief=brief, story=story)

        assert model.bound_schema is IllustrationPlan


def _story_for(brief: StoryBrief) -> Story:
    """Build a Story matching a brief, for the prompt tests that need one."""
    from sparkstory.entities.stories import (
        CharacterSketch,
        NarrativeFunction,
        PagePlan,
        ScenePlan,
        StoryBeat,
        StoryOutline,
        StoryPage,
    )

    outline = StoryOutline(
        title="Kit and the Paper Rocket",
        logline="Maryam builds a paper rocket to send her friend Kit to the moon.",
        theme="turning a wish into something you can send into the sky",
        characters=[
            CharacterSketch(
                name="Maryam",
                role="protagonist",
                description="A five-year-old who wants to reach the moon.",
            ),
            CharacterSketch(
                name="Kit",
                role="friend",
                description="A fox who keeps her company.",
            ),
        ],
        beats=[
            StoryBeat(
                position=n,
                function=NarrativeFunction.SETUP,
                title=f"beat {n}",
                summary=f"something happens, number {n}",
                characters_present=["Maryam"],
            )
            for n in range(1, 5)
        ],
    )
    page_plan = PagePlan(
        pages=[
            ScenePlan(
                page_number=n,
                beat_position=n,
                setting="a garden at night",
                visual_action="rocket tips over, Kit's ears flatten",
                emotional_shift="hope turns to doubt",
                characters_present=["Maryam", "Kit"],
            )
            for n in range(1, 5)
        ]
    )
    return Story(
        outline=outline,
        page_plan=page_plan,
        pages=[StoryPage(page_number=n, text=f"Page {n} words.") for n in range(1, 5)],
    )


class TestConditioning:
    async def test_pages_are_edited_against_portraits(
        self,
        brief: StoryBrief,
        story: Story,
        provider: FakeImageProvider,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """The whole feature: a page is drawn *from* its characters' portraits."""
        art = await run_illustration_pipeline(brief, story, tmp_path)

        assert provider.generate_prompts, "a portrait must be generated first"
        assert provider.edit_calls, "pages must be edited, not generated"
        assert art.fully_conditioned

    async def test_the_style_bible_reaches_every_prompt(
        self,
        brief: StoryBrief,
        story: Story,
        provider: FakeImageProvider,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """One shared look is the consistency mechanism for everything the
        portraits do not cover."""
        await run_illustration_pipeline(brief, story, tmp_path)

        every_prompt = provider.generate_prompts + [p for p, _ in provider.edit_calls]
        for prompt in every_prompt:
            assert "burnt orange" in prompt

    async def test_every_prompt_forbids_text_in_the_image(
        self,
        brief: StoryBrief,
        story: Story,
        provider: FakeImageProvider,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """Words drawn into an illustration cannot be corrected afterwards."""
        await run_illustration_pipeline(brief, story, tmp_path)

        every_prompt = provider.generate_prompts + [p for p, _ in provider.edit_calls]
        for prompt in every_prompt:
            assert "No text" in prompt

    async def test_a_page_prompt_says_what_each_character_looks_like(
        self,
        brief: StoryBrief,
        story: Story,
        provider: FakeImageProvider,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """Finding U, from the first live run, and the reason this test exists.

        A page prompt that only *names* its characters relies entirely on the
        reference image to carry identity. That held for the child and failed for the
        fox, who was drawn as a cat and then a dog while the portrait was a correct
        fox the whole time. Identity has to be in the text as well as the image.
        """
        await run_illustration_pipeline(brief, story, tmp_path)

        for prompt, _ in provider.edit_calls:
            assert "small orange fox" in prompt, (
                "the page prompt must restate the appearance, not just the name"
            )

    # An undescribed character used to be tolerated here, on the reasoning that the
    # picture is still drawn. `validate_illustration_plan` now rejects it, and the
    # validator is right: silently dropping the description is finding U's exact
    # cause. The behaviour is asserted in TestValidatingThePlan instead. Two tests
    # asserting opposite things about one input is how a design decision gets lost.

    async def test_pages_are_bounded_by_the_providers_rate_limit(
        self,
        brief: StoryBrief,
        story: Story,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Finding T: six concurrent requests hit `429 ... actual/limit 5/5`.

        A retry alone cannot fix an over-subscription -- it reproduces it. So the
        concurrency is bounded, and this asserts the bound rather than the symptom.
        """
        import asyncio

        in_flight = 0
        peak = 0

        class Counting(FakeImageProvider):
            async def edit(self, prompt: str, references: list[bytes]) -> object:  # type: ignore[override]
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                try:
                    await asyncio.sleep(0)
                    return await super().edit(prompt, references)
                finally:
                    in_flight -= 1

        fake = Counting()
        monkeypatch.setattr(
            illustrate_module, "get_image_model", lambda _m: fake.as_model()
        )
        monkeypatch.setattr(
            illustrate_module, "get_chat_model", lambda _m: FakeModel(_plan())
        )

        await run_illustration_pipeline(brief, story, tmp_path)

        assert peak <= illustrate_module._MAX_CONCURRENT_IMAGES
        assert len(fake.edit_calls) == len(story.pages), (
            "every page must still be drawn"
        )

    async def test_images_are_written_to_the_given_directory(
        self,
        brief: StoryBrief,
        story: Story,
        provider: FakeImageProvider,
        directed: None,
        tmp_path: Path,
    ) -> None:
        art = await run_illustration_pipeline(brief, story, tmp_path)

        for item in art.portraits + art.pages:
            assert item.path is not None
            assert item.path.exists()
            assert item.path.parent == tmp_path


class TestValidatingThePlan:
    """The one illustration failure that raises instead of degrading.

    Replaced a `MAX_IMAGES_PER_BOOK` setting that could never fire: the image count
    is derived from the page count, which `StoryBrief` already caps at 24. What was
    worth keeping was the structural check.
    """

    def test_a_plan_missing_a_page_raises(self, story: Story) -> None:
        """The case a soft failure cannot express.

        A page the Director never planned and a page whose image failed both render
        as a blank frame -- but only the second appears in `StoryArt`. Without this
        check, a six-page book planned with three pictures looks like three provider
        failures.
        """
        plan = _plan(pages=len(story.pages) - 1)

        with pytest.raises(StoryStructureError, match="but the book has"):
            validate_illustration_plan(story, plan)

    def test_a_plan_covering_every_page_passes(self, story: Story) -> None:
        validate_illustration_plan(story, _plan(pages=len(story.pages)))

    def test_a_duplicated_page_raises(self, story: Story) -> None:
        plan = _plan(pages=len(story.pages))
        plan.pages[1].page_number = 1

        with pytest.raises(StoryStructureError):
            validate_illustration_plan(story, plan)

    def test_a_page_naming_an_undescribed_character_raises(self, story: Story) -> None:
        """Finding U's failure mode as a structural error.

        A character with no appearance silently stops being described in its page
        prompt. When the *Director* caused that -- rather than a failed portrait --
        it disagreed with itself, which is a bug, not a degraded provider.
        """
        plan = _plan(pages=len(story.pages))
        plan.pages[0].characters_present = ["Kit", "Ghost"]

        with pytest.raises(StoryStructureError, match="without describing"):
            validate_illustration_plan(story, plan)

    async def test_it_runs_before_any_image_is_paid_for(
        self,
        brief: StoryBrief,
        story: Story,
        provider: FakeImageProvider,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A cheap chat call is already spent; images must not be."""
        monkeypatch.setattr(
            illustrate_module,
            "get_chat_model",
            lambda _m: FakeModel(_plan(pages=len(story.pages) - 1)),
        )

        with pytest.raises(StoryStructureError):
            await run_illustration_pipeline(brief, story, tmp_path)

        assert not provider.generate_prompts, "no portrait should have been drawn"
        assert not provider.edit_calls, "no page should have been drawn"


class TestFailingSoft:
    async def test_a_failed_page_leaves_the_others_intact(
        self,
        brief: StoryBrief,
        story: Story,
        monkeypatch: pytest.MonkeyPatch,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """One 503 must not destroy a book that passed both critics."""
        fake = FakeImageProvider(fail_on=("page 2",))
        monkeypatch.setattr(
            illustrate_module, "get_image_model", lambda _m: fake.as_model()
        )
        art = await run_illustration_pipeline(brief, story, tmp_path)

        by_key = {item.key: item for item in art.pages}
        assert by_key["2"].status is ArtStatus.FAILED
        assert by_key["1"].status is ArtStatus.CONDITIONED
        assert by_key["2"].path is None

    async def test_a_failed_page_is_reported_not_hidden(
        self,
        brief: StoryBrief,
        story: Story,
        monkeypatch: pytest.MonkeyPatch,
        directed: None,
        tmp_path: Path,
    ) -> None:
        fake = FakeImageProvider(fail_on=("page 2",))
        monkeypatch.setattr(
            illustrate_module, "get_image_model", lambda _m: fake.as_model()
        )
        art = await run_illustration_pipeline(brief, story, tmp_path)

        assert not art.fully_conditioned, "a failed page must not read as success"

    async def test_a_failed_portrait_leaves_its_pages_unconditioned(
        self,
        brief: StoryBrief,
        story: Story,
        monkeypatch: pytest.MonkeyPatch,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """Finding N's exact failure mode: the pictures still appear, but the
        consistency mechanism did not run, and the artifact has to say so."""
        fake = FakeImageProvider(fail_on=("standing, facing forward",))
        monkeypatch.setattr(
            illustrate_module, "get_image_model", lambda _m: fake.as_model()
        )
        art = await run_illustration_pipeline(brief, story, tmp_path)

        assert art.portraits[0].status is ArtStatus.FAILED
        assert all(item.status is ArtStatus.UNCONDITIONED for item in art.pages)
        assert not art.fully_conditioned
        # Every page was still drawn -- degraded, not lost.
        assert all(item.path is not None for item in art.pages)

    async def test_the_reason_is_recorded(
        self,
        brief: StoryBrief,
        story: Story,
        monkeypatch: pytest.MonkeyPatch,
        directed: None,
        tmp_path: Path,
    ) -> None:
        """A run must answer *why* it degraded by reading a file."""
        fake = FakeImageProvider(fail_on=("standing, facing forward",))
        monkeypatch.setattr(
            illustrate_module, "get_image_model", lambda _m: fake.as_model()
        )
        art = await run_illustration_pipeline(brief, story, tmp_path)

        assert art.portraits[0].detail
        assert art.pages[0].detail == "no reference portrait available"


class TestRenderingArt:
    def test_a_page_with_no_art_renders_as_the_text_only_book(
        self, story: Story, tmp_path: Path
    ) -> None:
        """`None` and an empty StoryArt must behave identically, or a fully failed
        run degrades to a third code path nobody tests."""
        plain = tmp_path / "plain.pdf"
        empty = tmp_path / "empty.pdf"

        render_pdf(story, plain)
        render_pdf(story, empty, StoryArt(style_bible="x" * 60))

        assert plain.read_bytes()[:5] == b"%PDF-"
        assert len(empty.read_bytes()) == pytest.approx(
            len(plain.read_bytes()), rel=0.02
        )

    def test_a_corrupt_image_does_not_destroy_the_book(
        self, story: Story, tmp_path: Path
    ) -> None:
        """A bad download costs a picture, never the whole book."""
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"this is not an image")
        art = StoryArt(
            style_bible="x" * 60,
            pages=[
                ArtItem(key="1", status=ArtStatus.CONDITIONED, path=broken),
            ],
        )
        out = tmp_path / "book.pdf"

        render_pdf(story, out, art)

        assert out.read_bytes()[:5] == b"%PDF-"

    def test_a_truncated_image_does_not_destroy_the_book(
        self, story: Story, tmp_path: Path
    ) -> None:
        """The case that found a real bug: PIL decodes lazily.

        A truncated PNG has a *valid header*, so `getSize()` succeeds and the
        failure lands inside `drawImage`. Guarding only the read passed the
        garbage-file test above and would still have crashed a real book on a
        partial download.
        """
        from sparkstory.models.fake_image_model import _PNG_4X3

        truncated = tmp_path / "truncated.png"
        truncated.write_bytes(_PNG_4X3[:40])
        art = StoryArt(
            style_bible="x" * 60,
            pages=[ArtItem(key="1", status=ArtStatus.CONDITIONED, path=truncated)],
        )
        out = tmp_path / "book.pdf"

        render_pdf(story, out, art)

        assert out.read_bytes()[:5] == b"%PDF-"

    def test_a_missing_file_does_not_destroy_the_book(
        self, story: Story, tmp_path: Path
    ) -> None:
        """A recorded path whose file was deleted or never written."""
        art = StoryArt(
            style_bible="x" * 60,
            pages=[
                ArtItem(
                    key="1",
                    status=ArtStatus.CONDITIONED,
                    path=tmp_path / "does-not-exist.png",
                )
            ],
        )
        out = tmp_path / "book.pdf"

        render_pdf(story, out, art)

        assert out.read_bytes()[:5] == b"%PDF-"

    def test_art_makes_the_pdf_bigger(self, story: Story, tmp_path: Path) -> None:
        """The weakest possible check that an image was actually embedded, and the
        strongest available without parsing PDF internals."""
        from sparkstory.models.fake_image_model import _PNG_4X3

        image = tmp_path / "page-01.png"
        image.write_bytes(_PNG_4X3)
        art = StoryArt(
            style_bible="x" * 60,
            pages=[ArtItem(key="1", status=ArtStatus.CONDITIONED, path=image)],
        )
        plain = tmp_path / "plain.pdf"
        illustrated = tmp_path / "illustrated.pdf"

        render_pdf(story, plain)
        render_pdf(story, illustrated, art)

        assert len(illustrated.read_bytes()) > len(plain.read_bytes())
