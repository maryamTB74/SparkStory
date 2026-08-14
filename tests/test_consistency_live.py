"""The acceptance run: does the judge see what a person sees?

**This is a gate, not an extra.** Every other test of this feature uses a fake and
therefore proves plumbing, not perception. A `vision`-marked suite nobody runs is
finding M's failure mode -- the check is *unfalsified rather than proven* -- so the
judge may not be described as working until this has passed.

Marked `vision` and excluded from `make test` and `ci-local`, in the shape `corpus`
already takes: it needs a key and a network. Run it with `make test-vision`.

**Why these inputs.** They are images already committed in `outputs/`, from three
runs made before this feature existed, whose defects were found by a person reading
JPEGs and are recorded as finding HH. So the expectations here were written from the
pictures rather than from what the code does -- which is the only reason a passing
run means anything.

Two honest limits, both worth stating in the file rather than in a review nobody
re-reads:

* The expectations are **one person's labels**, written in the same session that
  designed the rubric. That is not lesson 30's alignment score, which needs
  independent labels from someone who did not write the prompt.
* A pass here is a pass on *four* known drifts. It says nothing about the rate of
  false positives across a whole book, which is why the Maryam case below is the
  most important test in the file.
"""

from pathlib import Path

import pytest

from sparkstory.config import settings
from sparkstory.entities.illustration import ConsistencyAttribute
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.consistency_judge import ConsistencyJudgeNode

# `asyncio_mode = "auto"` means async tests need no marker of their own, so this is
# only the opt-out marker.
pytestmark = pytest.mark.vision

_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
_ANT = _OUTPUTS / "20260805-225100-an-ant-looking-for-sunlight-through-the-"
_FOX = _OUTPUTS / "20260804-193506-a-fox-who-wants-to-visit-the-moon"

#: Verbatim from `direct_illustrations-1.json` in the ant run. The portrait
#: contradicts it -- it came back green -- which is the whole point of the first
#: test: a model that recites its input says "black" and agrees.
_ANT_APPEARANCE = (
    "Small black ant with three visible body segments, six thin legs, "
    "two antennae, and a shiny smooth surface."
)

#: Verbatim from the fox run. This portrait is *correct*, which is what makes the
#: page tests below meaningful: they measure drift from a good reference.
_MARYAM_APPEARANCE = (
    "Small five-year-old build, warm brown skin, short curly black hair in two "
    "pigtails, always wearing a bright yellow scarf around her neck."
)


def _requires(*paths: Path) -> None:
    """Skip rather than fail when a run directory has been cleaned away.

    `outputs/` is disposable by this project's own convention and has already lost
    eight books. A missing fixture is not a defect in the judge, and a red suite that
    means "the images are gone" would train everyone to ignore it.
    """
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"committed run images not present: {missing}")


async def _verdict(
    name: str,
    image: Path,
    *,
    reference: Path | None = None,
    appearance: str | None = None,
):
    """One real judging call."""
    node = ConsistencyJudgeNode(
        model=get_chat_model(settings.consistency_judge_model),
        name=name,
        image=image.read_bytes(),
        reference_image=reference.read_bytes() if reference else None,
        appearance=appearance,
    )
    return await node.ainvoke()


class TestCheckAFindsAWrongPortrait:
    """The cheap check, and the one the spike already validated once."""

    async def test_the_green_ant_is_caught(self) -> None:
        """The acceptance question of the whole feature, in one call.

        The Director wrote "small black ant" and the portrait came back green, and
        every page then copied the green. This is also the rule 13 trap made
        concrete: the contradicting description is handed to the model, so a judge
        that restates its input rather than looking reports a match and fails here.
        """
        portrait = _ANT / "portrait-Pip.jpg"
        _requires(portrait)

        verdict = await _verdict("Pip", portrait, appearance=_ANT_APPEARANCE)

        assert verdict.matches is False
        assert verdict.attribute is ConsistencyAttribute.COLOUR
        assert "green" in verdict.difference.lower(), (
            "the difference must name what the picture shows, not what it was told: "
            f"got {verdict.difference!r}"
        )

    async def test_a_correct_portrait_is_left_alone(self) -> None:
        """Rule 24: without this, a judge that rejected every portrait would pass
        the test above and destroy every book by dropping all its references."""
        portrait = _FOX / "portrait-Maryam.jpg"
        _requires(portrait)

        verdict = await _verdict("Maryam", portrait, appearance=_MARYAM_APPEARANCE)

        assert verdict.matches is True, (
            f"a good portrait was rejected as {verdict.attribute}: "
            f"{verdict.difference!r}"
        )


class TestCheckBFindsPageDrift:
    """The unproven half. These are the tests most likely to fail, and a failure
    here is a finding about the judge rather than a bug in the pipeline."""

    async def test_the_fox_paws_changed_colour(self) -> None:
        """The portrait has white paws and page 4 draws them black, while ears,
        tail, face and the star charm all carry across correctly. So this asks the
        judge to find a *local* colour drift on a character that is otherwise
        right -- much harder than the whole-body swap above."""
        portrait, page = _FOX / "portrait-Fox.jpg", _FOX / "page-04.jpg"
        _requires(portrait, page)

        verdict = await _verdict("Fox", page, reference=portrait)

        assert verdict.matches is False
        assert verdict.attribute in {
            ConsistencyAttribute.COLOUR,
            ConsistencyAttribute.MARKINGS,
        }, f"expected a colour or marking difference, got {verdict.attribute}"

    async def test_a_character_at_tiny_scale_is_not_flagged(self) -> None:
        """Page 6's fox is inside a rocket, occupying roughly 4% of the frame.

        **This test asserts the opposite of what it was written to assert, and the
        reversal is the finding.** Finding HH recorded page 6 as a `PROP` drift
        (silver charm to gold) from a smaller view of the image. Looking at the full
        JPEG after the judge disagreed: the charm is a gold star in the portrait
        *too*, and the only candidate difference is the collar band, which is a few
        pixels at this scale. The label was wrong, not the verdict.

        So what this now pins is the sensible behaviour it actually found: a
        correctly-drawn character at tiny scale is reported as matching rather than
        as a difference invented to seem thorough. That is the right call, and it is
        also the limit of check B -- a real drift this small would not be caught
        either, which is a property of the check to record rather than tune away.
        """
        portrait, page = _FOX / "portrait-Fox.jpg", _FOX / "page-06.jpg"
        _requires(portrait, page)

        verdict = await _verdict("Fox", page, reference=portrait)

        assert verdict.matches is True, (
            "a 4%-of-frame character was flagged as "
            f"{verdict.attribute}: {verdict.difference!r}"
        )


class TestTheJudgeDoesNotFlagEverything:
    """**The most important test in this file, and the one a demo would omit.**

    Maryam is consistent across the whole fox run -- same pigtails, same yellow
    scarf, same tunic, same face -- while her pose, framing, scale and background
    all change between pages. A judge that cannot tell "different moment" from
    "different character" will flag her, and it would then drive a redraw loop to
    ruin pages that were fine.

    If these fail, the finding is that the judge over-fires, and the conclusion is
    that the redraw loop must not be built yet.
    """

    @pytest.mark.parametrize("page_name", ["page-04.jpg", "page-06.jpg"])
    async def test_a_consistent_character_is_not_flagged(self, page_name: str) -> None:
        portrait, page = _FOX / "portrait-Maryam.jpg", _FOX / page_name
        _requires(portrait, page)

        verdict = await _verdict("Maryam", page, reference=portrait)

        assert verdict.matches is True, (
            f"{page_name}: a consistent character was flagged as "
            f"{verdict.attribute}: {verdict.difference!r}"
        )


class TestTheJudgeIsRepeatable:
    """Rule 29, measured rather than assumed.

    `temperature 0.0` did not make the prose judge repeatable -- the same five books
    judged twice moved by up to 0.25 on `delight`, which is two pages of an
    eight-page book. So the same question is asked twice here. A judge that answers
    differently on identical input turns every verdict above into noise, and this is
    two calls to find out.
    """

    async def test_the_same_pair_gets_the_same_verdict(self) -> None:
        portrait, page = _FOX / "portrait-Fox.jpg", _FOX / "page-04.jpg"
        _requires(portrait, page)

        first = await _verdict("Fox", page, reference=portrait)
        second = await _verdict("Fox", page, reference=portrait)

        assert first.matches == second.matches, (
            "the same picture judged twice disagreed with itself: "
            f"{first.difference!r} then {second.difference!r}"
        )
