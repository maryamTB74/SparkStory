"""The consistency judge, offline.

`FakeModel` records the messages it was sent, which is what makes the important
test here possible: asserting the image bytes were *actually attached*. A judge
that silently sent text alone would pass every other test in this file and would be
useless in a live run -- rule 33's trap, which shipped a corrupt `story.mp3` in
Session 14 while three layers of offline tests passed.
"""

import pytest

from sparkstory.entities.illustration import (
    ConsistencyAttribute,
    ConsistencyVerdict,
)
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.consistency_judge import (
    CONSISTENCY_JUDGE_SYSTEM_PROMPT,
    ConsistencyJudgeNode,
)

PAGE = b"\xff\xd8\xff\xe0page-bytes"
PORTRAIT = b"\xff\xd8\xff\xe0portrait-bytes"
APPEARANCE = "Small black ant with three visible body segments and six thin legs."


def _matches() -> ConsistencyVerdict:
    return ConsistencyVerdict(matches=True)


def _drifted() -> ConsistencyVerdict:
    return ConsistencyVerdict(
        matches=False,
        attribute=ConsistencyAttribute.COLOUR,
        difference="the ant is green in the picture and black in the reference",
    )


def _text_of(call: list) -> str:
    """Every text part of one recorded call, joined."""
    parts: list[str] = []
    for message in call:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content:
            if part.get("type") == "text":
                parts.append(part["text"])
    return "\n".join(parts)


def _image_urls(call: list) -> list[str]:
    """Every image URL of one recorded call, in the order they were sent."""
    urls: list[str] = []
    for message in call:
        if isinstance(message.content, str):
            continue
        for part in message.content:
            if part.get("type") == "image_url":
                urls.append(part["image_url"]["url"])
    return urls


class TestTheJudgeSendsThePicture:
    """The tests that would catch this degrading to a text-only check."""

    async def test_a_page_is_sent_with_both_images(self) -> None:
        model = FakeModel(_matches())
        node = ConsistencyJudgeNode(
            model=model, name="Pip", image=PAGE, reference_image=PORTRAIT
        )

        await node.ainvoke()

        urls = _image_urls(model.calls[0])
        assert len(urls) == 2, "a page comparison needs the portrait and the page"
        assert all(url.startswith("data:image/jpeg;base64,") for url in urls)

    async def test_the_reference_is_sent_first(self) -> None:
        """The prompt says "the first picture is the reference"; if the order here
        drifts from that sentence the model compares them backwards and the verdict
        still looks plausible, which is the worst kind of bug."""
        import base64

        model = FakeModel(_matches())
        node = ConsistencyJudgeNode(
            model=model, name="Pip", image=PAGE, reference_image=PORTRAIT
        )

        await node.ainvoke()

        urls = _image_urls(model.calls[0])
        expected = base64.b64encode(PORTRAIT).decode()
        assert expected in urls[0], "the reference portrait must be the first image"

    async def test_a_portrait_is_sent_with_one_image_and_its_description(self) -> None:
        model = FakeModel(_matches())
        node = ConsistencyJudgeNode(
            model=model, name="Pip", image=PORTRAIT, appearance=APPEARANCE
        )

        await node.ainvoke()

        assert len(_image_urls(model.calls[0])) == 1
        assert APPEARANCE in _text_of(model.calls[0])

    async def test_the_character_is_named_in_the_instruction(self) -> None:
        model = FakeModel(_matches())
        node = ConsistencyJudgeNode(
            model=model, name="Pip", image=PAGE, reference_image=PORTRAIT
        )

        await node.ainvoke()

        assert "Pip" in _text_of(model.calls[0])


class TestTheJudgeContract:
    def test_binds_the_verdict_schema(self) -> None:
        model = FakeModel(_matches())
        ConsistencyJudgeNode(
            model=model, name="Pip", image=PAGE, reference_image=PORTRAIT
        )

        assert model.bound_schema is ConsistencyVerdict

    async def test_returns_the_verdict_unchanged(self) -> None:
        model = FakeModel(_drifted())
        node = ConsistencyJudgeNode(
            model=model, name="Pip", image=PAGE, reference_image=PORTRAIT
        )

        verdict = await node.ainvoke()

        assert verdict.matches is False
        assert verdict.attribute is ConsistencyAttribute.COLOUR

    def test_a_comparison_needs_exactly_one_reference(self) -> None:
        """Neither is a comparison against nothing; both would silently favour
        whichever the prompt mentioned. Both are our bugs, so both raise."""
        with pytest.raises(ValueError, match="not neither and not both"):
            ConsistencyJudgeNode(model=FakeModel(_matches()), name="Pip", image=PAGE)

        with pytest.raises(ValueError, match="not neither and not both"):
            ConsistencyJudgeNode(
                model=FakeModel(_matches()),
                name="Pip",
                image=PAGE,
                reference_image=PORTRAIT,
                appearance=APPEARANCE,
            )


class TestThePrompt:
    """Prompt text is behaviour. These pin the instructions that a live run showed
    to be load-bearing, so a later edit that drops one fails here."""

    def test_colour_is_checked_first(self) -> None:
        """Finding HH: colour is the only attribute that ever drifted across three
        runs. The ordering is the finding, so it is asserted."""
        lowered = CONSISTENCY_JUDGE_SYSTEM_PROMPT.lower()
        assert lowered.index("colour") < lowered.index("markings")
        assert lowered.index("markings") < lowered.index("body shape")

    def test_forbids_restating_the_description(self) -> None:
        """The spike's whole result: both models were fed "black" against a green
        ant and reported green. This instruction is why that is expected rather than
        lucky, so removing it must fail a test."""
        lowered = CONSISTENCY_JUDGE_SYSTEM_PROMPT.lower()
        assert "do not repeat the description back" in lowered

    def test_exempts_pose_and_lighting(self) -> None:
        """Without this the judge flags every page: the same character legitimately
        changes pose, framing and light between pictures. This is the instruction
        aimed at the false-positive half, which the acceptance run measures."""
        lowered = CONSISTENCY_JUDGE_SYSTEM_PROMPT.lower()
        for free in ("pose", "expression", "lighting", "framing"):
            assert free in lowered

    def test_discourages_inventing_a_difference(self) -> None:
        """A judge that finds something on every page looks diligent and costs a
        redraw of a picture that was fine."""
        assert "do not invent one" in CONSISTENCY_JUDGE_SYSTEM_PROMPT.lower()
