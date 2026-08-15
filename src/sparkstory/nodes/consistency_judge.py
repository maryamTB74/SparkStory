"""Consistency Judge: does this picture show the same character as its reference?

**Why this exists.** `workflows/illustrate.py` already draws a reference portrait
per character and conditions every page on it, and it already records whether it
did so. What it cannot record is whether the conditioning *worked*. Three live runs
reported every item as conditioned while a fox's white paws came back black and a
green ant came back black -- so the record said the mechanism ran and the pictures
said it had not. `StoryArt`'s own docstring says its purpose is answering "was this
book actually reference-conditioned?" by reading a file rather than by looking at
the pictures. This node is what makes that true.

**Why it judges one pair at a time.** A node that scored a whole book in one call
would have to hold every page and every portrait at once, and the verdict would be
a list the model could pad or truncate. One picture against one reference is a
question with a single answer, and it means a page with no reference is simply not
asked about rather than being scored against nothing.

**Two different comparisons, one node.** A portrait is checked against the *words*
it was drawn from; a page is checked against the *portrait* it was conditioned on.
Both ask "does this picture match the character it should show", so both bind the
same output. The reason the first exists at all is that a portrait can be wrong
before any page is drawn -- one run's Director wrote "small black ant" and the
portrait came back green, and every page then copied the green faithfully. A judge
that only compared pages to portraits would have called that book consistent.

**The one thing this node must not do is agree.** Asked "does this match?", the
cheapest defensible answer is always yes, and a judge that agrees costs a call and
buys nothing -- an instruction gets satisfied the laziest legal way, so ask what
the laziest satisfying answer is before writing the rubric. So the output requires
the difference to be *named*, and the prompt says explicitly not to restate the
description as though it were what the picture shows. That instruction is load-
bearing: the spike fed both models a description ("small black ant") that
contradicted the image (a green ant), and both reported green. Weakening it would
remove the only evidence that this node reads pixels at all.

**Colour first.** Across three runs, faces, body plans, ears, tails and props all
carried across untouched and only colour moved. So colour is named first in the
prompt and first in `ConsistencyAttribute`, which is the opposite of how a
consistency check would be written without having looked at the pictures.
"""

import base64
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.illustration import ConsistencyVerdict
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


CONSISTENCY_JUDGE_SYSTEM_PROMPT = """\
You check whether the pictures in a children's book show the same character all \
the way through.

You will be shown one picture and told what that character is supposed to look \
like. Decide whether the picture shows that character.

Check these, in this order:

1. Colour -- the colour of the body, fur, skin, clothing. This is the difference \
that occurs most often, so look at it first and look at it carefully.
2. Markings -- a white chest, a dark tail tip, a pattern on clothing.
3. Anything worn or carried -- a collar, a charm, a scarf, a basket, and its colour.
4. Body shape -- proportions, and the number of limbs or segments on an animal.

How to answer well:

Say what you actually see. If the picture shows a green animal and the description \
says black, the answer is that the picture is green -- do not repeat the \
description back as though it were what you are looking at. Describing what you \
were told instead of what is in front of you is the one mistake that makes this \
check worthless.

Name the difference concretely, and name both sides of it: "the collar is gold in \
the picture and silver in the reference" is useful, "the collar is different" is \
not.

Small variations are not differences. The same character may be standing, sitting, \
turned away, in shadow, close up or far away, and drawn at a different size. Pose, \
expression, lighting and framing are all free to change. Judge identity and \
colouring, not the moment the picture captures.

If the picture shows the character correctly, say so plainly and leave the \
difference empty. Reporting a difference that is not there costs a picture that was \
fine, so do not invent one to seem thorough.\
"""

#: What a page is compared against: the character's reference portrait.
_PAGE_INSTRUCTION = """\
The first picture is the reference portrait of {name}. The second is a page of the \
book that should show the same character.

Does the character in the page picture match the reference portrait?\
"""

#: What a portrait is compared against: the words it was drawn from.
_PORTRAIT_INSTRUCTION = """\
This picture is meant to be a reference portrait of {name}, drawn from this \
description:

{appearance}

Does the picture show that character as described?\
"""


def _image_part(data: bytes) -> dict[str, Any]:
    """One image as a content part, ready to send.

    A ``data:`` URI rather than a provider-specific bytes wrapper, because that is
    what goes through ``get_chat_model`` unchanged -- verified by a spike against
    grok-4 and grok-3-mini before this node was written. It is the reason this
    feature needed no new model seam.

    The media type is declared ``image/jpeg`` because that is what the illustrator
    writes; it is not sniffed from the bytes. If a second image format ever reaches
    here, this is the line that has to learn about it, and a wrong media type is
    the kind of thing that fails as an opaque provider error.
    """
    encoded = base64.b64encode(data).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


class ConsistencyJudgeNode(Node):
    """Compares one picture against the reference it should match.

    Constructed with the pictures themselves rather than paths: the workflow
    already holds the bytes -- portraits are returned alongside their records
    precisely so pages need not re-read them -- and a node that opened files would
    add a second failure mode for no gain.
    """

    output_schema = ConsistencyVerdict

    def __init__(
        self,
        model: Runnable[Any, Any],
        *,
        name: str,
        image: bytes,
        reference_image: bytes | None = None,
        appearance: str | None = None,
    ) -> None:
        """Set up one comparison.

        Args:
            model: An unbound chat model that can accept an image. Tests pass a
                ``FakeModel``.
            name: The character being checked, so the prompt can name them.
            image: The picture under test -- a page, or a portrait.
            reference_image: The portrait ``image`` should match. Supply this to
                check a page.
            appearance: The written description ``image`` should match. Supply this
                to check a portrait.

        Raises:
            ValueError: neither or both references were supplied. A comparison with
                nothing to compare against would produce a confident verdict about
                nothing, and one with two references would silently favour whichever
                the prompt happened to mention -- both are programming errors here,
                not model failures, so they raise rather than degrade.
        """
        super().__init__(model)
        if (reference_image is None) == (appearance is None):
            raise ValueError(
                "judge a page against a reference_image or a portrait against an "
                "appearance, not neither and not both"
            )
        self.name = name
        self.image = image
        self.reference_image = reference_image
        self.appearance = appearance

    async def ainvoke(self) -> ConsistencyVerdict:
        """Look at the picture and return a verdict."""
        content: list[dict[str, Any]] = []
        if self.reference_image is not None:
            # Reference first, then the page. The order is stated in the prompt as
            # "the first picture ... the second is", so these two must not drift
            # apart: a swap would have the model comparing them backwards and the
            # verdict would still look plausible.
            content.append(_image_part(self.reference_image))
            content.append(_image_part(self.image))
            instruction = _PAGE_INSTRUCTION.format(name=self.name)
        else:
            content.append(_image_part(self.image))
            instruction = _PORTRAIT_INSTRUCTION.format(
                name=self.name, appearance=self.appearance
            )
        content.append({"type": "text", "text": instruction})

        logger.debug(
            "judging %r against %s",
            self.name,
            "its portrait" if self.reference_image is not None else "its description",
        )
        verdict: ConsistencyVerdict = await self.model.ainvoke(
            [
                SystemMessage(CONSISTENCY_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
        return verdict
