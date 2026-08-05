"""An image model that returns a canned picture, for tests and offline runs.

The sibling of ``FakeModel`` and ``FakeEmbedder``, and the thing that keeps "no
network required" true now that the package can generate images. Same reasoning as
``FakeModel``: a fake passed to a constructor cannot rot the way a monkeypatch of a
module attribute by string path can, and it lets a test assert on *what was asked
for* rather than only on what came back.

**It records prompts and reference counts**, because the interesting assertions in
this feature are about those rather than about pixels: did the page ask for the
right characters, was it conditioned on portraits at all, and did a failure leave
the frame blank instead of killing the book.

``FakeImageModel`` is deliberately not a ``FakeModel`` subclass. An image model has
no ``with_structured_output`` and binds no schema, so inheriting would offer
methods that mean nothing here -- the same argument that keeps ``get_image_model``
a separate seam from ``get_chat_model``.
"""

import base64

from sparkstory.entities.exceptions import ImageGenerationError
from sparkstory.entities.illustration import MAX_REFERENCE_IMAGES
from sparkstory.models.get_image_model import GeneratedImage, ImageModel

#: A real, decodable 4x3 PNG. Real bytes rather than `b"fake"` on purpose --
#: reportlab has to *decode* these in the renderer tests, not merely open them.
#:
#: It is 4x3 rather than 1x1 for a reason worth keeping: the renderer scales art to
#: fit its frame without distorting it, and a square image cannot tell a correct
#: fit-inside from a stretch-to-fill. A non-square one can.
#:
#: The first attempt here was a hand-copied "1x1 transparent PNG" that PIL rejects
#: as a broken data stream. It passed every test in this module and failed only in
#: the renderer -- which is precisely the argument for a fake being *valid* rather
#: than merely plausible.
_PNG_4X3 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAFElEQVR4nGM8"
    "ESDCAANMDEgAhQMALKwBMrl5U4MAAAAASUVORK5CYII="
)


class FakeImageProvider:
    """Stands in for an image provider, recording how it was asked.

    Args:
        fail_on: Substrings of a prompt that should fail instead of returning an
            image. A prompt containing any of them raises
            ``ImageGenerationError``. Matching on the prompt rather than on a call
            index is what lets a test say "the *portrait* fails" or "page 3 fails"
            without depending on the order the workflow happens to run in --
            which matters here because pages are generated concurrently.
    """

    def __init__(self, *, fail_on: tuple[str, ...] = ()) -> None:
        self.fail_on = fail_on
        #: Prompts passed to ``generate``, in call order.
        self.generate_prompts: list[str] = []
        #: ``(prompt, number_of_references)`` per ``edit`` call.
        self.edit_calls: list[tuple[str, int]] = []

    def _check(self, prompt: str) -> None:
        for needle in self.fail_on:
            if needle in prompt:
                raise ImageGenerationError(
                    f"FakeImageProvider was told to fail on {needle!r}"
                )

    async def generate(self, prompt: str) -> GeneratedImage:
        """Record the prompt and return the canned PNG."""
        self.generate_prompts.append(prompt)
        self._check(prompt)
        return GeneratedImage(data=_PNG_4X3, image_format="png")

    async def edit(self, prompt: str, references: list[bytes]) -> GeneratedImage:
        """Record the prompt and reference count, then return the canned PNG.

        Enforces the same two guards the real seam does. A fake that accepted
        calls the provider would reject is worse than no fake: it makes a test
        prove something about the fake rather than about the code.
        """
        self.edit_calls.append((prompt, len(references)))
        if not references:
            raise ImageGenerationError(
                "edit() requires at least one reference image; use generate() "
                "for a prompt-only image."
            )
        if len(references) > MAX_REFERENCE_IMAGES:
            raise ImageGenerationError(
                f"edit() accepts at most {MAX_REFERENCE_IMAGES} reference "
                f"images, got {len(references)}."
            )
        self._check(prompt)
        return GeneratedImage(data=_PNG_4X3, image_format="png")

    def as_model(self) -> ImageModel:
        """Wrap self in the ``ImageModel`` the workflow expects."""
        return ImageModel(generate=self.generate, edit=self.edit)
