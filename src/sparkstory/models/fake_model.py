"""A model that returns canned answers, for tests and offline runs.

Why this rather than monkeypatching ``get_chat_model``: a patch targets a module
attribute by string path, so renaming a module leaves the patch silently pointing
at nothing and the test passes for the wrong reason. Passing a fake to a node's
constructor cannot rot that way, and it lets a test assert on what the node
*sent* and which schema it *bound*.
"""

from typing import Any

from pydantic import BaseModel


class FakeModel:
    """Stands in for a chat model, recording how it was used.

    Implements only the two methods :class:`~sparkstory.nodes.base.Node` needs --
    ``with_structured_output`` and ``ainvoke`` -- rather than subclassing a
    LangChain base class. A narrow fake fails loudly when a node starts using
    something new, which is information; a broad one silently absorbs it.
    """

    def __init__(self, *responses: BaseModel) -> None:
        """Queue the objects this model will return, in order.

        Args:
            *responses: Returned one per call. The final response repeats if the
                model is invoked more often than there are responses, so a test
                that only cares about one call need supply only one.
        """
        if not responses:
            raise ValueError("FakeModel needs at least one response to return")
        self._responses = list(responses)
        #: Schema passed to ``with_structured_output``, or None if never bound.
        self.bound_schema: type[BaseModel] | None = None
        #: Message lists received, one entry per ``ainvoke`` call.
        self.calls: list[list[Any]] = []

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> FakeModel:
        """Record the schema and return self, as a real bound runnable would."""
        self.bound_schema = schema
        return self

    async def ainvoke(self, messages: list[Any], **_: Any) -> BaseModel:
        """Record the messages and return the next queued response."""
        self.calls.append(messages)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    @property
    def messages(self) -> list[Any]:
        """Messages from the most recent call, for the common single-call case."""
        if not self.calls:
            raise AssertionError("FakeModel was never invoked")
        return self.calls[-1]
