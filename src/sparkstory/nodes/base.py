"""The shape every agent takes.

A node is a class constructed with everything it needs
-- its model plus the inputs for this one invocation -- and
invoked with no arguments. Instances are therefore single-use and cheap, built
per story rather than held as long-lived services.

Two consequences worth understanding before adding a node.

**The model is injected, never constructed here.** A node receives a runnable and
knows nothing about registries, API keys or providers; those stay behind
``models/get_model.py``. This is also the test seam: a test passes a
``FakeModel`` instead of patching module attributes by string path, which is a
seam that survives renaming.

**The node binds its own output schema.** ``output_schema`` is the node's
contract -- it is the difference between "a model" and "a Plot Planner" -- so it
is declared beside the prompt that has to satisfy it, and applied here once for
every subclass. The factory deliberately does not do this: each node's schema
differs, so there is nothing shared to centralise, and a node whose schema lived
elsewhere could be constructed against the wrong one.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from langchain_core.runnables import Runnable
from pydantic import BaseModel


class Node(ABC):
    """An agent: a model, its inputs, and the prompt that joins them."""

    #: Structured output this node's model is bound to produce. Subclasses must
    #: set it; the attribute is not optional and has no sensible default.
    output_schema: ClassVar[type[BaseModel]]

    def __init__(self, model: Runnable[Any, Any]) -> None:
        """Bind ``output_schema`` to the injected model.

        Args:
            model: An unbound chat model, normally from ``get_chat_model``. Tests
                pass a ``FakeModel``, which implements the same two methods this
                relies on -- ``with_structured_output`` and ``ainvoke``.
        """
        self.model = model.with_structured_output(self.output_schema)

    @abstractmethod
    async def ainvoke(self) -> BaseModel:
        """Run this node and return its validated output.

        Takes no arguments: everything a node needs arrived through its
        constructor. Subclasses narrow the return type to their own schema.
        """
