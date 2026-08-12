"""What a finished book taught us about this child's world.

**A model call, and it runs after the book is delivered.** Pulling "a fox with a
white-tipped tail" out of prose is genuinely fuzzy, so a model is the right tool
-- and because it runs once the ``Story`` exists, a bad extraction costs a poor
memory rather than a poor book.

**Facts about characters, not about the plot.** The prompt asks for details a
*later* story would need to stay consistent. "Kit is a fox" is worth keeping;
"Kit was sad on page 4" is not -- it belongs to the story that just ended, and
storing it would have a later book treat a passing mood as a permanent trait.

**The prose is what the model reads, not the outline.** Extracting from the
outline would only re-read what the planner already wrote, and would miss the
details the Writer invented -- which are often the memorable ones.

**Rule 13 was asked of this prompt: what is the laziest thing that satisfies
it?** Restating each ``CharacterSketch`` description verbatim. That would look
like success and store nothing the outline did not already have. The wording
below asks for what is *permanently true* to push away from plot summary, and
Task 9 step 3 checks the answer against exactly this failure -- because no test
can.
"""

from typing import Any, ClassVar

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from sparkstory.entities.stories import Story
from sparkstory.memory.types import ChildId, MemoryKind, MemoryRecord
from sparkstory.nodes.base import Node


class ExtractedFact(BaseModel):
    """One durable detail about a character or place."""

    subject: str = Field(
        max_length=80,
        description="Who or what this is about, for example 'Kit'.",
    )
    text: str = Field(
        max_length=500,
        description=(
            "The detail itself, in one short sentence. Describe what is "
            "permanently true, not what happened in this story."
        ),
    )


class ExtractedMemories(BaseModel):
    """What to remember from one book."""

    facts: list[ExtractedFact] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Details a later story would need in order to keep this world "
            "consistent. Return an empty list if this story established nothing "
            "lasting."
        ),
    )
    episode: str = Field(
        default="",
        max_length=300,
        description=(
            "One sentence saying what happened, so a later story does not "
            "repeat it. Empty if there is nothing worth recording."
        ),
    )


class MemoryExtractor(Node):
    """Reads a finished book and says what is worth keeping."""

    output_schema: ClassVar[type[BaseModel]] = ExtractedMemories

    def __init__(
        self,
        model: Runnable[Any, Any],
        story: Story,
        child_id: ChildId,
        request_id: str,
    ) -> None:
        """Bind the schema and hold the book this run produced.

        Args:
            model: An unbound chat model. ``Node`` binds ``output_schema`` to it.
            story: The finished book. Its prose is what the model reads.
            child_id: Whose memory these records belong to.
            request_id: The run that produced the book, stamped onto every record
                so a memory can be traced back to the story it came from.
        """
        super().__init__(model)
        self.story = story
        self.child_id = child_id
        self.request_id = request_id

    async def ainvoke(self) -> ExtractedMemories:
        """Extract durable facts and one episode summary."""
        pages = "\n".join(page.text for page in self.story.pages)
        prompt = (
            "Here is a children's story that has just been finished.\n\n"
            f"Title: {self.story.outline.title}\n\n"
            f"{pages}\n\n"
            "Two things, so that the next story written for this child stays "
            "consistent with this one.\n\n"
            "First, list the details that are permanently true about the "
            "characters and their world -- what someone looks like, what they "
            "are, where they live. Do not list what happened, how anyone felt, "
            "or anything that was only true during this story.\n\n"
            "Second, write one sentence saying what happened in this story, so "
            "that a later story can avoid telling it again.\n\n"
            "If this story established nothing lasting, say so by returning "
            "nothing. An invented detail is worse than a missing one, because a "
            "later story would treat it as true."
        )
        return await self.model.ainvoke([HumanMessage(content=prompt)])

    def to_records(self, extracted: ExtractedMemories) -> list[MemoryRecord]:
        """Turn the model's answer into rows, stamped with child and run.

        Separate from ``ainvoke`` so the mapping is testable without a model, and
        so a caller can inspect what would be written before writing it.
        """
        records = [
            MemoryRecord(
                child_id=self.child_id,
                kind=MemoryKind.SEMANTIC,
                text=fact.text,
                subject=fact.subject,
                source_request_id=self.request_id,
            )
            for fact in extracted.facts
        ]
        # `strip()` rather than a bare truthiness check: a model answering with a
        # blank line would otherwise create an episode row containing whitespace,
        # which `render_memory` would then present as a story to avoid repeating.
        if extracted.episode.strip():
            records.append(
                MemoryRecord(
                    child_id=self.child_id,
                    kind=MemoryKind.EPISODIC,
                    text=extracted.episode,
                    source_request_id=self.request_id,
                )
            )
        return records
