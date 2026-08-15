"""What one turn of a client conversation produced.

The distinction under test is `tool_calls` against `executed`: what the model
asked for, against what actually ran. Every other test here exists to keep that
distinction honest.
"""

import dataclasses

import pytest

from sparkstory.mcp.client.types import ExecutedCall, ToolCall, TurnResult


def test_a_turn_can_request_a_call_without_executing_it() -> None:
    # The reason these are two fields rather than one. `try_prompt.py`'s whole
    # method is inspecting what a model *asked* for -- `write_story` is
    # deliberately never called there, because its arguments answer the question
    # and running it buys a book to learn nothing more. The REPL needs the
    # opposite view. A single list would force one front end to reconstruct the
    # other's, and inspect mode would have no way to say "it wanted this and we
    # did not do it".
    result = TurnResult(
        text="",
        tool_calls=[ToolCall(id="call-1", name="write_story", args={})],
        executed=[],
    )

    assert [c.name for c in result.tool_calls] == ["write_story"]
    assert result.executed == []


def test_a_turn_with_no_tool_calls_is_a_plain_answer() -> None:
    # This is the loop's stop condition, so it has to be representable: a loop
    # that ends on an empty list cannot end if an empty list is invalid.
    result = TurnResult(text="Here is the plan.", tool_calls=[], executed=[])

    assert result.text == "Here is the plan."
    assert not result.tool_calls


def test_an_executed_call_records_its_result() -> None:
    result = ExecutedCall(
        id="call-1", name="plan_story", args={"brief": {}}, result={"title": "Kit"}
    )

    assert result.result == {"title": "Kit"}
    assert result.error is None


def test_a_failed_call_still_counts_as_executed() -> None:
    # It ran, it cost time, and the model has to see the error to recover from
    # it. Dropping a failure would make a failed turn indistinguishable from a
    # skipped one -- which is exactly the confusion inspect mode introduces, so
    # the two must not blur here as well.
    failed = ExecutedCall(
        id="call-1", name="plan_story", args={}, result=None, error="boom"
    )

    assert failed.error == "boom"
    assert failed.result is None


def test_a_call_defaults_to_having_neither_result_nor_error() -> None:
    # Both optional, because a call is built before it is run. Requiring either
    # at construction would mean building the record twice.
    call = ExecutedCall(id="call-1", name="plan_story", args={})

    assert call.result is None
    assert call.error is None


@pytest.mark.parametrize(
    "record",
    [
        ToolCall(id="call-1", name="plan_story", args={}),
        ExecutedCall(id="call-1", name="plan_story", args={}),
        TurnResult(text="", tool_calls=[], executed=[]),
    ],
)
def test_turn_records_are_frozen(record: object) -> None:
    # These describe what already happened. A front end that could mutate one
    # after the fact would be rewriting history the model has already been shown,
    # and the transcript would stop being evidence of anything.
    assert dataclasses.is_dataclass(record)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.name = "something-else"  # type: ignore[attr-defined]
