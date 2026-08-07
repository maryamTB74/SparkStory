"""One book should be one Opik thread.

`--stage all` runs two entrypoints, and each used to mint its own `request_id`.
That is correct for the MCP path -- `plan_story` and `write_story` are separate
tool calls, possibly minutes apart, and pretending otherwise would invent a
continuity the architecture does not have. It is wrong for a single operator
running one script, where the two traces could not be joined at all without
knowing both uuids.

So the id is now an optional argument: supplied by a caller who knows the two
stages are one book, minted per-pipeline otherwise.
"""

import ast
import inspect
from pathlib import Path

from sparkstory.workflows.plan_outline import run_outline_pipeline
from sparkstory.workflows.write_story import run_story_pipeline


class TestPipelinesAcceptASuppliedRequestId:
    def test_outline_pipeline_takes_a_request_id(self) -> None:
        """Without this parameter a caller cannot join the two traces."""
        parameters = inspect.signature(run_outline_pipeline).parameters
        assert "request_id" in parameters

    def test_story_pipeline_takes_a_request_id(self) -> None:
        parameters = inspect.signature(run_story_pipeline).parameters
        assert "request_id" in parameters

    def test_outline_request_id_defaults_to_none(self) -> None:
        """The MCP path passes nothing and must keep minting its own.

        A required argument here would break both tool modules, and a shared
        default would make two unrelated client calls look like one book.
        """
        parameters = inspect.signature(run_outline_pipeline).parameters
        assert parameters["request_id"].default is None

    def test_story_request_id_defaults_to_none(self) -> None:
        parameters = inspect.signature(run_story_pipeline).parameters
        assert parameters["request_id"].default is None

    def test_request_id_is_keyword_only_in_both(self) -> None:
        """Positional would make `run_story_pipeline(brief, outline, callback)`
        silently rebind: the third positional is already `on_task_result`."""
        for pipeline in (run_outline_pipeline, run_story_pipeline):
            parameter = inspect.signature(pipeline).parameters["request_id"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, pipeline


class TestTheScriptSuppliesOneIdForTheWholeRun:
    def test_write_one_story_mints_one_id_and_passes_it_to_both(self) -> None:
        """The script knows both stages are one book; the pipelines cannot.

        Asserted on the source rather than by running it, because the script is
        deliberately untested and a real run costs model calls. This catches the
        regression that matters: someone removing the argument from one of the
        two call sites, which would silently restore two threads.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "write_one_story.py"
        source = script.read_text(encoding="utf-8")
        assert source.count("request_id=request_id") == 2

    def test_the_id_is_in_scope_where_it_is_used(self) -> None:
        """Both call sites must resolve the name.

        The count assertion above passes even when the id is minted in one
        function and used in another -- which is exactly what happened, and ruff
        caught it rather than this test. Compiling the module proves the name
        binds; `compile` does not execute it, so this stays free and offline.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "write_one_story.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        run_stages = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_stages"
        )
        assigned = {
            target.id
            for node in ast.walk(run_stages)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "request_id" in assigned
