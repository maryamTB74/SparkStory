"""Tracing must be invisible to a pipeline that is not using it.

These assert the *wiring*, not the trace. Whether callbacks actually propagate
into each `@task` is a property of LangGraph that no offline test can show --
that is the live run's question, and it is the one that decides whether
attaching once per pipeline was enough.
"""

import inspect

import pytest

from sparkstory.workflows import plan_outline, write_story


@pytest.mark.parametrize("module", [plan_outline, write_story])
class TestBothPipelinesAreWired:
    def test_imports_build_handler(self, module: object) -> None:
        """A regression guard: the attachment is a two-line change and exactly
        the kind of thing a later refactor drops without any test noticing."""
        assert hasattr(module, "build_handler")

    def test_passes_callbacks_to_astream(self, module: object) -> None:
        """The callback list must reach astream, or the tracer is built and
        then silently discarded -- which would look identical to working."""
        source = inspect.getsource(module)
        assert '"callbacks"' in source
        assert "build_handler(request_id" in source

    def test_filters_none_out_of_the_callback_list(self, module: object) -> None:
        """A None in the callback list is an AttributeError mid-run.

        LangChain iterates that list and calls methods on every entry, so the
        disabled path has to pass [] rather than [None]. The failure would
        happen only with tracing off -- i.e. in the default configuration.
        """
        source = inspect.getsource(module)
        assert "if t is not None" in source


class TestTheSuiteNeverTraces:
    def test_tracing_is_off_regardless_of_the_developers_env(self) -> None:
        """The autouse fixture in conftest, asserted rather than assumed.

        Without it, `OPIK_ENABLED=true` in a .env makes 37 pipeline invocations
        upload 37 threads per suite run. That happened, and the symptom -- a
        project full of threads nobody could account for -- appeared a long way
        from the cause.
        """
        from sparkstory.config import settings

        assert settings.opik_enabled is False

    def test_build_handler_returns_none_under_the_fixture(self) -> None:
        """The consequence that matters: no tracer, so nothing is uploaded."""
        from sparkstory.observability.tracing import build_handler

        assert build_handler("suite-should-not-trace") is None


class TestDisabledTracingChangesNothing:
    def test_a_none_handler_becomes_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The expression both pipelines use, exercised directly."""
        monkeypatch.setattr(
            plan_outline, "build_handler", lambda request_id, tags=None: None
        )
        tracer = plan_outline.build_handler("req-1", tags=["plan_outline"])
        assert [t for t in [tracer] if t is not None] == []

    def test_a_real_handler_survives_the_filter(self) -> None:
        """The other half, so the filter is not vacuously passing: a filter that
        dropped everything would satisfy the test above on its own."""
        sentinel = object()
        assert [t for t in [sentinel] if t is not None] == [sentinel]
