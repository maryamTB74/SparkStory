"""The tracer, and the disabled path that must import nothing.

``build_handler`` imports ``OpikTracer`` inside its body, so these tests patch
the attribute on the module the import reads from. Patching a name in this test
module would do nothing -- the import happens at call time, not at collection.
"""

import logging
import sys

import pytest

from sparkstory.observability import tracing
from sparkstory.observability.tracing import build_handler, trace_metadata


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend Opik configured cleanly, so a test fails for its own reason."""
    monkeypatch.setattr(tracing, "configure", lambda: True)


class TestDisabledPath:
    def test_returns_none_when_configure_declines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tracing, "configure", lambda: False)
        assert build_handler("req-1") is None

    def test_the_disabled_path_imports_no_opik(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The import rule, asserted rather than trusted.

        A module-scope import would make a heavy dependency tree mandatory for
        every run, including runs that never trace. This test fails the moment
        someone hoists the import to the top of the file.
        """
        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", False)
        for name in [n for n in sys.modules if n.startswith("opik")]:
            monkeypatch.delitem(sys.modules, name)

        assert build_handler("req-1") is None
        assert not any(name.startswith("opik") for name in sys.modules)


class TestTraceMetadata:
    def test_carries_what_varies_between_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two traces are only comparable if the models are recorded on them."""
        monkeypatch.setattr("sparkstory.config.settings.planner_model", "grok-3-mini")
        monkeypatch.setattr("sparkstory.config.settings.writer_model", "grok-4")

        metadata = trace_metadata()

        assert metadata["planner_model"] == "grok-3-mini"
        assert metadata["writer_model"] == "grok-4"

    @pytest.mark.parametrize(
        "key",
        [
            "planner_model",
            "plot_model",
            "writer_model",
            "researcher_model",
            "outline_critic_model",
            "prose_critic_model",
            "max_outline_revisions",
            "max_prose_revisions",
        ],
    )
    def test_every_stage_is_recorded(self, key: str) -> None:
        """A stage missing from the metadata is a run whose difference from
        another run is invisible in the trace -- which is the one thing the
        metadata exists for."""
        assert key in trace_metadata()

    def test_no_credential_reaches_the_metadata(self) -> None:
        """Trace metadata leaves the machine. Model names are fine; keys are
        not, and SecretStr masking does not apply once something is serialised
        into a metadata dict."""
        rendered = repr(trace_metadata()).lower()
        assert "api_key" not in rendered
        assert "secret" not in rendered


class TestBuildHandler:
    def test_is_keyed_by_request_id(
        self, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """request_id is the thread id, so a log line leads to its trace."""
        import opik.integrations.langchain as langchain_integration

        captured: dict[str, object] = {}

        class FakeTracer:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(langchain_integration, "OpikTracer", FakeTracer)

        handler = build_handler("req-42", tags=["plan_outline"])

        assert handler is not None
        assert captured["thread_id"] == "req-42"
        assert captured["tags"] == ["plan_outline"]
        assert captured["metadata"] == trace_metadata()

    def test_a_failing_constructor_yields_none(
        self,
        configured: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Opik breaking must not break the pipeline that asked for a trace."""
        import opik.integrations.langchain as langchain_integration

        def boom(**_kwargs: object) -> None:
            raise RuntimeError("no backend")

        monkeypatch.setattr(langchain_integration, "OpikTracer", boom)

        with caplog.at_level(logging.WARNING):
            assert build_handler("req-1") is None
        assert "no backend" in caplog.text
