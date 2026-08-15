"""configure() must never raise. A book is worth more than its trace.

Every test here patches the module-level ``settings`` singleton rather than
constructing a ``Settings``, because that singleton is what ``configure()``
reads. Constructing an instance would leave the function looking at the real
environment and make these tests depend on whether a .env happens to exist.
"""

import logging

import pytest
from pydantic import SecretStr

from sparkstory.observability.opik_utils import configure


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn tracing on with a usable key, so a test can fail for its own reason."""
    monkeypatch.setattr("sparkstory.config.settings.opik_enabled", True)
    monkeypatch.setattr("sparkstory.config.settings.opik_api_key", SecretStr("k"))


class TestConfigureFailsOpen:
    def test_returns_false_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default path, and it must not touch opik at all."""
        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", False)
        assert configure() is False

    def test_returns_false_and_warns_when_the_key_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Enabled but keyless is a misconfiguration, not a crash.

        The warning names the variable, because "tracing is off" without saying
        which setting to fix is the kind of log line nobody acts on.
        """
        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", True)
        monkeypatch.setattr("sparkstory.config.settings.opik_api_key", None)
        with caplog.at_level(logging.WARNING):
            assert configure() is False
        assert "OPIK_API_KEY" in caplog.text

    def test_a_blank_key_is_treated_as_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The blank-key path end to end: the validator normalises
        `OPIK_API_KEY=` to None,
        and this is the half that proves configure() then does the right thing
        with it rather than authenticating with an empty string."""
        from sparkstory.config import Settings

        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", True)
        monkeypatch.setattr(
            "sparkstory.config.settings.opik_api_key",
            Settings(opik_api_key="   ").opik_api_key,
        )
        with caplog.at_level(logging.WARNING):
            assert configure() is False

    def test_swallows_a_raising_backend(
        self,
        enabled: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The case that matters.

        An auth failure or an unreachable workspace must cost a trace, never a
        book. The error text is logged so the cause is recoverable.
        """
        import opik

        def boom(**_kwargs: object) -> None:
            raise RuntimeError("workspace unreachable")

        monkeypatch.setattr(opik, "configure", boom)
        with caplog.at_level(logging.WARNING):
            assert configure() is False
        assert "workspace unreachable" in caplog.text

    def test_returns_true_when_opik_accepts_the_configuration(
        self, enabled: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The success path, asserted so the three failure paths above are not
        vacuously passing -- a configure() that always returned False would
        satisfy every other test in this class."""
        import opik

        captured: dict[str, object] = {}

        def record(**kwargs: object) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(opik, "configure", record)
        assert configure() is True
        assert captured["api_key"] == "k"
        assert captured["use_local"] is False
