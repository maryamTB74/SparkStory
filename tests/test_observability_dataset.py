"""Dataset upload, and the caveat that must travel with the data."""

import pytest

from sparkstory.observability import dataset as dataset_module
from sparkstory.observability.dataset import (
    JUDGED_RESOLUTION_LIMIT,
    experiment_config,
    upload_fixture_briefs,
)


class TestUploadRefusesLoudly:
    def test_refuses_when_tracing_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flag that appears to work and uploads nothing is the failure this
        project has recorded twice (finding M, rule 24). Raise instead.

        The message names the setting, so the fix does not need a code read.
        """
        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", False)
        with pytest.raises(RuntimeError, match="OPIK_ENABLED"):
            upload_fixture_briefs()

    def test_refuses_when_opik_cannot_be_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled but unconfigurable is still a failed upload, not a silent one."""
        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", True)
        monkeypatch.setattr(dataset_module, "configure", lambda: False)
        with pytest.raises(RuntimeError, match="could not be configured"):
            upload_fixture_briefs()


class TestExperimentConfig:
    def test_records_the_measured_resolution_limit(self) -> None:
        """Opik will plot a 0.1 movement as a trend. The measured noise floor
        has to sit next to the chart, not only in a README."""
        config = experiment_config()
        assert config["judged_resolution_limit"] == JUDGED_RESOLUTION_LIMIT
        assert config["judged_resolution_limit"] == 0.25
        assert "0.25" in config["judge_variance_note"]

    def test_carries_the_models_that_produced_the_scores(self) -> None:
        """An experiment whose model choices are not recorded cannot be
        compared against another one."""
        config = experiment_config()
        for key in (
            "planner_model",
            "writer_model",
            "outline_critic_model",
            "prose_critic_model",
            "judge_model",
        ):
            assert key in config

    def test_no_credential_reaches_the_experiment_config(self) -> None:
        """This is uploaded to a third party."""
        rendered = repr(experiment_config()).lower()
        assert "api_key" not in rendered
        assert "secret" not in rendered


class TestUpload:
    @pytest.fixture
    def fake_opik(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """A stand-in Opik client that records what it was asked to do."""
        monkeypatch.setattr("sparkstory.config.settings.opik_enabled", True)
        monkeypatch.setattr(dataset_module, "configure", lambda: True)
        recorded: dict[str, object] = {"cleared": False}

        class FakeDataset:
            def clear(self) -> None:
                recorded["cleared"] = True

            def insert(self, items: list[dict[str, object]]) -> None:
                recorded["items"] = items

        class FakeClient:
            def get_or_create_dataset(self, name: str, description: str) -> FakeDataset:
                recorded["name"] = name
                recorded["description"] = description
                return FakeDataset()

        import opik

        monkeypatch.setattr(opik, "Opik", FakeClient)
        return recorded

    def test_uploads_one_item_per_fixture_brief(
        self, fake_opik: dict[str, object]
    ) -> None:
        count = upload_fixture_briefs()
        items = fake_opik["items"]
        assert isinstance(items, list)
        assert count == len(items)
        assert count == 5

    def test_clears_before_inserting_so_a_re_run_replaces(
        self, fake_opik: dict[str, object]
    ) -> None:
        """Five briefs uploaded twice must stay five items, not become ten."""
        upload_fixture_briefs()
        assert fake_opik["cleared"] is True

    def test_each_item_carries_the_brief_itself(
        self, fake_opik: dict[str, object]
    ) -> None:
        """An item without its brief cannot be re-run by an experiment, which
        is the only reason to upload one."""
        upload_fixture_briefs()
        items = fake_opik["items"]
        assert isinstance(items, list)
        for item in items:
            assert "name" in item
            assert "brief" in item
            # Serialised, not a live model: opik has to JSON-encode this.
            assert isinstance(item["brief"], dict)
            assert "child" in item["brief"]
