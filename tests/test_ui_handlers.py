"""Every UI route, called as a function.

**The research seam must be stubbed.** Research runs before planning, so a test
that fakes only `get_chat_model` reaches a real embedder and a real provider --
25 seconds and a live call for a test that looks like it passed. The research
seam is stubbed autouse below; any new test file touching these handlers must do
the same.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.entities.stories import (
    CharacterSketch,
    NarrativeFunction,
    PagePlan,
    ScenePlan,
    Story,
    StoryBeat,
    StoryOutline,
    StoryPage,
)
from sparkstory.mcp.ui import handlers
from sparkstory.mcp.ui.jobs import JobRegistry, JobState


@pytest.fixture(autouse=True)
def _no_research(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the research seam reaching a real provider."""

    async def _nothing(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "sparkstory.workflows.plan_outline.research_premise", _nothing, raising=False
    )


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch: pytest.MonkeyPatch) -> JobRegistry:
    """A per-test registry, so tests cannot see each other's jobs."""
    fresh = JobRegistry()
    monkeypatch.setattr(handlers, "registry", fresh)
    return fresh


def _outline() -> StoryOutline:
    return StoryOutline(
        title="Maryam and the Paper Rocket",
        logline="Maryam sends a paper rocket up to the moon for a fox.",
        theme="a wish sent into the sky",
        characters=[
            CharacterSketch(
                name="Maryam", role="protagonist", description="Builds things"
            ),
            CharacterSketch(name="Kit", role="friend", description="A fox"),
        ],
        beats=[
            StoryBeat(
                position=1,
                function=NarrativeFunction.SETUP,
                title="The fox arrives",
                summary="A fox appears at the bottom of the garden.",
            ),
            StoryBeat(
                position=2,
                function=NarrativeFunction.RISING_ACTION,
                title="Kit points up",
                summary="Kit will not stop looking at the moon.",
            ),
            StoryBeat(
                position=3,
                function=NarrativeFunction.CLIMAX,
                title="Folding the rocket",
                summary="Maryam folds a paper rocket in the last of the light.",
            ),
            StoryBeat(
                position=4,
                function=NarrativeFunction.RESOLUTION,
                title="The rocket flies",
                summary="The paper rocket climbs into the evening sky.",
            ),
        ],
    )


def _request(
    body: dict[str, Any] | None = None, path_params: dict[str, str] | None = None
) -> Any:
    """A minimal Starlette-shaped request stub.

    Hand-built rather than via TestClient: these handlers are plain async
    functions and a full ASGI round trip would test Starlette rather than us.
    """

    class _Request:
        def __init__(self) -> None:
            self.path_params = path_params or {}

        async def json(self) -> dict[str, Any]:
            if body is None:
                raise ValueError("no body")
            return body

    return _Request()


_VALID_FORM = {
    "name": "Maryam",
    "age": "5",
    "pronouns": "she/her",
    "reading_level": "early_reader",
    "interests": "foxes, the moon",
    "premise": "a fox who wants to visit the moon",
    "tone": "gentle",
    "world_rules": "imaginative",
    "page_count": "6",
    "must_include": "a paper rocket",
    "avoid": "spiders",
}


def test_get_form_returns_html() -> None:
    response = asyncio.run(handlers.get_form(_request()))

    assert response.status_code == 200
    assert b"Make a storybook" in response.body


def test_build_brief_assembles_the_nested_child() -> None:
    brief = handlers._build_brief(_VALID_FORM)

    assert brief.child.name == "Maryam"
    assert brief.child.age == 5
    assert brief.child.pronouns.value == "she/her"
    assert brief.premise == "a fox who wants to visit the moon"
    assert brief.page_count == 6


def test_build_brief_splits_comma_separated_lists() -> None:
    brief = handlers._build_brief(_VALID_FORM)

    assert brief.child.interests == ["foxes", "the moon"]
    assert brief.must_include == ["a paper rocket"]
    assert brief.avoid == ["spiders"]


def test_build_brief_treats_blank_lists_as_empty() -> None:
    payload = {**_VALID_FORM, "interests": "", "must_include": "  ", "avoid": ","}

    brief = handlers._build_brief(payload)

    assert brief.child.interests == []
    assert brief.must_include == []
    assert brief.avoid == []


def test_post_plan_rejects_a_bad_brief_with_400_and_creates_no_job(
    _fresh_registry: JobRegistry,
) -> None:
    payload = {**_VALID_FORM, "age": "99"}  # ChildProfile caps at 12

    response = asyncio.run(handlers.post_plan(_request(payload)))

    assert response.status_code == 400
    assert _fresh_registry._jobs == {}


def test_post_plan_rejects_a_missing_premise_with_400() -> None:
    payload = {**_VALID_FORM, "premise": ""}

    response = asyncio.run(handlers.post_plan(_request(payload)))

    assert response.status_code == 400


def test_post_plan_starts_a_job_and_returns_its_id(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    async def _plan(
        brief: Any, on_task_result: Any = None, **kwargs: Any
    ) -> StoryOutline:
        return _outline()

    monkeypatch.setattr(handlers, "run_outline_pipeline", _plan)

    async def _drive() -> Any:
        response = await handlers.post_plan(_request(_VALID_FORM))
        # Let the background task finish before asserting on its effect.
        for _ in range(10):
            await asyncio.sleep(0)
        return response

    response = asyncio.run(_drive())
    body = json.loads(response.body)

    assert response.status_code == 202
    assert body["job_id"]


def test_planning_records_the_outline_and_awaits_approval(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    async def _plan(
        brief: Any, on_task_result: Any = None, **kwargs: Any
    ) -> StoryOutline:
        return _outline()

    monkeypatch.setattr(handlers, "run_outline_pipeline", _plan)

    async def _drive() -> str:
        response = await handlers.post_plan(_request(_VALID_FORM))
        job_id = json.loads(response.body)["job_id"]
        for _ in range(10):
            await asyncio.sleep(0)
        return job_id

    job_id = asyncio.run(_drive())
    job = _fresh_registry.get(job_id)

    assert job.state is JobState.AWAITING_APPROVAL
    assert job.outline.title == "Maryam and the Paper Rocket"


def test_a_configuration_error_fails_the_job_with_its_message(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    async def _boom(
        brief: Any, on_task_result: Any = None, **kwargs: Any
    ) -> StoryOutline:
        raise ConfigurationError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(handlers, "run_outline_pipeline", _boom)

    async def _drive() -> str:
        response = await handlers.post_plan(_request(_VALID_FORM))
        job_id = json.loads(response.body)["job_id"]
        for _ in range(10):
            await asyncio.sleep(0)
        return job_id

    job_id = asyncio.run(_drive())
    job = _fresh_registry.get(job_id)

    assert job.state is JobState.FAILED
    assert job.error == "GOOGLE_API_KEY is not set"


def test_an_unexpected_error_fails_the_job_without_leaking_its_message(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    # An unexpected exception's text is written for a developer. A parent gets a
    # sentence; the traceback goes to the log.
    async def _boom(
        brief: Any, on_task_result: Any = None, **kwargs: Any
    ) -> StoryOutline:
        raise ValueError("connection pool exhausted at 0x7f3a")

    monkeypatch.setattr(handlers, "run_outline_pipeline", _boom)

    async def _drive() -> str:
        response = await handlers.post_plan(_request(_VALID_FORM))
        job_id = json.loads(response.body)["job_id"]
        for _ in range(10):
            await asyncio.sleep(0)
        return job_id

    job_id = asyncio.run(_drive())
    job = _fresh_registry.get(job_id)

    assert job.state is JobState.FAILED
    assert "0x7f3a" not in job.error
    assert "connection pool" not in job.error


def test_get_job_returns_404_for_an_unknown_id() -> None:
    response = asyncio.run(handlers.get_job(_request(path_params={"job_id": "nope"})))

    assert response.status_code == 404


def test_get_status_reports_state_and_detail(_fresh_registry: JobRegistry) -> None:
    brief = handlers._build_brief(_VALID_FORM)
    job = _fresh_registry.create(brief)
    _fresh_registry.update(job.id, detail="critiquing the outline")

    response = asyncio.run(
        handlers.get_status(_request(path_params={"job_id": job.id}))
    )
    body = json.loads(response.body)

    assert body["state"] == "planning"
    assert body["detail"] == "critiquing the outline"


def test_get_status_returns_404_for_an_unknown_id() -> None:
    response = asyncio.run(
        handlers.get_status(_request(path_params={"job_id": "nope"}))
    )

    assert response.status_code == 404


def test_no_handler_writes_to_stdout(capsys) -> None:
    # stdout carries JSON-RPC under stdio transport, so a stray print corrupts
    # the protocol.
    asyncio.run(handlers.get_form(_request()))
    asyncio.run(handlers.get_status(_request(path_params={"job_id": "nope"})))

    assert capsys.readouterr().out == ""


def _story() -> Story:
    return Story(
        outline=_outline(),
        page_plan=PagePlan(
            pages=[
                ScenePlan(
                    page_number=number,
                    beat_position=number,
                    setting="A garden",
                    visual_action="A fox appears",
                    emotional_shift="surprise",
                )
                for number in range(1, 5)
            ]
        ),
        pages=[
            StoryPage(page_number=1, text="A fox appeared."),
            StoryPage(page_number=2, text="The fox looked up."),
            StoryPage(page_number=3, text="She folded the paper."),
            StoryPage(page_number=4, text="The rocket flew."),
        ],
    )


def _awaiting_job(fresh: JobRegistry) -> Any:
    brief = handlers._build_brief(_VALID_FORM)
    job = fresh.create(brief)
    return fresh.update(job.id, state=JobState.AWAITING_APPROVAL, outline=_outline())


def test_approve_writes_the_book_from_the_jobs_outline(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    # THE load-bearing test. Everything this design claims over the MCP path
    # rests on the server building from its own record, never from the request.
    seen: dict[str, Any] = {}

    async def _write(
        brief: Any, outline: Any, on_task_result: Any = None, **kwargs: Any
    ) -> Story:
        seen["outline"] = outline
        return _story()

    monkeypatch.setattr(handlers, "run_story_pipeline", _write)
    job = _awaiting_job(_fresh_registry)

    tampered = _outline().model_dump()
    tampered["title"] = "A Completely Different Book"

    async def _drive() -> None:
        await handlers.post_approve(_request({"outline": tampered}, {"job_id": job.id}))
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(_drive())

    assert seen["outline"].title == "Maryam and the Paper Rocket"


def test_double_approve_returns_409_and_writes_once(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    calls = {"count": 0}

    async def _write(
        brief: Any, outline: Any, on_task_result: Any = None, **kwargs: Any
    ) -> Story:
        calls["count"] += 1
        return _story()

    monkeypatch.setattr(handlers, "run_story_pipeline", _write)
    job = _awaiting_job(_fresh_registry)

    async def _drive() -> Any:
        first = await handlers.post_approve(_request({}, {"job_id": job.id}))
        second = await handlers.post_approve(_request({}, {"job_id": job.id}))
        for _ in range(20):
            await asyncio.sleep(0)
        return first, second

    first, second = asyncio.run(_drive())

    assert first.status_code == 202
    assert second.status_code == 409
    assert calls["count"] == 1


def test_approve_returns_404_for_an_unknown_job() -> None:
    response = asyncio.run(handlers.post_approve(_request({}, {"job_id": "nope"})))

    assert response.status_code == 404


def test_revise_from_the_wrong_state_returns_409(
    _fresh_registry: JobRegistry,
) -> None:
    brief = handlers._build_brief(_VALID_FORM)
    job = _fresh_registry.create(brief)  # still PLANNING

    response = asyncio.run(
        handlers.post_revise(_request({"note": "make it shorter"}, {"job_id": job.id}))
    )

    assert response.status_code == 409


def test_revise_appends_the_note_and_keeps_the_original_premise(
    monkeypatch: pytest.MonkeyPatch, _fresh_registry: JobRegistry
) -> None:
    seen: dict[str, Any] = {}

    async def _plan(
        brief: Any, on_task_result: Any = None, **kwargs: Any
    ) -> StoryOutline:
        seen["premise"] = brief.premise
        return _outline()

    monkeypatch.setattr(handlers, "run_outline_pipeline", _plan)
    job = _awaiting_job(_fresh_registry)

    async def _drive() -> None:
        await handlers.post_revise(
            _request(
                {"note": "Make Maryam the one who wants to go"}, {"job_id": job.id}
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)

    asyncio.run(_drive())

    assert "Make Maryam the one who wants to go" in seen["premise"]
    assert "a fox who wants to visit the moon" in seen["premise"]
    assert (
        _fresh_registry.get(job.id).original_premise
        == "a fox who wants to visit the moon"
    )


def test_get_book_returns_404_when_the_story_is_not_written(
    _fresh_registry: JobRegistry,
) -> None:
    job = _awaiting_job(_fresh_registry)

    response = asyncio.run(handlers.get_book(_request(path_params={"job_id": job.id})))

    assert response.status_code == 404


def test_get_file_serves_an_allowlisted_artifact(
    tmp_path: Path, _fresh_registry: JobRegistry
) -> None:
    run = tmp_path / "20260814-101500-a-fox"
    run.mkdir()
    (run / "page-01.jpg").write_bytes(b"jpeg-bytes")
    brief = handlers._build_brief(_VALID_FORM)
    job = _fresh_registry.create(brief)
    _fresh_registry.update(job.id, run_directory=run)

    response = asyncio.run(
        handlers.get_file(
            _request(path_params={"job_id": job.id, "name": "page-01.jpg"})
        )
    )

    assert response.status_code == 200
    assert response.media_type == "image/jpeg"
    assert response.body == b"jpeg-bytes"


def test_get_file_rejects_traversal_with_404(
    tmp_path: Path, _fresh_registry: JobRegistry
) -> None:
    run = tmp_path / "20260814-101500-a-fox"
    run.mkdir()
    (tmp_path / "secret.env").write_text("XAI_API_KEY=real")
    brief = handlers._build_brief(_VALID_FORM)
    job = _fresh_registry.create(brief)
    _fresh_registry.update(job.id, run_directory=run)

    response = asyncio.run(
        handlers.get_file(
            _request(path_params={"job_id": job.id, "name": "../secret.env"})
        )
    )

    assert response.status_code == 404


def test_get_file_rejects_brief_json_even_though_it_exists(
    tmp_path: Path, _fresh_registry: JobRegistry
) -> None:
    # brief.json holds the child's name. It is not media and must not be served.
    run = tmp_path / "20260814-101500-a-fox"
    run.mkdir()
    (run / "brief.json").write_text('{"child": {"name": "Maryam"}}')
    brief = handlers._build_brief(_VALID_FORM)
    job = _fresh_registry.create(brief)
    _fresh_registry.update(job.id, run_directory=run)

    response = asyncio.run(
        handlers.get_file(
            _request(path_params={"job_id": job.id, "name": "brief.json"})
        )
    )

    assert response.status_code == 404


def test_get_file_returns_404_when_the_job_has_no_run_directory(
    _fresh_registry: JobRegistry,
) -> None:
    brief = handlers._build_brief(_VALID_FORM)
    job = _fresh_registry.create(brief)

    response = asyncio.run(
        handlers.get_file(
            _request(path_params={"job_id": job.id, "name": "page-01.jpg"})
        )
    )

    assert response.status_code == 404
