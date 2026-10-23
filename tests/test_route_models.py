"""route_models: an operator allowlist that routing honours and review ignores."""
from __future__ import annotations

import copy
import json

import pytest

from heart import routing
from heart.agents_api import models_json_path
from heart.taskspec import TaskSpec

STUB = {
    name: {"agent": agent, "skills": {"coding": 0.9}, "context": 200_000,
           "max_difficulty": "hard", "cost": cost}
    for name, agent, cost in (("opus", "claude", 3.0), ("sol", "api:sol", 2.5),
                              ("sonnet", "api:sonnet", 0.5), ("terra", "api:terra", 0.3))
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))


@pytest.fixture
def emitted(monkeypatch):
    seen = []
    monkeypatch.setattr(routing, "emit", lambda *a, **kw: seen.append(kw))
    return seen


def _want(difficulty="medium", **kw):
    return routing.demand(TaskSpec(task_id="t", repo_path=".", base_commit="x",
                                   prompt="add a function", skills=["coding"],
                                   difficulty=difficulty, **kw))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_allowlist_excludes_others(difficulty, emitted):
    choice = routing.pick(_want(difficulty), manifest=STUB, route_models=["opus", "sol"])
    assert choice.name in ("opus", "sol")
    assert "sonnet:not in route_models" in emitted[-1]["filtered"]
    assert "terra:not in route_models" in emitted[-1]["filtered"]


def test_empty_allowlist_is_off():
    assert routing.pick(_want(), manifest=STUB, route_models=[]).name == "terra"


def test_no_config_is_off():
    assert not models_json_path().exists()
    assert routing.route_models() == []
    assert routing.pick(_want(), manifest=STUB).name == "terra"


def test_unknown_name_warns_and_is_dropped(capsys):
    assert routing.pick(_want(), manifest=STUB, route_models=["opus", "nope"]).name == "opus"
    assert "nope" in capsys.readouterr().err
    assert routing.pick(_want(), manifest=STUB, route_models=["nope"]).name == "terra"


def test_fallback_stays_inside_allowlist():
    manifest = copy.deepcopy(STUB)
    for n in ("opus", "sol"):
        manifest[n]["context"] = 32_000
    choice = routing.pick(_want(min_context=100_000), manifest=manifest,
                          route_models=["opus", "sol"])
    assert choice.name in ("opus", "sol")
    assert choice.reason.startswith("no model cleared constraints")

    manifest = copy.deepcopy(STUB)
    for n in ("opus", "sol"):
        manifest[n]["max_difficulty"] = "easy"
    choice = routing.pick(_want("hard"), manifest=manifest, route_models=["opus", "sol"])
    assert choice.name in ("opus", "sol")


def test_config_read_on_callers_manifest_path():
    path = models_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": STUB, "route_models": ["opus", "sol"]}))
    choice = routing.pick(_want(), manifest=routing.load_manifest())
    assert choice.name in ("opus", "sol")


def test_empty_manifest_never_reads_route_models(monkeypatch):
    def boom():
        raise AssertionError("route_models read before the empty-manifest branch")
    monkeypatch.setattr(routing, "route_models", boom)
    assert routing.pick(_want(), manifest={}).reason == "no model manifest; tier default"


@pytest.mark.parametrize("raw", ['{"route_models": "opus"}', '{"route_models": []}',
                                 '{"route_models": ["opus", 3]}', '{"route_models": null}',
                                 "not json", "[1, 2]"])
def test_route_models_bad_config_is_empty(raw):
    path = models_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    assert routing.route_models() == []


def test_route_models_reads_config():
    path = models_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": STUB, "route_models": ["sol"]}))
    assert routing.route_models() == ["sol"]


def test_unknown_name_warning_text(capsys):
    routing.pick(_want(), manifest=STUB, route_models=["opus", "nope"])
    assert "heart: route_models: unknown model 'nope' ignored" in capsys.readouterr().err


def test_allowlist_tie_breaks_cheapest():
    assert routing.pick(_want(), manifest=STUB, route_models=["opus", "sol"]).name == "sol"


def test_fallback_order_and_filtered(emitted):
    manifest = copy.deepcopy(STUB)
    manifest["opus"].update(context=32_000, max_difficulty="medium")
    manifest["sol"].update(context=16_000, max_difficulty="hard")
    choice = routing.pick(_want("hard", min_context=100_000), manifest=manifest,
                          route_models=["opus", "sol"])
    assert choice.name == "sol"  # difficulty rank beats context
    assert choice.effort == "high"
    manifest["sol"]["max_difficulty"] = "medium"
    assert routing.pick(_want("hard", min_context=100_000), manifest=manifest,
                        route_models=["opus", "sol"]).name == "opus"  # then context
    f = emitted[-1]["filtered"]
    assert "sonnet:not in route_models" in f and "opus:context<100000" in f


def test_fallback_honours_availability():
    manifest = copy.deepcopy(STUB)
    for n in ("opus", "sol"):
        manifest[n]["context"] = 32_000
    up = lambda agent: agent != "claude"
    choice = routing.pick(_want(min_context=100_000), manifest=manifest, available=up,
                          route_models=["opus", "sol"])
    assert choice.name == "sol"
    with pytest.raises(ValueError, match="no available model satisfies context constraints"):
        routing.pick(_want(), manifest=STUB, available=lambda a: False,
                     route_models=["opus", "sol"])
