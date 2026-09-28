"""Prose must name heart's state where it now lives, under ~/.vascular.

sandbox.py is skipped on purpose: its old path quotes an error message
measured at the time and stays as history. ~/.config/heart/secrets is out of
scope and not in OLD.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OLD = re.compile(r"\.config/heart/models\.json|\.cache/heart-ws|\.local/share/heart")
FILES = ["README.md", "SPINE.md", "PRICING.md", "tests/conftest.py",
         "tests/test_heart.py", "src/heart/agents_api.py", "src/heart/routing.py"]


@pytest.mark.parametrize("rel", FILES)
def test_no_old_heart_paths(rel):
    hits = [f"{rel}:{n}: {line}" for n, line in
            enumerate((ROOT / rel).read_text().splitlines(), 1) if OLD.search(line)]
    assert not hits, "\n".join(hits)


def test_agents_md_rate_card_uses_new_path():
    text = (ROOT / "AGENTS.md").read_text()
    card = text.split("## Rate card", 1)[1]
    assert not OLD.search(card)
    assert "~/.vascular/config/heart/models.json" in card


def test_spine_mentions_vascular_home_override():
    assert "VASCULAR_HOME" in (ROOT / "SPINE.md").read_text()


def test_src_heart_clean_except_sandbox_history():
    hits = [f"{p.name}:{n}" for p in (ROOT / "src/heart").rglob("*.py")
            if p.name != "sandbox.py"
            for n, line in enumerate(p.read_text().splitlines(), 1) if OLD.search(line)]
    assert not hits, hits
