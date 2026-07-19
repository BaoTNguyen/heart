"""Export scored episodes to a single JSONL — the contract training code consumes.

ponytail: arteries decisions/events join lands here once the arteries decision
ledger exists; until then episodes carry only heart-side data.
"""
from __future__ import annotations

import json
from pathlib import Path


def export_episodes(runs_dir: str | Path, out_path: str | Path) -> int:
    episodes = []
    for ep_file in sorted(Path(runs_dir).glob("*/episode.json")):
        episodes.append(json.loads(ep_file.read_text()))
    with open(out_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return len(episodes)
