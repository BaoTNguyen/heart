"""Auto-detect verifiers for `heart work` runs in arbitrary repos."""
from __future__ import annotations

import json
from pathlib import Path

from .taskspec import Verifier


def detect_verifiers(repo_path: str | Path) -> list[Verifier]:
    repo = Path(repo_path)
    verifiers: list[Verifier] = []

    has_py_tests = (
        (repo / "tests").is_dir()
        or any(repo.glob("test_*.py"))
        or any(repo.glob("tests/**/*.py"))
    )
    if has_py_tests:
        verifiers.append(Verifier(name="pytest", command="python3 -m pytest -x -q"))

    pkg = repo / "package.json"
    if pkg.exists():
        try:
            test_script = json.loads(pkg.read_text()).get("scripts", {}).get("test", "")
        except json.JSONDecodeError:
            test_script = ""
        if test_script and "no test specified" not in test_script:
            verifiers.append(Verifier(name="npm-test", command="npm test --silent"))

    if (repo / "Cargo.toml").exists():
        verifiers.append(Verifier(name="cargo-test", command="cargo test --quiet"))
    if (repo / "go.mod").exists():
        verifiers.append(Verifier(name="go-test", command="go test ./..."))

    return verifiers
