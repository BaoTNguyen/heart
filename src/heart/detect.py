"""Auto-detect verifiers for `heart work` runs in arbitrary repos."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from .taskspec import Verifier


def _has_ruff_config(repo: Path) -> bool:
    if (repo / "ruff.toml").exists() or (repo / ".ruff.toml").exists():
        return True
    pyproject = repo / "pyproject.toml"
    return pyproject.exists() and "[tool.ruff]" in pyproject.read_text(errors="replace")


def _has_mypy_config(repo: Path) -> bool:
    if (repo / "mypy.ini").exists() or (repo / ".mypy.ini").exists():
        return True
    pyproject = repo / "pyproject.toml"
    return pyproject.exists() and "[tool.mypy]" in pyproject.read_text(errors="replace")


def _eslint_config_exists(repo: Path) -> bool:
    names = (
        ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
        ".eslintrc.yml", ".eslintrc.yaml", "eslint.config.js", "eslint.config.mjs",
    )
    return any((repo / n).exists() for n in names)


def _biome_config_exists(repo: Path) -> bool:
    return (repo / "biome.json").exists() or (repo / "biome.jsonc").exists()


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

    # Static verifiers appended after test verifiers; each requires both the
    # tool's config and the tool itself locally present — never anything that
    # downloads (npx, pip install, etc.), since sandboxed verifiers have no
    # network.
    if _has_ruff_config(repo) and shutil.which("ruff"):
        verifiers.append(Verifier(name="ruff", command="ruff check"))

    if _has_mypy_config(repo) and (
        importlib.util.find_spec("mypy") is not None or shutil.which("mypy")
    ):
        verifiers.append(Verifier(name="mypy", command="python3 -m mypy ."))

    tsc = repo / "node_modules" / ".bin" / "tsc"
    if (repo / "tsconfig.json").exists() and tsc.exists():
        verifiers.append(Verifier(name="tsc", command="node_modules/.bin/tsc --noEmit"))

    eslint = repo / "node_modules" / ".bin" / "eslint"
    if _eslint_config_exists(repo) and eslint.exists():
        verifiers.append(Verifier(name="eslint", command="node_modules/.bin/eslint ."))

    biome = repo / "node_modules" / ".bin" / "biome"
    if _biome_config_exists(repo) and biome.exists():
        verifiers.append(Verifier(name="biome", command="node_modules/.bin/biome check ."))

    return verifiers
