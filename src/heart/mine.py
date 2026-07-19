"""Mine TaskSpecs from a repo's git history: commits that touched both tests and
source, where the tests already existed at the parent commit.

ponytail: commits that *introduce* tests are skipped — porting new tests back to
the base commit is SWE-bench machinery, add when mined-task volume runs dry.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

TEST_HINTS = ("test_", "_test.", "/tests/", "/test/", ".spec.", ".test.")
CODE_EXTS = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb", ".c", ".cpp")


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def _is_test(path: str) -> bool:
    return any(h in path.lower() for h in TEST_HINTS)


def mine(
    repo: str,
    out_dir: str | Path,
    limit: int = 20,
    scan: int = 500,
    test_cmd: str = "python3 -m pytest -x {files}",
    timeout_seconds: int = 300,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = str(Path(repo).resolve())
    written: list[Path] = []

    for sha in _git(repo, "rev-list", "--no-merges", "-n", str(scan), "HEAD").split():
        files = [f for f in _git(repo, "show", "--name-only", "--format=", sha).splitlines() if f]
        tests = [f for f in files if _is_test(f) and f.endswith(CODE_EXTS)]
        srcs = [f for f in files if not _is_test(f) and f.endswith(CODE_EXTS)]
        if not tests or not srcs:
            continue
        try:
            parent = _git(repo, "rev-parse", f"{sha}^").strip()
        except RuntimeError:
            continue  # root commit
        exists_at_parent = all(
            subprocess.run(
                ["git", "-C", repo, "cat-file", "-e", f"{parent}:{f}"], capture_output=True
            ).returncode == 0
            for f in tests
        )
        if not exists_at_parent:
            continue

        subject = _git(repo, "show", "-s", "--format=%s", sha).strip()
        task_id = f"mined-{Path(repo).name}-{sha[:10]}"
        spec = {
            "task_id": task_id,
            "repo_path": repo,
            "base_commit": parent,
            "fix_commit": sha,
            "prompt": (
                f"Make the following tests pass: {', '.join(tests)}.\n"
                f"Context: {subject}\n"
                f"Do not modify the test files."
            ),
            "denied_paths": tests,
            "public_verifiers": [{"name": "tests", "command": test_cmd.format(files=" ".join(tests))}],
            "timeout_seconds": timeout_seconds,
            "tags": ["mined"],
        }
        path = out_dir / f"{task_id}.json"
        path.write_text(json.dumps(spec, indent=2))
        written.append(path)
        if len(written) >= limit:
            break
    return written
