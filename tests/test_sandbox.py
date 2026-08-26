"""The task spec is the privilege boundary.

path_violations reads allowed_paths off the finished diff, which catches a
breach that the mount table could have made impossible. These pin the
translation: what the spec permits is what the container can write.
"""
from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart.sandbox import CONTEXT, JOURNAL, WORK, profile_for, verifier_profile_for
from heart.taskspec import TaskSpec


def _task(**kw) -> TaskSpec:
    base = {"task_id": "t1", "repo_path": "/repo", "base_commit": "abc", "prompt": "do it"}
    return TaskSpec(**{**base, **kw})


def _profile(task):
    return profile_for(task, "/ws/a", "/ctx", "/jr", env={"ARTERIES_RUN_ID": "r1"})


def _mount(profile, target):
    return next((m for m in profile.mounts if m.target == target), None)


def test_unrestricted_task_gets_a_writable_worktree():
    p = _profile(_task())
    assert _mount(p, WORK).writable is True


def test_allowed_paths_invert_the_default():
    """Saying "only these" is exact; denying everything else is a list you get
    wrong. So the worktree drops to read-only and each allowance is layered on."""
    p = _profile(_task(allowed_paths=["src", "tests"]))
    assert _mount(p, WORK).writable is False
    assert _mount(p, f"{WORK}/src").writable is True
    assert _mount(p, f"{WORK}/tests").writable is True


def test_denied_paths_win_over_allowed():
    p = _profile(_task(allowed_paths=["src"], denied_paths=["src/secrets"]))
    targets = [m.target for m in p.mounts]
    assert targets.index(f"{WORK}/src") < targets.index(f"{WORK}/src/secrets"), \
        "the denial must be layered after the allowance to win"
    assert _mount(p, f"{WORK}/src/secrets").writable is False


def test_context_is_read_only_and_journal_is_writable():
    p = _profile(_task())
    assert _mount(p, CONTEXT).writable is False
    assert _mount(p, JOURNAL).writable is True


@pytest.mark.parametrize("escape", ["../..", "/etc", "src/../../..", ""])
def test_a_path_that_escapes_the_worktree_is_refused(escape):
    """A spec is data. A denied_paths entry of '../..' must not mount the host."""
    p = _profile(_task(denied_paths=[escape]))
    for mount in p.mounts:
        assert mount.target in (WORK, CONTEXT, JOURNAL) or mount.target.startswith(WORK + "/")
        assert not mount.source.startswith("/ws/a/..")


def test_network_is_denied_unless_the_task_asks():
    assert _profile(_task()).network == "none"
    assert _profile(_task(network="model")).network != "none"
    assert _profile(_task(network="nonsense")).network == "none"


def test_the_container_is_told_where_the_journal_is_not_how_to_reach_a_database():
    p = _profile(_task())
    assert p.env["EVENT_JOURNAL_DIR"] == JOURNAL
    joined = " ".join(f"{k}={v}" for k, v in p.env.items())
    for secret in ("PGPASSWORD", "postgres", "DB_CONFIG", "docker.sock"):
        assert secret not in joined


def test_docker_args_drop_privileges():
    args = " ".join(_profile(_task()).docker_args())
    for flag in ("--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--pids-limit", "--memory", "--network none"):
        assert flag in args


def test_the_timeout_comes_from_the_task():
    assert _profile(_task(timeout_seconds=900)).timeout_seconds == 900


# --- verifier role -------------------------------------------------------
# Same mechanism, different arguments. What differs is what the role is allowed
# to be, not how strongly it is contained.


def _verifier(task):
    return verifier_profile_for(task, "/ws/a", "/jr")


def test_a_verifier_cannot_write_the_tree_it_judges():
    """The reward-integrity property. A verifier that can edit the code it is
    scoring collapses "produce" and "judge" into one step."""
    p = _verifier(_task(allowed_paths=["src"]))
    assert _mount(p, WORK).writable is False
    assert all(m.writable is False for m in p.mounts if m.target.startswith(WORK))


def test_a_verifier_never_gets_the_network_even_when_the_task_asks():
    assert _verifier(_task(network="model")).network == "none"
    assert _verifier(_task(network="build")).network == "none"


def test_a_verifier_cannot_read_the_continuity_packet():
    """It judges code, not memory. A verifier that can read the packet can be
    steered by it."""
    assert _mount(_verifier(_task()), CONTEXT) is None


def test_a_verifier_still_reports_through_the_journal():
    assert _mount(_verifier(_task()), JOURNAL).writable is True


def test_test_caches_land_in_memory_not_the_read_only_tree():
    args = " ".join(_verifier(_task()).docker_args())
    assert f"--tmpfs {WORK}/.pytest_cache" in args
    assert "PYTHONDONTWRITEBYTECODE=1" in args


def test_both_roles_render_from_one_profile_type():
    agent = profile_for(_task(), "/ws/a", "/ctx", "/jr")
    verifier = _verifier(_task())
    assert type(agent) is type(verifier)
    for flag in ("--read-only", "--cap-drop ALL", "--security-opt no-new-privileges"):
        assert flag in " ".join(agent.docker_args())
        assert flag in " ".join(verifier.docker_args())


# --- git inside the container -------------------------------------------
# A worktree's .git is a file naming an absolute host path. Mount only the tree
# and every git command fails, including the `git diff` the review roles run.


def test_the_object_store_is_mounted_read_only_at_its_host_path(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git" / "worktrees" / "ws1").mkdir(parents=True)
    task = _task(repo_path=str(repo))

    p = profile_for(task, tmp_path / "ws1", "/ctx", "/jr")
    git = _mount(p, str(repo / ".git"))

    assert git is not None, "without this, git cannot resolve the gitdir pointer"
    assert git.writable is False
    assert git.source == git.target, "the pointer is absolute, so the path must match"


def test_the_per_worktree_metadata_stays_writable(tmp_path):
    """git updates this worktree's index and HEAD to answer a diff. It holds no
    objects and no refs belonging to anything else."""
    repo = tmp_path / "repo"
    (repo / ".git" / "worktrees" / "ws1").mkdir(parents=True)

    p = profile_for(_task(repo_path=str(repo)), tmp_path / "ws1", "/ctx", "/jr")
    per_wt = _mount(p, str(repo / ".git" / "worktrees" / "ws1"))

    assert per_wt is not None and per_wt.writable is True


def test_a_repo_without_a_git_dir_needs_no_git_mounts(tmp_path):
    """A clone or a plain export carries its own .git inside the tree."""
    p = profile_for(_task(repo_path=str(tmp_path / "nope")), tmp_path / "ws1", "/ctx", "/jr")
    assert all(".git" not in m.target for m in p.mounts)


def test_a_verifier_gets_no_git_at_all(tmp_path):
    """It runs commands against a tree; it has no reason to read history, and
    the object store is the largest thing it could be given."""
    repo = tmp_path / "repo"
    (repo / ".git" / "worktrees" / "ws1").mkdir(parents=True)
    p = verifier_profile_for(_task(repo_path=str(repo)), tmp_path / "ws1", "/jr")
    assert all(".git" not in m.target for m in p.mounts)


# --- when the predicted scope is wrong ----------------------------------
# A profile derived from a task spec is a prediction. When it is too tight the
# agent writes nothing, and an empty diff reads as "did nothing" unless the
# refusal is named.


def test_a_refused_write_is_named_not_read_as_no_change(tmp_path):
    from heart.episode import _scope_denials

    (tmp_path / "implement.log").write_text(
        "opening src/app.py\nOSError: [Errno 30] Read-only file system: 'src/app.py'\n")

    denials = _scope_denials(tmp_path)

    assert denials and "implement" in denials[0]
    assert "read-only file system" in denials[0].lower()


def test_a_git_object_refusal_counts_too(tmp_path):
    from heart.episode import _scope_denials
    (tmp_path / "test.log").write_text(
        "error: insufficient permission for adding an object to repository database\n")
    assert _scope_denials(tmp_path)


def test_a_clean_log_reports_no_denial(tmp_path):
    from heart.episode import _scope_denials
    (tmp_path / "implement.log").write_text("wrote 3 files\nall tests passed\n")
    assert _scope_denials(tmp_path) == []


def test_review_reasoning_survives_an_approval(tmp_path):
    """An APPROVE that also flags a real problem is the most considered thing an
    episode produces. It used to be deleted with the run directory."""
    from heart.episode import _review_notes

    log = tmp_path / "review.log"
    log.write_text("The retry logic will break under concurrency.\nAPPROVE looks correct\n")

    notes = _review_notes(log)

    assert "retry logic will break" in notes


def test_missing_review_log_yields_nothing():
    from heart.episode import _review_notes
    assert _review_notes(Path("/nonexistent/review.log")) == ""


def test_a_containerised_agent_is_pointed_at_a_reachable_model():
    """127.0.0.1 inside a container is the container. An agent resolving the
    host's endpoint would fail to connect with no useful error."""
    from heart.sandbox import CONTAINER_MODEL_ENDPOINT

    assert profile_for(_task(network="model"), "/ws", "/c", "/i").env[
        "HEART_API_ENDPOINT"] == CONTAINER_MODEL_ENDPOINT
    assert "HEART_API_ENDPOINT" not in profile_for(_task(), "/ws", "/c", "/i").env


def test_a_scope_denial_carries_no_reward_signal():
    """0.0 asserts the episode did badly. A sandbox refusing writes says nothing
    about the model, and training on it teaches that the task is impossible."""
    from heart.episode import UNSCOREABLE

    assert "scope_denied" in UNSCOREABLE
    assert "pass" not in UNSCOREABLE and "fail" not in UNSCOREABLE


def test_a_refusal_on_forbidden_ground_is_not_an_escape_from_scoring():
    """The hack this closes: an agent heading for a bad score writes one byte
    into a denied path, gets an empty diff plus a refusal, and scope_denied
    would hand it reward=None instead of 0.0."""
    from heart.episode import _probed_forbidden

    denial = ["implement: [Errno 30] Read-only file system: 'src/secrets/key.pem'"]

    assert _probed_forbidden(denial, ["src/secrets"]) is True
    assert _probed_forbidden(denial, ["config"]) is False
    assert _probed_forbidden(denial, []) is False


def test_a_refusal_on_permitted_ground_stays_a_misconfiguration():
    from heart.episode import _probed_forbidden

    denial = ["implement: [Errno 30] Read-only file system: 'src/app.py'"]
    assert _probed_forbidden(denial, ["src/secrets"]) is False


def test_paths_are_extracted_quoted_or_bare():
    from heart.episode import _denial_paths

    assert "src/app.py" in _denial_paths("Read-only file system: 'src/app.py'")
    assert "src/app.py" in _denial_paths("permission denied writing src/app.py")


def test_a_prefix_collision_does_not_count_as_forbidden():
    """src_gen/ must not match a denial of src/ -- the same bug path_violations
    has a test for."""
    from heart.episode import _probed_forbidden

    assert _probed_forbidden(["EACCES: 'src_gen/x.py'"], ["src"]) is False
