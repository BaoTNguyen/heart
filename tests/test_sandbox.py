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

from dockerprobe import DOCKER_USABLE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart.sandbox import (CONTEXT, HOME, JOURNAL, WORK, profile_for,
                           verifier_profile_for)
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
    from heart.sandbox import AGENT_BIN, NPM_PREFIX

    for mount in p.mounts:
        if mount.target == NPM_PREFIX or mount.target.startswith(AGENT_BIN + "/"):
            continue  # heart's own tool mounts: fixed targets, not task-derived
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


def test_a_container_refusal_names_a_path_the_task_spec_can_use():
    """The consumer of `sandbox.denied` decides whether to widen allowed_paths,
    and allowed_paths is worktree-relative. A refusal inside the container is
    absolute and carries heart's own /work mount point, which means nothing to
    that decision -- and an absolute path is exactly what the downstream copy
    of this parser could not read, so every container refusal distilled to
    nothing at all."""
    from heart.episode import _refused_paths

    assert _refused_paths(
        ["solo: sh: 1: cannot create /work/pyproject.toml: Read-only file system"]
    ) == ["pyproject.toml"]
    assert _refused_paths(
        ["OSError: [Errno 30] Read-only file system: '/work/config/app.yaml'"]
    ) == ["config/app.yaml"]


def test_a_shell_does_not_get_reported_as_the_refused_path():
    """`/bin/sh: 1: cannot create test_calc.py: Read-only file system` -- the
    shell names itself first, and the path it could not write has no slash. A
    real episode recorded scope_refused_paths as ['bin/sh'], which tells a
    reader nothing and would widen a scope in the wrong direction."""
    from heart.episode import _refused_paths

    assert _refused_paths(
        ["test: /bin/sh: 1: cannot create test_calc.py: Read-only file system"]
    ) == ["test_calc.py"]
    assert _refused_paths(
        ["test: bash: line 2: cannot create build/out.txt: Permission denied"]
    ) == ["build/out.txt"]


def test_one_path_refused_twice_in_two_dialects_is_one_path():
    # roles fail in whatever language their tooling speaks; a consumer widening
    # a scope should see the path once, not once per phrasing
    from heart.episode import _refused_paths

    assert _refused_paths([
        "implement: PermissionError: src/a.py: Permission denied",
        "test: sh: cannot create /work/src/a.py: Read-only file system",
    ]) == ["src/a.py"]


# --- agent CLIs are mounted from the host, not baked into the image -----
# Baking costs the CLI's full size per agent and a rebuild on every update;
# mounting costs nothing and runs whatever build the host has. What these pin
# is the part that is not obvious: the three shapes a CLI install takes.


def test_every_mounted_tool_is_read_only_and_lands_under_the_agent_prefix():
    from heart.sandbox import AGENT_BIN, NPM_PREFIX, agent_tool_mounts

    for m in agent_tool_mounts():
        assert m.writable is False, f"{m.target} must not be writable"
        assert m.target == NPM_PREFIX or m.target.startswith(AGENT_BIN + "/")


def test_a_bundled_launcher_gets_its_whole_directory_and_a_place_on_path():
    """cursor-agent is a shell script that runs the node binary sitting beside
    it. Mount the script alone and it dies on
    '/opt/agent-bin/node: No such file or directory', so the bundle has to come
    with it -- and a directory heart invented is one nothing in the image can
    name, which is why heart supplies PATH rather than extending the image's."""
    from heart.sandbox import agent_tool_mounts, agent_tool_path

    path = agent_tool_path().split(":")
    for m in agent_tool_mounts():
        if m.target.endswith(".d"):
            assert m.target in path


def test_a_versioned_single_file_binary_is_renamed_back_to_its_command():
    # claude resolves to .../claude/versions/2.1.246 -- mounted under that name
    # nothing would ever find it
    import shutil

    from heart.sandbox import AGENT_BIN, agent_tool_mounts

    if not shutil.which("claude"):
        pytest.skip("no claude on this host")
    targets = [m.target for m in agent_tool_mounts()]
    assert f"{AGENT_BIN}/claude" in targets or f"{AGENT_BIN}/claude.d" in targets


def test_the_container_path_puts_mounted_tools_ahead_of_the_image():
    from heart.sandbox import AGENT_BIN, NPM_PREFIX, agent_tool_path

    path = agent_tool_path().split(":")
    assert AGENT_BIN in path and f"{NPM_PREFIX}/bin" in path
    assert path.index(AGENT_BIN) < path.index("/usr/bin")


def test_an_agent_container_is_told_that_path():
    from heart.sandbox import agent_tool_path

    args = " ".join(_profile(_task()).docker_sbx_args())
    assert f"PATH={agent_tool_path()}" in args


# --- credentials ---------------------------------------------------------


def test_only_named_variables_are_forwarded(monkeypatch):
    """An allowlist, never a copy of the environment: a shell that can run
    heart holds every credential its operator has, and 'the container has no
    network' is a weak consolation once the keys are already inside it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-demo")
    monkeypatch.setenv("A_PRIVATE_THING", "hunter2")
    monkeypatch.setenv("HEART_SANDBOX_ENV", "ANTHROPIC_API_KEY,NEVER_SET")
    env = _profile(_task()).env
    assert env["ANTHROPIC_API_KEY"] == "sk-demo"
    assert "A_PRIVATE_THING" not in env
    assert "NEVER_SET" not in env  # unset names are skipped, not forwarded empty


def test_a_verifier_is_never_handed_a_credential(monkeypatch):
    # a verifier with an API key is an exfiltration path with a test suite
    # wrapped around it, and it has no model to call
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-demo")
    monkeypatch.setenv("HEART_SANDBOX_ENV", "ANTHROPIC_API_KEY")
    assert "ANTHROPIC_API_KEY" not in _verifier(_task()).env


def test_a_seat_file_lands_where_the_cli_looks_for_it(tmp_path, monkeypatch):
    """A subscription seat authenticates with an OAuth file under $HOME, not a
    key, so the file has to hold the same position inside the container that it
    holds on the host, and read-only so a refresh cannot rotate the host's
    token out from under it."""
    from heart.sandbox import HOME

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    cred = tmp_path / ".claude" / ".credentials.json"
    cred.parent.mkdir()
    cred.write_text("{}")
    monkeypatch.setenv("HEART_SANDBOX_HOME_FILES", str(cred))

    seat = _mount(_profile(_task()), f"{HOME}/.claude/.credentials.json")
    assert seat is not None and seat.writable is False
    # the CLI writes session state beside its credentials, and docker creates a
    # mountpoint's parent root-owned -- without the tmpfs, claude cannot start
    assert seat.source.endswith(".credentials.json")


def test_a_seat_file_is_never_handed_to_a_verifier(tmp_path, monkeypatch):
    from heart.sandbox import HOME

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    cred = tmp_path / ".credentials.json"
    cred.write_text("{}")
    monkeypatch.setenv("HEART_SANDBOX_HOME_FILES", str(cred))
    assert _mount(_verifier(_task()), f"{HOME}/.credentials.json") is None


def test_only_files_under_the_host_home_are_placed(tmp_path, monkeypatch):
    """Whole directories and paths outside home are skipped: ~/.claude is a home
    full of transcripts, not a credential store, and /etc/shadow has nowhere
    obvious to land."""
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    outside = tmp_path.parent / "elsewhere.json"
    outside.write_text("{}")
    a_dir = tmp_path / ".claude"
    a_dir.mkdir()
    monkeypatch.setenv("HEART_SANDBOX_HOME_FILES", f"{outside},{a_dir}")
    monkeypatch.delenv("HEART_SANDBOX_HOME_DIRS", raising=False)
    from heart.sandbox import home_file_mounts

    assert home_file_mounts() == ()


def test_the_dev_environment_comes_in_read_only_and_never_a_credential_store(tmp_path, monkeypatch):
    """Skills and plugins make a contained agent behave like the host one. They
    arrive read-only -- a writable plugin dir is a hook the agent can plant for
    the host to run -- and a directory holding a token is refused whole."""
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    claude = tmp_path / ".claude"
    (claude / "skills" / "tagore").mkdir(parents=True)
    (claude / "plugins").mkdir()
    (claude / ".credentials.json").write_text("{}")
    store = tmp_path / "store" / "settings.json"          # home-manager's symlink target
    store.parent.mkdir()
    store.write_text("{}")
    (claude / "settings.json").symlink_to(store)
    monkeypatch.setenv("HEART_SANDBOX_HOME_FILES", f"{claude / 'settings.json'}")
    monkeypatch.setenv("HEART_SANDBOX_HOME_DIRS",
                       f"{claude / 'skills'},{claude / 'plugins'},{claude},{tmp_path}")
    from heart.sandbox import home_file_mounts

    got = {(m.source, m.target) for m in home_file_mounts()}
    assert all(not m.writable for m in home_file_mounts())
    assert (str(store), f"{HOME}/.claude/settings.json") in got   # resolved source, link's place
    assert (str(claude / "skills"), f"{HOME}/.claude/skills") in got
    # plugin manifests name absolute host paths, so the host path works too
    assert (str(claude / "plugins"), str(claude / "plugins")) in got
    targets = {t for _, t in got}
    assert f"{HOME}/.claude" not in targets, "a dir with a credential in it is refused"
    assert HOME not in targets and str(tmp_path) not in targets, "the home itself is refused"


# --- docker-sbx: the same profile, a second renderer ----------------------
# The profile is data and docker_args() was the only function that knew Docker
# existed, so a second runtime is a renderer rather than a rewrite. What these
# pin is what survives the trip and what does not.


def test_the_mount_table_survives_the_second_renderer(tmp_path):
    from heart.sandbox import WORK

    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "secrets").mkdir()
    task = _task(allowed_paths=["src"], denied_paths=["secrets"], network="api")
    args = " ".join(profile_for(task, ws, tmp_path, tmp_path).docker_sbx_args())
    assert f"-v {ws}/src:{WORK}/src" in args             # allowed: writable
    assert f"-v {ws}/secrets:{WORK}/secrets:ro" in args  # denied: read-only
    assert f"-v {ws}:{WORK}:ro" in args                  # restricted tree: read-only


def test_a_mount_whose_source_is_missing_is_dropped_not_fatal(tmp_path):
    """`docker run` created a missing bind source; the plugin refuses the whole
    sandbox over one. A denied path that does not exist yet therefore cannot be
    pre-denied, and path_violations on the diff is the backstop."""
    from heart.sandbox import WORK

    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    task = _task(allowed_paths=["src"], denied_paths=["not-created-yet"], network="api")
    args = " ".join(profile_for(task, ws, tmp_path, tmp_path).docker_sbx_args())
    assert f"{WORK}/src" in args
    assert "not-created-yet" not in args


def test_every_sandbox_gets_a_unique_name(tmp_path):
    """The plugin names sandboxes by creation time to the second, so two started
    in the same second collide with "container name is already in use" -- which
    is every --candidates run and every parallel batch."""
    def name_of(p):
        a = p.docker_sbx_args()
        return a[a.index("--name") + 1]

    task = _task(network="api")
    names = {name_of(profile_for(task, tmp_path, tmp_path, tmp_path)) for _ in range(20)}
    assert len(names) == 20


def test_the_workspace_flag_is_not_aimed_at_the_worktree():
    """The plugin mounts --workspace read-write at its *host* path. Aimed at the
    worktree that is a second door onto a tree the mount table calls read-only.
    The journal inbox is already writable, so pointing there grants nothing."""
    from heart.sandbox import JOURNAL

    args = _profile(_task(allowed_paths=["src"], network="api")).docker_sbx_args()
    workspace = args[args.index("--workspace") + 1]
    assert workspace == "/jr"
    assert workspace == next(m.source for m in _profile(_task(network="api")).mounts
                             if m.target == JOURNAL)


def test_the_container_home_is_a_real_directory_the_agent_owns(monkeypatch, tmp_path):
    """This runtime has no --tmpfs, and a credential mounted beneath a directory
    docker invents lands root-owned and unwritable -- so HOME is a real path the
    Dockerfile creates and chowns to the agent uid."""
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    cred = tmp_path / ".claude" / ".credentials.json"
    cred.parent.mkdir()
    cred.write_text("{}")
    monkeypatch.setenv("HEART_SANDBOX_HOME_FILES", str(cred))
    args = " ".join(_profile(_task(network="api")).docker_sbx_args())
    assert f"-e HOME={HOME}" in args
    assert f"{HOME}/.claude/.credentials.json:ro" in args


@pytest.mark.skipif(not DOCKER_USABLE, reason="no docker daemon, or the sandbox image is not built")
def test_the_network_is_applied_after_creation_because_the_plugin_takes_no_flag():
    """Measured on the plugin at v0.6.0: NetworkMode=bridge, CapDrop=[],
    ReadonlyRootfs=false, Memory=0. It leaves an ordinary container behind
    though, and with -d nothing has run in it yet, so the flags it will not take
    are applied in the gap between creating the sandbox and exec'ing into it."""
    from heart.runner import sandbox_wrap

    quiet = sandbox_wrap(["claude"], False, "/ws", {}, mode="docker-sbx",
                         profile=_profile(_task()))[0][2]
    assert 'docker network disconnect bridge "$sbx"' in quiet
    assert "docker network connect" not in quiet, 'network "none" attaches to nothing'

    egress = sandbox_wrap(["claude"], False, "/ws", {}, mode="docker-sbx",
                          profile=_profile(_task(network="api")))[0][2]
    assert "docker network connect" in egress
    assert "docker update --memory 4g" in egress


@pytest.mark.skipif(not DOCKER_USABLE, reason="no docker daemon, or the sandbox image is not built")
def test_a_failed_network_step_is_fatal_not_a_quiet_widening():
    """A sandbox that keeps its bridge leg because the disconnect failed is the
    silent widening the feature exists to prevent; one that reaches no network
    because the connect failed produces an agent that did nothing. Exit 125 puts
    docker's message in the log, where sandbox_start_failure raises on it."""
    from heart.runner import sandbox_wrap

    script = sandbox_wrap(["claude"], False, "/ws", {}, mode="docker-sbx",
                          profile=_profile(_task(network="api")))[0][2]
    for line in script.splitlines():
        if line.startswith("docker network "):
            assert "exit 125" in line, line
    # cleanup is a trap now, not a line per handler: an early exit from any step
    # -- or the client being killed by the outer timeout -- still removes it
    assert any(l.startswith("trap ") and "docker sandbox rm" in l
               for l in script.splitlines()), script


@pytest.mark.skipif(not DOCKER_USABLE, reason="no docker daemon, or the sandbox image is not built")
def test_the_sandbox_is_removed_even_when_the_turn_fails():
    # the plugin does not take --rm; without the removal every episode leaves a
    # sandbox behind. It used to be the second-to-last line, which covered the
    # turn failing and nothing else: 103 containers from one killed batch were
    # still on the box two days later, each holding its worktree mount.
    from heart.runner import sandbox_wrap
    from heart.sandbox import WORK

    cmd, _ = sandbox_wrap(["claude"], False, "/ws", {}, mode="docker-sbx",
                          profile=_profile(_task(network="api")))
    script = cmd[2]
    assert f"docker exec -w {WORK}" in script, "else the agent runs in / and writes nothing"
    lines = script.rstrip().splitlines()
    trap = next(i for i, l in enumerate(lines) if l.startswith("trap "))
    assert "docker sandbox rm" in lines[trap]
    # every path out is covered, including SIGTERM and the timeout kill
    assert "EXIT" in lines[trap] and "INT" in lines[trap] and "TERM" in lines[trap]
    # armed before anything can fail, and after the name exists to remove
    assert lines[trap - 1].startswith("sbx=$(")
    assert "rc=$?" in lines[-2]
    assert lines[-1] == "exit $rc"


def test_every_caller_gates_on_the_same_mode_name():
    """The mode name was spelled out in four places and drifted twice, both
    times leaving `heart check-task` raising "no verifier sandbox profile was
    supplied" -- the caller tested for the old name, built no profile, and
    run_verifiers refused. One constant, checked here so a rename cannot do it
    a third time."""
    import re

    from heart.runner import SANDBOX_MODE

    root = Path(__file__).resolve().parent.parent / "src" / "heart"
    for path in ("verify.py", "episode.py", "runner.py"):
        text = (root / path).read_text()
        stray = [m for m in re.findall(r'"(docker(?:-sbx)?)"', text)
                 if m != SANDBOX_MODE or "SANDBOX_MODE" not in text]
        assert SANDBOX_MODE not in stray, (
            f"{path} hardcodes the mode name; use runner.SANDBOX_MODE")


def test_a_multiline_value_survives_the_plugin(monkeypatch):
    """`docker sandbox run` truncates a -e value at the first newline and says
    nothing. Measured: a three-line HEART_PROMPT arrived as its first line,
    which silently dropped the scope note and the retrieved packet off every
    prompt. `docker run` had no such limit, so nothing caught it until an agent
    was asked to echo what it had been given."""
    from heart.sandbox import _B64_SUFFIX

    args = " ".join(_profile(_task(network="api")).docker_sbx_args(
        {"HEART_PROMPT": "one\ntwo", "PLAIN": "single"}))
    assert f"HEART_PROMPT{_B64_SUFFIX}=" in args
    assert "-e HEART_PROMPT=one" not in args, "the truncated form must not be sent"
    assert "PLAIN=single" in args, "a value with no newline needs no encoding"


@pytest.mark.skipif(not DOCKER_USABLE, reason="no docker daemon, or the sandbox image is not built")
def test_the_decoder_runs_before_the_agent_sees_the_environment():
    from heart.runner import sandbox_wrap
    from heart.sandbox import _B64_SUFFIX

    cmd, _ = sandbox_wrap("run", True, "/ws", {"HEART_PROMPT": "a\nb"},
                          mode="docker-sbx", profile=_profile(_task(network="api")))
    script = cmd[2]
    exec_line = next(ln for ln in script.splitlines() if "docker exec" in ln)
    assert _B64_SUFFIX in exec_line and "base64 -d" in exec_line
    assert exec_line.index("base64 -d") < exec_line.rindex("run"), \
        "decoding has to happen before the agent command"


def test_the_situation_is_the_tasks_own_words():
    """It used to be `[role] <prompt> | skills: ... | touching: ...`, written for
    a human reading the episode record and handed to a retriever that treats it
    as natural language. Capillaries runs INTENT_KEYWORDS over the whole string
    and those hints are eligibility, not a boost -- so `[implement]` matched
    "build", `[review]` matched "analyze", and `[solo]` matched nothing, which
    silently narrowed which prompts could come back at all."""
    from heart.episode import _situation

    task = _task(prompt="verify my understanding of this code",
                 skills=["coding"], allowed_paths=["src"], difficulty="hard")
    assert _situation(task) == "verify my understanding of this code"
    for leak in ("[", "skills:", "touching:", "difficulty:"):
        assert leak not in _situation(task), \
            f"heart's own vocabulary must not reach the corpus query: {leak}"


# --- retrieval reaches the agent -------------------------------------------
# The corpus half lives in arteries now: it owns the gate ("calling capillaries
# again is wasted work" when the turn is already covered), capillaries owns no
# gate by its own account, and the direction is capillaries -> arteries -> heart.
# Heart asks for one packet and never learns capillaries exists.


def test_heart_does_not_reach_around_arteries_to_the_corpus():
    import pathlib as _p

    src = (_p.Path(__file__).resolve().parent.parent / "src" / "heart").rglob("*.py")
    for path in src:
        text = path.read_text()
        assert "agent/route" not in text and "agent/feedback" not in text, \
            f"{path.name} calls capillaries directly; the gate is arteries'"


def test_a_retrieved_packet_is_put_in_front_of_the_agent():
    """/context was mounted read-only and filled for three turns before anyone
    noticed the agent never read it: an `api:` agent does not go looking, so
    retrieval was paid for and thrown away."""
    from heart.episode import _retrieved_note

    note = _retrieved_note({"text": "REMEMBERED FACT\n## Suggested Approach\nSTEPS"})
    assert "REMEMBERED FACT" in note
    assert "STEPS" in note, "the corpus suggestion arrives merged into the packet"
    assert "background, not instructions" in note, \
        "retrieved text must not outrank the task"


def test_nothing_is_appended_when_nothing_was_retrieved():
    from heart.episode import _retrieved_note

    assert _retrieved_note({"status": "empty"}) == ""
    assert _retrieved_note({"status": "skipped", "text": ""}) == ""








# --- egress ---------------------------------------------------------------


def _proxy_module():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "contrib" / "egress-proxy.py"
    spec = importlib.util.spec_from_file_location("egress_proxy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_allowlist_matches_on_boundaries_not_substrings(monkeypatch):
    """`anthropic.com.evil.net` ends with the allowed name and must not pass;
    an allowlist that can be suffixed is not an allowlist."""
    monkeypatch.setenv("ALLOW", "api.anthropic.com,anthropic.com")
    permitted = _proxy_module().permitted
    assert permitted("api.anthropic.com", 443)
    assert permitted("API.Anthropic.COM", 443)      # the client chooses the case
    assert permitted("statsig.anthropic.com", 443)  # a bare name covers subdomains
    assert permitted("api.anthropic.com", 8443)     # and a bare name, any port
    assert not permitted("anthropic.com.evil.net", 443)
    assert not permitted("notanthropic.com", 443)
    assert not permitted("evil.com", 443)


def test_a_port_in_an_entry_is_the_whole_point_of_listing_the_host_alias(monkeypatch):
    """host.docker.internal is on the list so an agent can call the model server
    on :8001. Without a port it also hands the agent the host's Postgres on 5432
    and heart's own server on 8000 -- a proxy defeating its own purpose."""
    monkeypatch.setenv("ALLOW", "host.docker.internal:8001")
    permitted = _proxy_module().permitted
    assert permitted("host.docker.internal", 8001)
    assert not permitted("host.docker.internal", 5432)
    assert not permitted("host.docker.internal", 8000)


def test_the_web_lane_reaches_public_names_and_nothing_that_skips_the_filter(monkeypatch):
    """`*` is any public host on a web port. An IP literal skips DNS, which is
    where the malware filter lives, and a port other than 80/443 is how a
    'web' lane turns into an SSH or database client."""
    monkeypatch.setenv("ALLOW", "*,host.docker.internal:8001")
    monkeypatch.setenv("DENY", "pastebin.com")
    permitted = _proxy_module().permitted
    assert permitted("docs.python.org", 443)
    assert permitted("example.com", 80)
    assert not permitted("example.com", 22)
    assert not permitted("1.1.1.1", 443)
    assert not permitted("::1", 443)
    assert not permitted("pastebin.com", 443)
    assert not permitted("x.pastebin.com", 443)
    # a named entry still wins, which is how the local model stays reachable
    assert permitted("host.docker.internal", 8001)


def test_public_means_the_address_not_the_name():
    """Names are resolved and every address must be public. The tailnet, the
    LAN, loopback and the Docker Desktop VM are the places a web lane must not
    become a route into."""
    public = _proxy_module()._public
    assert public("1.1.1.1")
    for private in ("127.0.0.1", "10.10.10.1", "192.168.65.254",
                    "100.87.230.112", "169.254.169.254", "::1", "fd7a:115c:a1e0::1"):
        assert not public(private), private


def test_an_injected_host_has_no_route_but_the_injector(monkeypatch, tmp_path):
    """A foreign credential must have nowhere to go. If CONNECT still reached
    api.anthropic.com, an agent could skip the injector and use an attacker's
    key directly -- the exfiltration path the injector exists to close. Only a
    seat the proxy holds loses its route: codex on a mounted file keeps
    chatgpt.com until its seat is injected too."""
    monkeypatch.setenv("ALLOW", "*,api.anthropic.com:443")
    monkeypatch.setenv("INJECT_PORT", "8889")
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path))
    (tmp_path / "anthropic").write_text("sk-ant-oat01-real")
    permitted = _proxy_module().permitted
    assert not permitted("api.anthropic.com", 443)
    assert permitted("statsig.anthropic.com", 443)
    assert permitted("chatgpt.com", 443), "no codex secret here, so codex keeps its route"


def test_codex_gets_a_sentinel_auth_file_and_the_proxy_as_its_server(monkeypatch, tmp_path):
    """Codex reads claims from its tokens before sending, so the stand-in is a
    parseable JWT; last_refresh is now, so Codex never reaches for a refresh."""
    import base64, json
    import heart.sandbox as sb
    import heart.runner as runner

    monkeypatch.setenv("HEART_WS_ROOT", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "heart" / "secrets").mkdir(parents=True)
    (tmp_path / "heart" / "secrets" / "sentinel").write_text("s33d")
    monkeypatch.setenv("HEART_SANDBOX_INJECT", "chatgpt")
    monkeypatch.setenv("HEART_SANDBOX_CODEX_PLAN", "pro")
    (mount,) = sb.codex_sentinel_mounts()
    assert mount.target == f"{HOME}/.codex/auth.json" and not mount.writable
    doc = json.loads(Path(mount.source).read_text())
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path / "heart" / "secrets"))
    assert doc["tokens"]["access_token"] == sb.sentinels("s33d")["chatgpt"] \
        == _proxy_module().sentinels()["chatgpt"]
    payload = doc["tokens"]["id_token"].split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    assert claims["https://api.openai.com/auth"]["chatgpt_plan_type"] == "pro"

    env = sb.inject_env("egress-web")
    assert env["HEART_CODEX_BASE"] == "http://egress-web:8889/chatgpt/backend-api"
    assert env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"].startswith("http://egress-web:8889/")
    assert "ANTHROPIC_BASE_URL" not in env

    class Wrapped(Exception):
        pass

    def stop(cmd, *a, **k):  # the command as it would enter the container
        raise Wrapped(cmd)

    monkeypatch.setattr(runner, "sandbox_wrap", stop)
    profile = sb.replace(_profile(_task(network="api")), env={"HEART_CODEX_BASE": env["HEART_CODEX_BASE"]})
    with pytest.raises(Wrapped) as got:
        runner.run_agent("codex", "do it", str(tmp_path), {}, 10, tmp_path / "log", profile=profile)
    cmd = got.value.args[0]
    assert cmd[:2] == ["codex", "exec"]
    assert 'openai_base_url="http://egress-web:8889/chatgpt/backend-api/codex"' in cmd
    assert 'chatgpt_base_url="http://egress-web:8889/chatgpt/backend-api/"' in cmd
    assert cmd[-1] == "do it"


def test_the_chatgpt_route_swaps_token_and_account(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path))
    proxy = _proxy_module()
    monkeypatch.setitem(proxy._SECRET_FILES, "chatgpt", str(tmp_path / "auth.json"))
    (tmp_path / "auth.json").write_text(
        '{"tokens": {"access_token": "real-access", "account_id": "real-acct"}}')
    assert proxy.secret("chatgpt") == "real-access" and proxy.account("chatgpt") == "real-acct"
    (tmp_path / "sentinel").write_text("s33d")
    own = proxy.sentinels()
    assert proxy.sentinel_only([("Authorization", f"Bearer {own['chatgpt']}")], own["chatgpt"])
    assert not proxy.sentinel_only([("Authorization", f"Bearer {own['anthropic']}")],
                                   own["chatgpt"]), "each route takes only its own"
    out = proxy.rewrite("POST", "/backend-api/codex/responses",
                        [("chatgpt-account-id", "heart-sentinel"),
                         ("Authorization", f"Bearer {own['chatgpt']}")],
                        b"{}", "chatgpt.com", "real-access", "real-acct").decode().lower()
    assert "authorization: bearer real-access" in out
    assert "chatgpt-account-id: real-acct" in out and "heart-sentinel" not in out


def test_the_upload_budget_is_per_client_and_refills(monkeypatch):
    """Browsing is small going out; a repo is not. The budget stops bulk
    uploads to web hosts per client and comes back after the window."""
    monkeypatch.setenv("UPLOAD_BUDGET", "1000")
    monkeypatch.setenv("UPLOAD_WINDOW", "600")
    proxy = _proxy_module()
    clock = [100.0]
    monkeypatch.setattr(proxy.time, "monotonic", lambda: clock[0])
    assert proxy.charge("10.0.0.2", 600)
    assert not proxy.charge("10.0.0.2", 600), "over budget"
    assert proxy.charge("10.0.0.3", 600), "another container has its own"
    assert not proxy.charge("10.0.0.2", 0), "spent stays spent inside the window"
    clock[0] += 601
    assert proxy.charge("10.0.0.2", 600), "and refills after it"


def _seeded_proxy(monkeypatch, tmp_path, seed="s33d"):
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path))
    (tmp_path / "sentinel").write_text(seed)
    return _proxy_module()


def test_the_injector_takes_the_sentinel_and_nothing_else(monkeypatch, tmp_path):
    proxy = _seeded_proxy(monkeypatch, tmp_path)
    s = proxy.sentinels()["anthropic"]
    assert proxy.sentinel_only([("Authorization", f"Bearer {s}")], s)
    assert proxy.sentinel_only([("x-api-key", s), ("content-type", "application/json")], s)
    assert not proxy.sentinel_only([], s)                           # no credential at all
    assert not proxy.sentinel_only([("x-api-key", "sk-ant-api03-attacker")], s)
    # the old published constant is just another foreign credential now
    assert not proxy.sentinel_only([("x-api-key", "sk-ant-oat01-heart-sentinel")], s)
    # the sentinel does not launder a second, foreign credential
    assert not proxy.sentinel_only([("Authorization", f"Bearer {s}"),
                                    ("x-api-key", "sk-ant-api03-attacker")], s)


def test_the_forwarded_request_carries_the_real_credential_once_and_closes(monkeypatch, tmp_path):
    proxy = _seeded_proxy(monkeypatch, tmp_path)
    s = proxy.sentinels()["anthropic"]
    out = proxy.rewrite("POST", "/v1/messages",
                        [("Host", "egress:8889"), ("Authorization", f"Bearer {s}"),
                         ("anthropic-beta", "oauth-2025-04-20"), ("Connection", "keep-alive"),
                         ("Transfer-Encoding", "chunked")],
                        b'{"x":1}', "api.anthropic.com", "sk-ant-oat01-REAL").decode()
    head = out.split("\r\n\r\n")[0].lower()
    assert out.startswith("POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\n")
    assert "authorization: bearer sk-ant-oat01-real" in head
    assert s not in out
    assert "anthropic-beta: oauth-2025-04-20" in head
    assert "connection: close" in head and "keep-alive" not in head
    assert "transfer-encoding" not in head and "content-length: 7" in head
    assert "cookie" not in proxy.rewrite(
        "GET", "/", [("Cookie", "session=attacker")], b"", "chatgpt.com", "t").decode().lower(), \
        "a session cookie is a credential and never rides along"
    assert out.endswith('{"x":1}')
    # an API key goes where the API expects one
    assert "x-api-key: sk-ant-api03-k" in proxy.rewrite(
        "GET", "/", [], b"", "api.anthropic.com", "sk-ant-api03-k").decode()


def test_the_secret_is_the_bare_token_or_a_credentials_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path))
    proxy = _proxy_module()
    (tmp_path / "anthropic").write_text("sk-ant-oat01-abc\n")
    assert proxy.secret("anthropic") == "sk-ant-oat01-abc"
    (tmp_path / "anthropic").write_text('{"claudeAiOauth": {"accessToken": "tok"}}')
    assert proxy.secret("anthropic") == "tok"
    assert proxy.secret("missing") == ""


def test_an_allowlist_refusal_is_not_an_agent_that_did_nothing():
    """A denied host reaches the agent as an ordinary API error, so the run ends
    with no diff and the ladder reads `no_change` at reward 0.0. Measured
    against a narrow allowlist before this existed: the episode scored zero for
    never reaching a model."""
    from heart.runner import EGRESS_DENIED_MARKER, sandbox_egress_denied

    envelope = ('{"is_error":true,"result":"API Error: 403 '
                f'{EGRESS_DENIED_MARKER} api.anthropic.com is not in the '
                'sandbox allowlist","usage":{"input_tokens":0}}')
    found = sandbox_egress_denied(envelope)
    assert found and "api.anthropic.com" in found
    assert len(found) < 200, "the envelope is thousands of chars; the host is the point"
    assert sandbox_egress_denied("FAILED tests/test_x.py - assert 1 == 2") is None


def test_the_proxy_and_heart_agree_on_the_marker():
    # a phrase either side could reword is a contract that drifts; both read the
    # same token
    from heart.runner import EGRESS_DENIED_MARKER

    assert _proxy_module().DENIED_MARKER == EGRESS_DENIED_MARKER


def test_a_proxy_is_offered_in_both_spellings(monkeypatch):
    """curl and python read the lowercase names, most node CLIs the uppercase.
    A proxy honored by half an image is worse than none: the half that ignores
    it has no gateway and fails with no explanation."""
    monkeypatch.setenv("HEART_SANDBOX_PROXY", "http://egress:8888")
    env = _profile(_task(network="api")).env
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env[name] == "http://egress:8888"
    # and a verifier, which has no network at all, is not told about one
    assert "HTTPS_PROXY" not in _verifier(_task()).env


def test_no_proxy_variables_when_none_is_configured(monkeypatch):
    import heart.sandbox as sb

    monkeypatch.delenv("HEART_SANDBOX_PROXY", raising=False)
    # a routable network, stated rather than read off whatever box runs this --
    # on one with the proxy provisioned the live lookup finds it
    monkeypatch.setattr(sb, "network_facts", lambda n: (False, ()))
    assert "HTTPS_PROXY" not in _profile(_task(network="api")).env


# --- a container that never started is not a verdict --------------------


def test_a_container_that_never_started_is_not_an_agent_that_did_nothing():
    """Docker exits 125 before running anything when the image or the network
    is missing, or a bind is unshared. Read as a verdict, a typo in
    HEART_SANDBOX_IMAGE produces a whole batch of empty diffs at reward 0.0 --
    marrow learning the model can do nothing, from runs that never happened."""
    from heart.runner import sandbox_start_failure

    msg = sandbox_start_failure(125, (
        "Unable to find image 'heart-agent:nope' locally\n"
        "docker: Error response from daemon: pull access denied for heart-agent"))
    assert msg and "pull access denied" in msg


def test_a_verifier_that_chooses_to_exit_125_is_still_a_verdict():
    # 125 alone means nothing: the exit code belongs to whatever ran, and only
    # docker's own message says docker refused
    from heart.runner import sandbox_start_failure

    assert sandbox_start_failure(125, "FAILED tests/test_x.py::test_y") is None
    assert sandbox_start_failure(1, "docker: Error response from daemon: x") is None


def test_an_unscoped_task_gets_no_scope_note():
    from heart.episode import _scope_note

    assert _scope_note(_task()) == ""


def test_the_agent_is_told_the_boundary_it_will_otherwise_discover_by_erofs():
    """The scope is authored by someone who has not read the code -- in the
    decomposed path, by a planner working from a prompt alone. The agent is the
    only party that finds out where that guess was wrong, and it cannot report
    what it was never told."""
    from heart.episode import _scope_note

    note = _scope_note(_task(allowed_paths=["src"], denied_paths=["secrets"]))
    assert "src" in note and "secrets" in note


def test_the_block_channel_is_offered_only_when_the_caller_named_one():
    """heart supplies the mechanism, the caller owns the vocabulary -- inventing
    a marker the caller does not parse would produce blocks nobody reads."""
    from heart.episode import _scope_note

    scoped = {"allowed_paths": ["src"]}
    assert "PLEXUS_BLOCKED:" not in _scope_note(_task(**scoped))
    assert "PLEXUS_BLOCKED:" in _scope_note(
        _task(**scoped, blocked_marker="PLEXUS_BLOCKED:"))


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
    host's endpoint connects to itself and reports the model as down."""
    from heart.sandbox import container_endpoint

    assert container_endpoint("http://127.0.0.1:8001/v1") == \
        "http://host.docker.internal:8001/v1"
    # the port survives: one llama-server per GPU on adjacent ports is the
    # normal arrangement, and a constant would send every profile to one of them
    assert container_endpoint("http://localhost:9999/v1").endswith(":9999/v1")
    assert container_endpoint("https://api.anthropic.com/v1") == \
        "https://api.anthropic.com/v1"


def test_the_profile_does_not_pin_an_endpoint_of_its_own():
    """Profile env outranks the caller's, so an endpoint pinned here silently
    beat the one api_agent_env resolves from the model profile -- every task
    sent to one hardcoded host:port whatever its profile named."""
    for task in (_task(network="model"), _task()):
        assert "HEART_API_ENDPOINT" not in profile_for(task, "/ws", "/c", "/i").env


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


def test_codex_does_not_build_a_sandbox_inside_heart_s(monkeypatch):
    """codex enforces `-s workspace-write` with bubblewrap, and bwrap cannot
    create a user namespace inside an unprivileged container: every command the
    agent tried came back `bwrap: No permissions to create a new namespace`, it
    exited 0 having changed nothing, and the episode scored `no_change` at 0.0.
    A nesting problem recorded as a model failure. Measured on
    20260915-111156-ddcc4b7d."""
    from heart.runner import _agent_command

    monkeypatch.setenv("HEART_SANDBOX", "docker-sbx")
    cmd, _ = _agent_command("codex:luna", "fix it")
    assert "danger-full-access" in cmd
    assert "workspace-write" not in cmd

    # outside heart's sandbox the inner one is the only one there is
    monkeypatch.setenv("HEART_SANDBOX", "off")
    cmd, _ = _agent_command("codex:luna", "fix it")
    assert "workspace-write" in cmd


def test_unnesting_leaves_other_agents_alone(monkeypatch):
    from heart.runner import _agent_command

    monkeypatch.setenv("HEART_SANDBOX", "docker-sbx")
    cmd, _ = _agent_command("claude:haiku", "fix it")
    assert cmd[:2] == ["claude", "-p"]
    assert "danger-full-access" not in cmd


# --- seats by injection, readers, contained commands ----------------------


def test_the_sentinel_is_one_value_on_both_sides(monkeypatch, tmp_path):
    from heart.sandbox import sentinels

    assert _seeded_proxy(monkeypatch, tmp_path, "abc").sentinels() == sentinels("abc")


def test_a_run_without_the_seed_cannot_use_the_injector(monkeypatch, tmp_path):
    """The seed is per box and handed only to runs that were given a seat, so
    a run with seats withheld can reach the injector but has nothing it will
    take. And a proxy with no seed accepts nothing at all."""
    import heart.sandbox as sb

    assert sb.sentinels("a") != sb.sentinels("b")
    assert _seeded_proxy(monkeypatch, tmp_path).sentinels()["anthropic"] != \
        "sk-ant-oat01-heart-sentinel", "not the constant this file used to publish"
    (tmp_path / "sentinel").unlink()
    assert _proxy_module().sentinels() == {}
    # injection asked for with no seed fails loudly, not as "not logged in"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    monkeypatch.setenv("HEART_SANDBOX_INJECT", "anthropic")
    with pytest.raises(RuntimeError, match="plexus doctor --fix"):
        sb.inject_env("egress")


def test_an_injected_seat_is_a_sentinel_and_a_base_url_never_a_token(monkeypatch):
    """With the seat injected, the container holds nothing worth stealing: a
    sentinel, and the address of the proxy that will swap it."""
    import heart.sandbox as sb

    monkeypatch.setattr(sb, "network_facts", lambda n: (True, ("egress-web", "heart-abc")))
    monkeypatch.delenv("HEART_SANDBOX_PROXY", raising=False)
    monkeypatch.setattr(sb, "sentinel_seed", lambda: "s33d")
    monkeypatch.setenv("HEART_SANDBOX_INJECT", "anthropic")
    env = sb.proxy_env("heart-web")
    assert env["HTTPS_PROXY"] == "http://egress-web:8888"
    assert env["ANTHROPIC_BASE_URL"] == "http://egress-web:8889/anthropic"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == sb.sentinels("s33d")["anthropic"]
    # the base URL is plain http on the internal network; through HTTP_PROXY it
    # would be an absolute URI for a host no allowlist names
    assert env["NO_PROXY"] == env["no_proxy"] == "egress-web"

    monkeypatch.delenv("HEART_SANDBOX_INJECT")
    assert "ANTHROPIC_BASE_URL" not in sb.proxy_env("heart-web")


def test_each_lane_finds_its_own_proxy_among_the_agents_on_it(monkeypatch):
    import heart.sandbox as sb

    monkeypatch.delenv("HEART_SANDBOX_PROXY", raising=False)
    monkeypatch.setattr(sb, "network_facts",
                        lambda n: (True, ("heart-1", "egress-web", "heart-2")))
    assert sb.proxy_env("heart-web")["HTTPS_PROXY"] == "http://egress-web:8888"


def test_a_reader_can_read_everything_and_write_nothing_but_git_metadata(tmp_path):
    """Planner, decomposer, reviewer: an agent's reach minus a writable tree.
    The per-worktree git dir stays writable because `git diff` refreshes the
    index to answer."""
    from heart.sandbox import reader_profile_for

    repo = tmp_path / "repo"
    (repo / ".git" / "worktrees" / "a").mkdir(parents=True)
    task = _task(repo_path=str(repo), allowed_paths=["src"], network="api")
    p = reader_profile_for(task, "/ws/a", "/ctx", "/jr")
    assert _mount(p, WORK).writable is False
    assert _mount(p, f"{WORK}/src") is None, "an allowance is a write grant; readers get none"
    assert _mount(p, CONTEXT).writable is False
    assert _mount(p, str(repo / ".git")).writable is False
    assert _mount(p, str(repo / ".git" / "worktrees" / "a")).writable is True
    assert p.network != "none"


def test_a_turn_outside_an_episode_is_contained_only_when_asked(monkeypatch, tmp_path):
    import heart.runner as runner
    import heart.sandbox as sb

    monkeypatch.delenv("HEART_SANDBOX", raising=False)
    assert runner.turn_profile(_task(), "/ws/a", tmp_path, "k") is None

    monkeypatch.setenv("HEART_SANDBOX", runner.SANDBOX_MODE)
    monkeypatch.setattr(sb, "inbox_for", lambda key: tmp_path / "inbox")
    reader = runner.turn_profile(_task(), "/ws/a", tmp_path, "k")
    assert reader.network != "none", "a reader needs a model; 'none' means 'api' here"
    assert _mount(reader, WORK).writable is False
    writer = runner.turn_profile(_task(), "/ws/a", tmp_path, "k", kind="writer")
    assert _mount(writer, WORK).writable is True
    judge = runner.turn_profile(_task(network="web"), "/ws/a", None, "k", kind="verifier")
    assert judge.network == "none", "a verifier never inherits the agent's reach"


def test_a_contained_command_runs_on_the_host_when_no_sandbox_is_asked(monkeypatch, tmp_path):
    from heart.runner import run_contained

    monkeypatch.delenv("HEART_SANDBOX", raising=False)
    r = run_contained("pwd; exit 3", str(tmp_path), timeout=10)
    assert r.returncode == 3 and r.stdout.strip() == str(tmp_path)


@pytest.mark.skipif(not DOCKER_USABLE, reason="no docker daemon, or the sandbox image is not built")
def test_code_an_agent_wrote_runs_with_no_network_and_no_home(monkeypatch):
    """The acceptance check plexus runs after an episode executes whatever the
    agent put in the tree. Contained, a conftest.py that phones home has no
    route, and the operator's home is not there to read."""
    from heart.env import _ws_root
    from heart.runner import SANDBOX_MODE, run_contained
    import tempfile

    monkeypatch.setenv("HEART_SANDBOX", SANDBOX_MODE)
    _ws_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_ws_root()) as ws:
        probe = ("python3 -c \"import socket; s=socket.socket(); s.settimeout(3); "
                 "s.connect(('1.1.1.1', 443))\" 2>/dev/null && echo NET || echo NONET; "
                 f"test -e {Path.home()}/.ssh && echo HOME || echo NOHOME; "
                 "touch made 2>/dev/null && echo WROTE || echo RO")
        r = run_contained(probe, ws, timeout=90)
        assert "NONET" in r.stdout and "NOHOME" in r.stdout and "RO" in r.stdout, r.stdout + r.stderr
        r = run_contained("touch made && echo WROTE", ws, timeout=90, writable=True)
        assert "WROTE" in r.stdout, r.stdout + r.stderr


def test_a_packet_built_for_an_agent_says_whose_it_is_and_which_lane(monkeypatch, tmp_path):
    """arteries filters retrieval on these two: an agent's packet may carry its
    own project's untrusted memory, and a web-lane packet nothing from another
    project, since whatever lands in /context can leave with the agent."""
    import heart.episode as ep

    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw["env"])
        raise RuntimeError("stop here")

    monkeypatch.setattr(ep.subprocess, "run", fake_run)
    monkeypatch.delenv("ARTERIES_RETRIEVAL", raising=False)
    got = ep._context_packet(_task(network="web"), "implement", "normal", tmp_path, "ep1")
    assert got["status"] == "failed"
    assert seen["ARTERIES_TRUST"] == "untrusted" and seen["ARTERIES_LANE"] == "web"
