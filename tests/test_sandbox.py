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
    from heart.sandbox import home_file_mounts

    assert home_file_mounts() == ()


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
            assert "docker sandbox rm" in line, f"a fatal step must still clean up: {line}"


def test_the_sandbox_is_removed_even_when_the_turn_fails():
    # the plugin does not take --rm; without the explicit rm every episode
    # leaves a running sandbox behind
    from heart.runner import sandbox_wrap
    from heart.sandbox import WORK

    cmd, _ = sandbox_wrap(["claude"], False, "/ws", {}, mode="docker-sbx",
                          profile=_profile(_task(network="api")))
    script = cmd[2]
    assert f"docker exec -w {WORK}" in script, "else the agent runs in / and writes nothing"
    # the removal that matters is the one after the turn, not the ones inside
    # the fatal-step handlers -- so check the tail, not the first match
    tail = script.rstrip().splitlines()[-3:]
    assert "rc=$?" in tail[0]
    assert tail[1].startswith("docker sandbox rm")
    assert tail[2] == "exit $rc"


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
    assert permitted("api.anthropic.com")
    assert permitted("API.Anthropic.COM")       # the client chooses the case
    assert permitted("statsig.anthropic.com")   # a bare name covers subdomains
    assert not permitted("anthropic.com.evil.net")
    assert not permitted("notanthropic.com")
    assert not permitted("evil.com")


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
    monkeypatch.delenv("HEART_SANDBOX_PROXY", raising=False)
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
