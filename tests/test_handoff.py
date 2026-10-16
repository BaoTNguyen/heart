"""Cold recovery and CLI handoff with real git, transcripts, processes and PTYs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from heart import handoff, routing


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "app.py").write_text("def value():\n    return 1\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)
    return path


def conversation(path, repo, tail=True):
    records = [
        {"type": "user", "uuid": "u1", "parentUuid": None, "cwd": str(repo),
         "message": {"content": "Change value to 2, preserve my notes, and run tests."}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "cwd": str(repo),
         "message": {"content": [{"type": "text", "text": "I will preserve notes."},
                                 {"type": "thinking", "thinking": "PRIVATE_REASONING"}]}},
        {"type": "attachment", "uuid": "attach", "parentUuid": "a1", "cwd": str(repo)},
        {"type": "assistant", "uuid": "abandoned", "parentUuid": "attach", "cwd": str(repo),
         "message": {"content": "Abandoned plan: delete notes"}},
        {"type": "user", "uuid": "u2", "parentUuid": "attach", "cwd": str(repo),
         "message": {"content": "Correction: return 3, not 2. Never remove notes."}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2", "cwd": str(repo),
         "message": {"content": [{"type": "tool_use", "id": "edit", "name": "Edit", "input": {"file_path": str(repo / "app.py")}}]}},
        {"type": "user", "uuid": "r2", "parentUuid": "a2", "cwd": str(repo),
         "message": {"content": [{"type": "tool_result", "tool_use_id": "edit", "content": "edited"}]}},
        {"type": "assistant", "uuid": "a3", "parentUuid": "r2", "cwd": str(repo),
         "message": {"content": [{"type": "tool_use", "id": "test", "name": "Bash", "input": {"command": "pytest -q"}}]}},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + ('{"type":' if tail else ""))
    return path


def test_cold_recovery_preserves_partial_edits_and_uncertain_commands(repo, tmp_path):
    (repo / "app.py").write_text("def value():\n    return 3\n# partial edit\n")
    (repo / "notes.txt").write_text("user work\n")
    source = conversation(tmp_path / "session.jsonl", repo)
    before = handoff.worktree(repo)
    packet = handoff.prepare(repo, source, tmp_path / "packet")
    assert handoff.worktree(repo) == before
    assert (repo / "notes.txt").read_text() == "user work\n"
    assert packet["context"]["requests"][-1].startswith("Correction:")
    assert packet["context"]["tools"][0]["status"].startswith("result recorded")
    assert packet["context"]["tools"][1]["status"].startswith("uncertain")
    output = json.dumps(packet)
    assert "PRIVATE_REASONING" not in output
    assert "Abandoned plan" not in output
    assert (tmp_path / "packet/checkpoint.json").stat().st_mode & 0o777 == 0o600


def test_locate_uses_recorded_cwd_and_explicit_session(repo, tmp_path):
    projects = tmp_path / "projects"
    wrong = conversation(projects / "encoded-wrong/other.jsonl", tmp_path)
    right = conversation(projects / "encoded-name/session.jsonl", repo)
    os.utime(wrong, (time.time() + 100, time.time() + 100))
    assert handoff.locate(repo, projects=projects) == right
    assert handoff.locate(repo, "session", projects) == right
    with pytest.raises(ValueError, match="No Claude transcript"):
        handoff.locate(repo, "missing", projects)
    with pytest.raises(ValueError, match="working directory"):
        handoff.transcript_state(wrong, repo)


@pytest.mark.parametrize("change", ["tracked", "untracked", "source"])
def test_stale_packet_refuses_release(repo, tmp_path, change):
    (repo / "notes.txt").write_text("before")
    source = conversation(tmp_path / "session.jsonl", repo)
    packet = handoff.prepare(repo, source, tmp_path / "packet")
    handoff.assert_fresh(packet, repo)
    if change == "tracked":
        (repo / "app.py").write_text("new edit")
    elif change == "untracked":
        (repo / "notes.txt").write_text("after")
    else:
        with source.open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        handoff.assert_fresh(packet, repo)


def test_duplicate_owner_is_rejected_and_lock_releases(repo):
    with handoff.ownership(repo):
        with pytest.raises(ValueError, match="already owns"):
            with handoff.ownership(repo):
                pytest.fail("second owner")
    with handoff.ownership(repo):
        pass


def fake_codex(tmp_path, monkeypatch, mode="ok"):
    binary = tmp_path / "bin/codex"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text("#!/usr/bin/env python3\n" + r'''
import json, os, pathlib, re, sys
args = sys.argv[1:]
if args[0] == 'resume':
    assert args[args.index('-s') + 1] == 'workspace-write'
    assert 'approval_policy="on-request"' in args
    assert 'codex-thread' in args
    assert pathlib.Path('app.py').read_text().endswith('# partial edit\n')
    assert pathlib.Path('notes.txt').read_text() == 'user work\n'
    pathlib.Path('app.py').write_text('def value():\n    return 3\n')
    pathlib.Path('continued.json').write_text(json.dumps(dict(thread='codex-thread', preserved=True)))
    sys.exit(0)
assert args[0] == 'exec'
assert args[args.index('-s') + 1] == 'read-only'
assert 'approval_policy="never"' in args
prompt = sys.stdin.read()
path = pathlib.Path(re.search(r'Checkpoint: (.+)', prompt).group(1))
packet = json.loads(path.read_text())
assert packet['context']['requests'][-1].startswith('Correction:')
ack = dict(packet_id=packet['packet_id'], task_id=packet['task_id'],
           repo=packet['worktree']['repo'], next_action='Inspect interrupted test, then complete partial edit')
mode = os.environ.get('FAKE_CODEX_MODE')
if mode == 'fail': sys.exit(1)
if mode == 'mismatch': ack['packet_id'] = 'wrong'
if mode == 'edit': pathlib.Path('app.py').write_text('changed by preflight')
pathlib.Path(args[args.index('-o') + 1]).write_text(json.dumps(ack))
if mode != 'no-thread': print(json.dumps(dict(type='thread.started', thread_id='codex-thread')))
''')
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)


def test_target_receives_checkpoint_in_read_only_preflight(repo, tmp_path, monkeypatch):
    fake_codex(tmp_path, monkeypatch)
    source = conversation(tmp_path / "session.jsonl", repo)
    directory = tmp_path / "packet"
    packet = handoff.prepare(repo, source, directory)
    before = handoff.worktree(repo)
    assert handoff.acknowledge(packet, directory, repo) == "codex-thread"
    assert handoff.worktree(repo) == before
    assert json.loads((directory / "transfer.json").read_text())["state"] == "acknowledged"


@pytest.mark.parametrize("mode,match", [("fail", "did not acknowledge"), ("mismatch", "does not match"),
                                      ("no-thread", "thread ID"), ("edit", "Worktree changed")])
def test_failed_target_never_releases_for_editing(repo, tmp_path, monkeypatch, mode, match):
    fake_codex(tmp_path, monkeypatch, mode)
    directory = tmp_path / "packet"
    packet = handoff.prepare(repo, conversation(tmp_path / "session.jsonl", repo), directory)
    with pytest.raises(ValueError, match=match):
        handoff.acknowledge(packet, directory, repo)
    assert not (directory / "transfer.json").exists()


def test_hook_only_marks_typed_main_session_rate_limits(tmp_path):
    directory = tmp_path / "packet"
    handoff.atomic_json(directory / "source.json", {"session_id": "main"})
    base = {"session_id": "main", "hook_event_name": "StopFailure", "error": "rate_limit"}
    handoff.hook_event(directory, dict(base, agent_id="subagent"))
    handoff.hook_event(directory, dict(base, session_id="other"))
    handoff.hook_event(directory, dict(base, error="authentication_failed"))
    handoff.hook_event(directory, dict(base, hook_event_name="PostToolUse"))
    assert not (directory / "exhausted.json").exists()
    handoff.hook_event(directory, base)
    assert (directory / "exhausted.json").exists()
    handoff.hook_event(directory, dict(base, hook_event_name="Stop"))
    assert (directory / "exhausted.json").exists()


def test_terminal_forwarding_and_source_stops_before_return(repo, tmp_path):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with (tmp_path / "terminal.log").open("wb") as log:
        code, stopped = handoff.terminal([sys.executable, "-c", "print('native terminal')"], repo,
                                         input_fd=read_fd, output_fd=log.fileno())
    os.close(read_fd)
    assert (code, stopped) == (0, False)
    assert "native terminal" in (tmp_path / "terminal.log").read_text()
    marker = tmp_path / "pid"
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with (tmp_path / "terminal.log").open("wb") as log:
        code, stopped = handoff.terminal(
            [sys.executable, "-c", f"import os,time,pathlib; pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"],
            repo, tick=marker.exists, input_fd=read_fd, output_fd=log.fileno())
    os.close(read_fd)
    assert stopped
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_no_available_model_never_forces_exhausted_provider(monkeypatch):
    monkeypatch.setattr(routing, "emit", lambda *a, **k: None)
    manifest = {"strong": {"agent": "claude", "context": 3000, "max_difficulty": "hard", "skills": {}, "cost": 1},
                "weak": {"agent": "codex", "context": 3000, "max_difficulty": "easy", "skills": {}, "cost": 1}}
    want = routing.Demand(task_id="t", tier="frontier", effort="high", skills=["coding"], difficulty="expert", min_context=2000)
    with pytest.raises(ValueError, match="no available model"):
        routing.pick(want, manifest, available=lambda _: False)
    choice = routing.pick(want, manifest, available=lambda agent: agent == "codex")
    assert choice.agent == "codex"
    with pytest.raises(ValueError, match="context constraints"):
        routing.pick(want, {name: dict(m, context=1000) for name, m in manifest.items()})


def test_supervised_claude_partial_edit_to_codex_end_to_end(repo, tmp_path, monkeypatch):
    fake_codex(tmp_path, monkeypatch)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.setenv("PYTHONPATH", str(Path(handoff.__file__).parents[1]))
    binary = tmp_path / "bin/claude"
    binary.write_text("#!/usr/bin/env python3\n" + '''
import json, os, pathlib, shlex, subprocess, sys, time
args = sys.argv[1:]
session = args[args.index('--session-id') + 1]
settings = json.loads(pathlib.Path(args[args.index('--settings') + 1]).read_text())
root = pathlib.Path.cwd()
source = pathlib.Path(os.environ['CLAUDE_CONFIG_DIR']) / 'projects' / 'encoded' / (session + '.jsonl')
source.parent.mkdir(parents=True)
def row(kind, id, parent, content):
    return dict(type=kind, uuid=id, parentUuid=parent, cwd=str(root), message=dict(content=content))
records = [row('user', 'u1', None, 'Change value to 2, preserve notes.'),
           row('user', 'u2', 'u1', 'Correction: return 3, not 2. Preserve notes.'),
           row('assistant', 'a2', 'u2', [dict(type='tool_use', id='edit', name='Edit', input=dict(file_path=str(root/'app.py')))])]
source.write_text(''.join(json.dumps(r)+'\\n' for r in records))
(root/'app.py').write_text('def value():\\n    return 3\\n# partial edit\\n')
(root/'notes.txt').write_text('user work\\n')
for name in ['PostToolUse', 'StopFailure']:
    event = dict(session_id=session, transcript_path=str(source), cwd=str(root), hook_event_name=name)
    if name == 'StopFailure': event['error'] = 'rate_limit'
    command = settings['hooks'][name][0]['hooks'][0]['command']
    subprocess.run(shlex.split(command), input=json.dumps(event), text=True, check=True)
time.sleep(30)
''')
    binary.chmod(0o755)
    args = type("Args", (), {"command": "session", "repo": str(repo),
                            "directory": str(tmp_path / "handoff"), "prompt": "Do the task", "resume": None})()
    assert handoff.run(args) == 0
    assert (repo / "app.py").read_text() == "def value():\n    return 3\n"
    assert (repo / "notes.txt").read_text() == "user work\n"
    assert json.loads((repo / "continued.json").read_text())["preserved"]
    packet = json.loads((tmp_path / "handoff/checkpoint.json").read_text())
    assert packet["context"]["tools"][0]["status"].startswith("uncertain")
    assert packet["context"]["requests"][-1].startswith("Correction:")


def test_unavailable_tier_default_is_not_selected(monkeypatch):
    monkeypatch.setattr(routing, "resolve", lambda *a, **k: "claude")
    want = routing.Demand(tier="mid", effort="medium", skills=["coding"], difficulty="easy", min_context=1000)
    with pytest.raises(ValueError, match="tier default"):
        routing.pick(want, {}, available=lambda _: False)


def test_cli_registers_recovery_and_supervisor(capsys):
    from heart.cli import main
    for command in ("handoff", "session"):
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0
        assert "--repo" in capsys.readouterr().out


def test_plan_mode_stays_read_only_on_target(repo, tmp_path, monkeypatch):
    source = conversation(tmp_path / "session.jsonl", repo, tail=False)
    with source.open("a") as stream:
        stream.write(json.dumps({"type": "permission-mode", "permissionMode": "plan"}) + "\n")
    packet = handoff.prepare(repo, source, tmp_path / "packet")
    calls = []
    monkeypatch.setattr(handoff, "active_claude", lambda repo: [])
    monkeypatch.setattr(handoff, "acknowledge", lambda *a: "target-thread")
    monkeypatch.setattr(handoff, "terminal", lambda argv, *a, **k: (calls.append(argv) or 0, False))
    assert handoff.release(packet, tmp_path / "packet", repo) == 0
    assert calls[0][calls[0].index("-s") + 1] == "read-only"


def test_live_source_blocks_replacement(repo, tmp_path, monkeypatch):
    source = conversation(tmp_path / "session.jsonl", repo)
    packet = handoff.prepare(repo, source, tmp_path / "packet")
    monkeypatch.setattr(handoff, "active_claude", lambda repo: [1234])
    monkeypatch.setattr(handoff, "acknowledge", lambda *a: pytest.fail("must not launch target"))
    with pytest.raises(ValueError, match="still running"):
        handoff.release(packet, tmp_path / "packet", repo)


@pytest.mark.parametrize("result", [{"interrupted": True}, {"backgroundTaskId": "job-1"}])
def test_recorded_result_does_not_hide_interrupted_or_background_work(repo, tmp_path, result):
    source = conversation(tmp_path / "session.jsonl", repo, tail=False)
    with source.open("a") as stream:
        stream.write(json.dumps({"type": "user", "uuid": "r3", "parentUuid": "a3", "cwd": str(repo),
                                "toolUseResult": result, "message": {"content": [
                                    {"type": "tool_result", "tool_use_id": "test", "content": "not done"}]}}) + "\n")
    state = handoff.transcript_state(source, repo)
    assert state["tools"][-1]["status"].startswith("uncertain")
