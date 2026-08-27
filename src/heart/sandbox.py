"""Least-privilege container profile derived from a TaskSpec.

The spec already says what a task is allowed to touch. Today `path_violations`
reads that back off the diff *after* the agent has finished, which detects a
breach it could have prevented. This module turns the same fields into the
mount table: a path the task may not write is mounted read-only, so writing it
fails at the syscall instead of at the review.

Nothing here is baked into an image. Every restriction is a `docker run` flag,
so two tasks in the same episode image can get genuinely different privileges.

The profile is deliberately shaped as data. `docker_sbx_args()` is the only
place that knows the runtime exists, which is what keeps a second one -- gVisor,
a microVM, a plain worktree -- from being a rewrite. It has already paid for
itself once: swapping `docker run` for the docker-sbx plugin was a second
renderer on this same profile, not a rewrite.
"""
from __future__ import annotations

import base64
import os
import shutil
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .taskspec import TaskSpec

WORK = "/work"
CONTEXT = "/context"
JOURNAL = "/journal"

# Where the agent CLIs keep their state -- config, caches, the session log.
# A real directory created by the Dockerfile and owned by the agent uid, not a
# tmpfs: this runtime has no --tmpfs, and a credential mounted beneath a
# directory docker invents lands root-owned and unwritable.
HOME = "/home/agent"

# Built from this repo's Dockerfile: git, node, heart, and no agent CLI at all
# (those are mounted from the host -- see agent_tool_mounts).
#
# Not `docker sandbox run`, and not its docker/sandbox-templates image. That
# plugin runs an agent in a container it builds itself, and measured against
# the running one it gives bridge networking with no way to change it, no
# cap-drop, no read-only rootfs, no resource limits, passwordless sudo, and a
# TTY it will not run without. It also accepts only `claude` and `gemini`,
# where heart also has `api:` and shell-template agents. Every one of those is
# something the profile below expresses, so heart drives `docker run` directly
# and the plugin stays unused.
DEFAULT_IMAGE = "heart-agent:latest"

#: Marks an environment variable whose value was base64'd past the plugin's
#: newline truncation. Stripped back to the real name inside the container.
_B64_SUFFIX = "__HEART_B64"


def decode_env_snippet() -> str:
    """Shell that restores base64'd environment variables, run before the agent.

    A line of sh rather than a helper baked into the image: the image is the
    operator's to replace, and a decoder living there is a second thing they
    would have to keep in step with heart.
    """
    return (
        "for _n in $(env | sed -n 's/^\\([A-Za-z_][A-Za-z0-9_]*\\)"
        + _B64_SUFFIX
        + "=.*/\\1/p'); do "
        'eval "_b=\\$${_n}' + _B64_SUFFIX + '"; '
        'export "$_n=$(printf %s "$_b" | base64 -d)"; '
        "done; "
    )


# Where the host's agent CLIs land inside the container. Both exist in the
# image, so the binds attach to directories that are already there.
AGENT_BIN = "/opt/agent-bin"
NPM_PREFIX = "/opt/npm-global"

# The executable each agent shells out to, by AGENT_COMMANDS name. Only the
# CLIs -- `api` runs heart's own loop and `shell` runs bash, both already in
# the image.
AGENT_TOOLS = {
    "claude": "claude",
    "codex": "codex",
    "gemini": "gemini",
    "opencode": "opencode",
    "pi": "pi",
    "cursor": "cursor-agent",
}

# Networks a task may ask for. "none" is the default because a task that cannot
# reach the network cannot exfiltrate, and that costs nothing to enforce.
#
# "model" and "build" name docker networks the operator creates, not something
# heart can conjure -- and heart cannot vouch for what they reach. Measured:
# `docker network create heart-model` produces Internal=false, so a container
# on it has full internet egress. A task asking for "model" in the belief that
# it reaches only a local model server gets the open internet instead.
#
# Isolation comes from contrib/egress-proxy.py, not from the network name: an
# --internal network with the proxy as its only reachable container, and an
# allowlist deciding what leaves. Both "model" and "api" default to that same
# network, which is worth saying out loud -- it means the two are synonyms
# unless the operator runs a second proxy with a narrower list. A task asking
# for "model" reaches whatever the shared allowlist permits, vendor APIs
# included. Split them when a task's reach should be narrower than the batch's.
NETWORKS = {
    "none": "none",
    # Egress for agents that call a model. Defaults to the same --internal
    # network as "model", where contrib/egress-proxy.py is the only reachable
    # container and its allowlist decides what leaves. `HEART_API_NETWORK=bridge`
    # restores plain unrestricted egress for anyone who does not run a proxy --
    # deliberately the thing you type rather than the thing you get, because an
    # agent with the open internet should be a decision rather than a default.
    "api": os.getenv("HEART_API_NETWORK", "heart-egress"),
    "model": os.getenv("HEART_MODEL_NETWORK", "heart-egress"),
    "build": os.getenv("HEART_BUILD_NETWORK", "heart-build"),
}

# Inside a container, 127.0.0.1 is the container. A model server bound to host
# loopback is unreachable no matter which network the container joins, so an
# agent that resolves its endpoint from the host's config would quietly fail to
# connect. The container is told the address that reaches the host from its
# side; the host keeps its own.
#
# Pair this with binding the model server to the bridge address rather than
# 0.0.0.0 -- the lazy fix exposes the model to the whole LAN.

@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    writable: bool = False

    def arg(self) -> str:
        return f"{self.source}:{self.target}:{'rw' if self.writable else 'ro'}"


@dataclass(frozen=True)
class SandboxProfile:
    image: str
    mounts: tuple[Mount, ...]
    network: str
    timeout_seconds: int
    env: dict[str, str] = field(default_factory=dict)
    memory: str = "4g"
    cpus: str = "2"
    pids: int = 512

    def docker_sbx_args(self, extra_env: dict[str, str] | None = None) -> list[str]:
        """The same profile rendered for `docker sandbox run` (the docker-sbx
        CLI plugin) instead of `docker run`.

        This is what the profile being data was for: a second runtime is a
        second renderer, not a rewrite. Measured against the plugin at v0.6.0,
        what carries over and what does not:

          carries: the mount table (it takes -v host:target[:ro], and :ro is
                   enforced), the environment, the image via -t.
          does not: --network (no flag exists; the container gets bridge),
                   --cap-drop, --read-only, --memory/--cpus/--pids-limit,
                   --user.

        So this mode gives the privilege boundary the task spec describes and
        none of the blast-radius limits. sandbox_wrap refuses to run a task
        that asked for network "none" here rather than pretend to honour it.

        --workspace is pointed at the journal inbox, not the worktree. The
        plugin mounts whatever it is given read-write at its *host* path, and
        aimed at the worktree that would be a writable second door onto a tree
        the mount table says is read-only. The inbox is already writable by
        design, so pointing there grants nothing that /journal did not.
        --credentials none for the same reason: heart supplies credentials
        through passthrough_env and home_file_mounts, and the plugin managing
        its own would be a second, unaudited path in.
        """
        journal = next((m.source for m in self.mounts if m.target == JOURNAL), None)
        if journal is None:
            raise RuntimeError("docker-sbx needs a journal mount to use as its workspace")
        # A unique name, because the plugin's own is the creation time to the
        # second -- two sandboxes started in the same second collide with
        # "container name is already in use", which is every parallel batch and
        # every --candidates run.
        args = ["-d", "--name", f"heart-{uuid.uuid4().hex[:12]}",
                "-t", self.image, "--workspace", journal, "--credentials", "none"]
        for mount in self.mounts:
            # `docker run` creates a missing bind source; this path does not,
            # and refuses the whole sandbox over it. A denied_paths entry that
            # does not exist yet therefore cannot be pre-denied here -- the diff
            # scan in path_violations stays the backstop it was always meant to
            # be for exactly this case.
            if not os.path.exists(mount.source):
                continue
            args += ["-v", mount.source + ":" + mount.target
                     + ("" if mount.writable else ":ro")]
        env = self.env
        merged = {k: v for k, v in (extra_env or {}).items() if k not in env}
        merged.update(env)
        for key, value in sorted(merged.items()):
            if "\n" in value:
                # `docker sandbox run` truncates a -v/-e value at the first
                # newline and says nothing. Measured: a three-line HEART_PROMPT
                # arrived as its first line, which silently dropped the scope
                # note and the retrieved packet off every prompt. `docker run`
                # had no such limit, so nothing caught it until an agent was
                # asked to echo what it had been given.
                #
                # Base64 rather than escaping: the value is arbitrary agent text
                # and any quoting scheme is a guess about what the plugin's
                # parser does next. The container decodes it in _decode_env
                # before the agent command runs.
                args += ["-e", f"{key}{_B64_SUFFIX}={base64.b64encode(value.encode()).decode()}"]
            else:
                args += ["-e", f"{key}={value}"]
        # The plugin takes an agent name positionally and checks the binary is
        # present before it will start. It selects scaffolding only -- with -d
        # it runs nothing, and heart execs the real command afterwards.
        args.append("claude")
        return args


def profile_for(
    task: TaskSpec,
    workspace: str | Path,
    context_dir: str | Path,
    inbox_dir: str | Path,
    env: dict[str, str] | None = None,
    image: str | None = None,
) -> SandboxProfile:
    """The container a subtask of `task` is allowed to run in."""
    return SandboxProfile(
        image=image or os.getenv("HEART_SANDBOX_IMAGE", DEFAULT_IMAGE),
        mounts=_mounts(task, Path(workspace), Path(context_dir), Path(inbox_dir))
        + home_file_mounts(),
        network=NETWORKS.get(_network(task), "none"),
        timeout_seconds=task.timeout_seconds,
        env=dict(env or {}) | _agent_env(task),
        memory=os.getenv("HEART_SANDBOX_MEMORY", "4g"),
        cpus=os.getenv("HEART_SANDBOX_CPUS", "2"),
    )


@lru_cache(maxsize=None)
def agent_tool_mounts() -> tuple[Mount, ...]:
    """The host's agent CLIs, read-only, so the image carries none of them.

    Baking a CLI into an image costs its full size per agent and a rebuild every
    time it updates. Mounting costs nothing and always runs the build the host
    has -- measured working for claude (a 237MB bun binary), opencode (a 176MB
    ELF) and codex (a node shim) against an image that had never heard of any of
    them. The tradeoff is reproducibility: the version now floats with the host
    rather than being pinned by an image tag, which is why run_agent records
    `agent_version` on every role.

    Node CLIs need their npm prefix mounted whole, not just the shim. The shim
    resolves its imports through `../lib/node_modules` relative to its own path,
    so the bin and lib directories have to keep their arrangement; moving the
    resolved file alone breaks every import it makes.

    Every CLI on the host is mounted, not just this episode's -- the profile is
    built once per episode while roles may each resolve a different agent
    (models.json routes tiers to different ones). They are read-only binaries
    the container could not otherwise reach, so the cost of the extra ones is
    nil next to threading the agent string through every role.

    ponytail: Docker Desktop shares only paths under $HOME. A CLI installed
    outside one -- /usr/local/bin -- mounts but will not execute, and the
    container fails to start. That surfaces as docker's own message through
    runner.sandbox_start_failure rather than as a bad episode, which is why
    there is no filtering here.
    """
    mounts: list[Mount] = []
    seen: set[str] = set()
    marker = f"{os.sep}node_modules{os.sep}"
    for exe in sorted(set(AGENT_TOOLS.values())):
        path = shutil.which(exe)
        if not path:
            continue
        real = os.path.realpath(path)
        if marker in real:
            # <prefix>/lib/node_modules/<pkg>/... -> <prefix>, mounted whole
            real = real[:real.index(marker)].rsplit(os.sep + "lib", 1)[0]
            target = NPM_PREFIX
        elif os.path.basename(real) == exe:
            # The name survived resolution, so this may be a launcher sitting in
            # its own bundle -- cursor-agent is a shell script that runs the
            # node binary beside it, and mounting the script alone gives
            # "/opt/agent-bin/node: No such file or directory". Mount the
            # directory so $0-relative siblings come with it.
            real, target = os.path.dirname(real), f"{AGENT_BIN}/{exe}.d"
        else:
            # A versioned single-file binary (claude resolves to
            # .../claude/versions/2.1.246). The name did not survive, so nothing
            # can be resolving siblings by it, and the file has to be renamed
            # back to the command on the way in.
            target = f"{AGENT_BIN}/{exe}"
        if real in seen:
            continue
        seen.add(real)
        mounts.append(Mount(real, target, writable=False))
    return tuple(mounts)


def agent_tool_path() -> str:
    """PATH for a container, with every mounted CLI on it.

    Spelled out rather than prepended to the image's own PATH: heart cannot read
    an image's ENV before running it, and a bundle mounted at <exe>.d is a
    directory this side invented, so nothing in the image could name it.
    """
    dirs = [m.target for m in agent_tool_mounts() if m.target.endswith(".d")]
    return ":".join([*dirs, AGENT_BIN, f"{NPM_PREFIX}/bin",
                     "/usr/local/sbin", "/usr/local/bin", "/usr/sbin",
                     "/usr/bin", "/sbin", "/bin"])


def container_endpoint(url: str) -> str:
    """A model endpoint rewritten for the inside of a container.

    Inside a container 127.0.0.1 is the container. A model server on the host's
    loopback is unreachable no matter which network the container joins, so an
    agent that resolves its endpoint from the host's config connects to itself
    and reports the model as down.

    The port is preserved because the profile chose it -- one llama-server per
    GPU on adjacent ports is the normal arrangement, and a constant would send
    every profile to the same one.

    This only makes the address reachable. The server still has to be listening
    on something other than loopback: `llama-server --host 127.0.0.1` cannot be
    reached from a container whatever the URL says. Bind it to the bridge
    address rather than 0.0.0.0 -- the lazy fix exposes the model to the LAN.
    """
    parts = urlsplit(url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return url
    host = os.getenv("HEART_CONTAINER_HOST", "host.docker.internal")
    return urlunsplit(parts._replace(
        netloc=f"{host}:{parts.port}" if parts.port else host))


def api_agent_env(model_profile: str) -> dict[str, str]:
    """The resolved model config for an `api:` agent running in a container.

    Resolved on the host and passed as the answer, not the question. The
    container has no models.json -- ~/.config is not mounted -- so an agent
    handed HEART_MODEL_PROFILE exits on
    `cannot read profiles from /home/agent/.config/heart/models.json`, which the
    outcome ladder then records as `no_change` at reward 0.0. Measured.

    Mounting models.json instead would not be enough: a profile's own
    `endpoint` outranks HEART_API_ENDPOINT in resolve_config, so the container
    would read the host's loopback address and talk to itself. Resolving here is
    also the same rule the context packet already follows -- built on the host,
    so the container never needs a credential to fetch what it was given.
    """
    from . import agents_api

    try:
        cfg = agents_api.profile_config(model_profile)
    except Exception:
        cfg = {}
    env = {}
    # The last resort is agents_api's own default rather than nothing: an
    # unresolved endpoint inside a container means 127.0.0.1, which is the
    # container, which is never a model server.
    endpoint = (cfg.get("endpoint") or os.environ.get("HEART_API_ENDPOINT")
                or "http://127.0.0.1:8000/v1")
    env["HEART_API_ENDPOINT"] = container_endpoint(endpoint)
    if model := (cfg.get("model") or os.environ.get("HEART_API_MODEL")):
        env["HEART_API_MODEL"] = model
    key = os.environ.get(cfg["api_key_env"], "") if cfg.get("api_key_env") else ""
    if key := (key or os.environ.get("HEART_API_KEY", "")):
        env["HEART_API_KEY"] = key
    return env


def home_file_mounts() -> tuple[tuple[Mount, ...], tuple[str, ...]]:
    """Host config files placed in the container's HOME.

    This is how a subscription seat gets into the sandbox. Claude Pro/Max and
    the ChatGPT seat authenticate with an OAuth file under $HOME, not an API
    key, so passthrough_env cannot carry them. Name the files:

        HEART_SANDBOX_HOME_FILES=~/.claude/.credentials.json,~/.claude.json

    Each is mounted read-only at the same position relative to HOME that it
    holds on the host.

    Read-only is deliberate, not caution for its own sake. An OAuth refresh
    rotates the refresh token at the provider, so a container that refreshes
    against a copy invalidates the one on the host and logs you out of your own
    machine. Refused instead: a run that outlives the access token fails with an
    auth error, which is recoverable, rather than silently taking your session
    with it.

    Files only, never a directory: mounting ~/.claude whole would hand the agent
    every project history and transcript on the machine, and it is not a
    credential store, it is a home.

    Agent roles only -- a verifier has no model to authenticate to.
    """
    names = [n.strip() for n in os.getenv("HEART_SANDBOX_HOME_FILES", "").split(",")
             if n.strip()]
    mounts = []
    host_home = Path.home()
    for name in names:
        path = Path(name).expanduser()
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(host_home)
        except ValueError:
            # Only paths under the host's home have a meaningful position in
            # the container's home; anything else has nowhere obvious to land.
            continue
        mounts.append(Mount(str(path), f"{HOME}/{rel}", writable=False))
    return tuple(mounts)


def proxy_env() -> dict[str, str]:
    """Point the agent at an egress proxy, if the operator runs one.

    `network: "api"` is plain bridge egress -- reaching api.anthropic.com means
    reaching everything else too. HEART_SANDBOX_PROXY plus an --internal network
    narrows that to an allowlist (see contrib/egress-proxy.py).

    Both spellings because the tools disagree: curl and python read the
    lowercase names, most node CLIs read the uppercase ones, and a proxy honored
    by half an image is worse than none -- the half that ignores it is the half
    that has no route and fails with no explanation.

    Fails closed on its own: on an --internal network an agent that ignores
    these has no gateway at all, so it reaches nothing rather than going direct.
    """
    url = os.getenv("HEART_SANDBOX_PROXY", "").strip()
    if not url:
        return {}
    return {k: url for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")}


def passthrough_env() -> dict[str, str]:
    """Host variables the operator named as safe to hand the agent.

    An allowlist, never a copy of the environment: os.environ holds every
    credential the operator's shell has, and a container that cannot reach the
    network is a weak consolation once the keys are already inside it.

    This is the answer for key-based auth on every CLI -- ANTHROPIC_API_KEY,
    OPENAI_API_KEY, CURSOR_API_KEY. It is not the answer for a subscription
    seat, whose OAuth lives in a file under $HOME -- home_file_mounts places
    those.

    Agent roles only. A verifier with a credential is an exfiltration path with
    a test suite wrapped around it, and it has no model to call anyway.
    """
    names = [n.strip() for n in os.getenv("HEART_SANDBOX_ENV", "").split(",") if n.strip()]
    return {n: os.environ[n] for n in names if os.environ.get(n)}


def _agent_env(task: TaskSpec) -> dict[str, str]:
    # No HEART_API_ENDPOINT here. It used to be pinned to a constant whenever
    # the task asked for the "model" network, and profile env outranks the
    # caller's, so it silently beat the endpoint api_agent_env resolves from the
    # profile -- sending every task to one hardcoded host:port regardless of
    # which server its profile named. Two code paths setting one fact, and the
    # stale one winning: the same shape as the duplicated refusal parser.
    return {"EVENT_JOURNAL_DIR": JOURNAL, "HOME": HOME,
            "PATH": agent_tool_path(), **CACHE_ENV, **proxy_env(),
            **passthrough_env()}


def _network(task: TaskSpec) -> str:
    return (getattr(task, "network", "") or "none").strip().lower()


def _mounts(task: TaskSpec, workspace: Path, context: Path, inbox: Path) -> tuple[Mount, ...]:
    """Worktree, context, journal -- and nothing else.

    allowed_paths inverts the default: the worktree goes in read-only and each
    allowed subtree is remounted writable on top. Saying "only these" is exact,
    whereas trying to deny everything else is a list you will get wrong.
    denied_paths then goes back over the top read-only, so it wins either way --
    the same precedence path_violations already documents.
    """
    restricted = bool(task.allowed_paths)
    mounts = [Mount(str(workspace), WORK, writable=not restricted)]

    for rel in task.allowed_paths:
        target = _under(WORK, rel)
        if target:
            mounts.append(Mount(str(workspace / rel), target, writable=True))

    for rel in task.denied_paths:
        target = _under(WORK, rel)
        if target:
            mounts.append(Mount(str(workspace / rel), target, writable=False))

    mounts += _git_mounts(task, workspace)
    mounts += agent_tool_mounts()

    # Read-only: the packet, retrieved memory and prompt are built on the host
    # so the container never needs a credential to fetch them.
    mounts.append(Mount(str(context), CONTEXT, writable=False))
    # The one outbound channel: this run's own journal inbox, not the shared
    # journal. Per-run means one container can neither read nor corrupt
    # another's record, and concurrent appends cannot interleave. The host
    # drains it -- appending needs no credential, a database connection would.
    mounts.append(Mount(str(inbox), JOURNAL, writable=True))
    return tuple(mounts)


def _git_mounts(task: TaskSpec, workspace: Path) -> list[Mount]:
    """What a worktree needs for git to work inside the container.

    A worktree's `.git` is a file, not a directory, holding an absolute path:
    `gitdir: <repo>/.git/worktrees/<name>`. Mount only the worktree and that
    path does not exist, so every git command fails -- including the `git diff`
    the review roles are told to run.

    Both mounts land at their host paths because the pointer is absolute. The
    worktree itself does not: git tolerates being read from somewhere other than
    where it was recorded, which is what lets the tree sit at /work.

    The object store goes in read-only, and that is the whole point. Measured:
    diff, status, log and blame all work against it; `git add` and `git commit`
    are refused ("unable to create temporary file: read-only file system", then
    "failed to insert into database"). The agent
    gets full history to debug with and cannot write a single object or move a
    ref. Heart commits afterwards on the host, where the store is writable.

    The per-worktree directory is writable because git updates its index and
    HEAD to answer those reads. It holds this worktree's metadata only -- no
    objects, no refs belonging to anything else.
    """
    git_dir = Path(task.repo_path) / ".git"
    if not git_dir.is_dir():
        return []  # a clone or a plain export needs none of this
    mounts = [Mount(str(git_dir), str(git_dir), writable=False)]
    per_worktree = git_dir / "worktrees" / workspace.name
    if per_worktree.is_dir():
        mounts.append(Mount(str(per_worktree), str(per_worktree), writable=True))
    return mounts


def _under(root: str, rel: str) -> str | None:
    """Join, refusing anything that escapes root. A task spec is data, and a
    denied_paths entry of '../..' must not become a mount of the host."""
    rel = rel.strip().lstrip("/")
    if not rel:
        return None
    target = os.path.normpath(os.path.join(root, rel))
    return target if target.startswith(root + "/") else None


# Caches a test run writes into the tree it is testing. On a read-only worktree
# those writes fail for reasons that have nothing to do with the code, so each
# tool is told to cache somewhere writable instead.
#
# Both roles get these, not just verifiers. allowed_paths is authored in the
# vocabulary of "which source files should this task change"; the mount table
# enforces "which bytes may be written". A toolchain writes bytes nobody
# predicted -- __pycache__, .pytest_cache at rootdir, lockfiles, build output --
# so even an accurate source-scope prediction is refused for reasons unrelated
# to scope. Routing those writes out of the tree is what keeps the two
# vocabularies comparable.
#
# It also keeps the learning loop clean. pytest reports a cache it cannot write
# as "could not create cache path ...: [Errno 30] Read-only file system", which
# matches _DENIAL_SIGNS and would be recorded as a path the task *needed*.
# plexus accumulates that union monotonically and never retracts it.
#
# Told, not tmpfs'd. Mounting a tmpfs at /work/.pytest_cache does not work: the
# mountpoint has to be created inside a bind that is already read-only, and
# docker refuses the container outright -- exit 125, before a verifier runs.
# An env var needs no mountpoint and survives the next cache directory some
# tool invents, as long as someone adds the line.
CACHE_ENV = {
    "PYTEST_ADDOPTS": "-p no:cacheprovider",
    "RUFF_CACHE_DIR": "/tmp/ruff-cache",
    "MYPY_CACHE_DIR": "/tmp/mypy-cache",
    "npm_config_cache": "/tmp/npm-cache",
    "XDG_CACHE_HOME": "/tmp/cache",
}


def verifier_profile_for(
    task: TaskSpec,
    workspace: str | Path,
    inbox_dir: str | Path,
    env: dict[str, str] | None = None,
    image: str | None = None,
) -> SandboxProfile:
    """The container a verifier runs in: same mechanism, different role.

    Not a second sandbox. The profile is data, so "stricter" is a different set
    of arguments to the same machinery -- one image, one runtime, one set of
    flags rendered differently.

    Three things change, and none of them is containment strength:

    Writability. The agent must write the worktree; the verifier must not. If a
    verifier can edit the tree it is judging, "produce" and "judge" stop being
    separate steps -- which is a reward-integrity property, not a security one,
    and the reason this profile exists at all.

    Network. Always none, never derived from the task. A test suite has no
    business on the network: it kills exfiltration-via-test and network-flaky
    verifiers in one move, and an agent granted "api" egress must not hand that
    egress to the thing scoring it.

    Context. Verifiers get no /context mount. They judge code, not memory, and a
    verifier that can read the continuity packet can be steered by it.
    """
    return SandboxProfile(
        image=image or os.getenv("HEART_SANDBOX_IMAGE", DEFAULT_IMAGE),
        mounts=(
            Mount(str(workspace), WORK, writable=False),
            Mount(str(inbox_dir), JOURNAL, writable=True),
            # The agent CLIs come along even here. Not for the verifier's use --
            # docker-sbx takes an agent name positionally and refuses to start a
            # sandbox whose image lacks that binary, so a verifier-only sandbox
            # is not something the plugin will create. Read-only, and no
            # credential reaches a verifier to use them with.
            *agent_tool_mounts(),
        ),
        network="none",
        timeout_seconds=task.timeout_seconds,
        env=dict(env or {}) | {
            "EVENT_JOURNAL_DIR": JOURNAL,
            "HOME": HOME,
            # same reasoning as verify.py: a stale .pyc from a same-second edit
            # must never decide whether a verifier passes
            "PYTHONDONTWRITEBYTECODE": "1",
            **CACHE_ENV,
        },
        memory=os.getenv("HEART_SANDBOX_MEMORY", "4g"),
        cpus=os.getenv("HEART_SANDBOX_CPUS", "2"),
    )


def inbox_for(key: str) -> Path:
    """The journal inbox a container writes into, asked of arteries.

    Keyed by episode, not by subtask: the inbox is the isolation unit (one per
    container) while ARTERIES_RUN_ID is the identity unit (one per role). Every
    role in an episode appends to the same directory and the drain separates
    them by the run_id on each event.

    Asked, never computed. Guessing the path is the failure that cannot be
    noticed: a container writing where the drain does not read loses everything
    that run remembered, silently, and the episode looks like it simply had no
    memory. Raising here follows the same rule as a missing docker binary -- a
    requested sandbox must fail loudly rather than quietly degrade.
    """
    import subprocess
    try:
        out = subprocess.run(["python3", "-m", "arteries.cli", "journal", "inbox", key],
                             capture_output=True, text=True, timeout=30, check=True)
    except Exception as exc:
        raise RuntimeError(
            "HEART_SANDBOX=docker needs arteries to name the journal inbox "
            f"(`art journal inbox {key}`); it is not answering: {exc}"
        ) from exc
    path = Path(out.stdout.strip())
    if not path.is_absolute():
        raise RuntimeError(f"arteries returned a non-absolute inbox path: {path!r}")
    return path
