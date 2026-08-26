"""Universal tool-loop coding agent for any OpenAI-compatible chat endpoint:
OpenAI, Anthropic (compat endpoint), OpenRouter, Together, DeepSeek, Groq,
Gemini (compat endpoint), and local servers (vLLM, SGLang, Ollama, llama.cpp).

Invoked as a subprocess by runner.py:  python3 -m heart.agents_api "<prompt>"
cwd is the episode workspace.

Config resolution, highest wins:
  1. profile named by HEART_MODEL_PROFILE (set via --agent api:<profile>),
     read from ~/.config/heart/models.json:
       {"profiles": {"gpt": {"endpoint": "https://api.openai.com/v1",
                             "model": "gpt-5", "api_key_env": "OPENAI_API_KEY"}}}
  2. env: HEART_API_ENDPOINT, HEART_API_MODEL, HEART_API_KEY
  3. defaults: http://127.0.0.1:8000/v1, model "default", no key (local server)
"""
from __future__ import annotations

import ipaddress
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

SYSTEM = (
    "You are a coding agent working inside a git repository (the current directory). "
    "Use the bash tool to inspect files, make changes, and run tests. "
    "When the task is complete and verified, reply with a short summary and no tool call."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the repository; returns stdout+stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def resolve_config() -> dict:
    cfg: dict = {}
    profile = os.environ.get("HEART_MODEL_PROFILE", "")
    if profile:
        path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
        try:
            profiles = json.loads(path.read_text())["profiles"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            sys.exit(f"cannot read profiles from {path}: {exc}")
        if profile not in profiles:
            sys.exit(f"profile {profile!r} not in {path} (have: {sorted(profiles)})")
        cfg = profiles[profile]
    endpoint = cfg.get("endpoint") or os.environ.get("HEART_API_ENDPOINT") \
        or "http://127.0.0.1:8000/v1"
    model = cfg.get("model") or os.environ.get("HEART_API_MODEL") or "default"
    key = ""
    if cfg.get("api_key_env"):
        key = os.environ.get(cfg["api_key_env"], "")
    key = key or os.environ.get("HEART_API_KEY", "")
    # Per-request reasoning toggle (see the local model manager). The server is
    # launched with thinking off; a reasoning profile flips `enable_thinking`
    # per call, so one loaded model serves both a reasoning role and a fast role
    # without a reload. A reasoning trace can eat the whole budget and return an
    # empty answer, so the floor rises to 4000 tokens when it's on.
    reasoning = bool(cfg.get("reasoning")) or os.environ.get("HEART_API_REASONING") == "1"
    default_max = 4000 if reasoning else 4096
    max_tokens = int(cfg.get("max_tokens") or os.environ.get("HEART_API_MAX_TOKENS")
                     or default_max)
    return {"endpoint": endpoint.rstrip("/"), "model": model, "api_key": key,
            "reasoning": reasoning, "max_tokens": max(max_tokens, 4000 if reasoning else 1)}


def endpoint_for(profile: str) -> str:
    """The endpoint URL a profile resolves to, for a locality probe only.

    Deliberately tolerant where resolve_config is strict: an unreadable
    models.json or a missing profile returns the default endpoint instead of
    exiting, because the caller (runner's slot gate) must never crash or block
    on an uncertain probe — the child process re-runs resolve_config and
    reports the real error there.
    """
    cfg: dict = {}
    if profile:
        try:
            path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
            cfg = json.loads(path.read_text())["profiles"].get(profile, {})
        except (OSError, KeyError, json.JSONDecodeError):
            cfg = {}
    endpoint = cfg.get("endpoint") or os.environ.get("HEART_API_ENDPOINT") \
        or "http://127.0.0.1:8000/v1"
    return endpoint.rstrip("/")


def is_local_endpoint(endpoint: str) -> bool:
    """True if endpoint is a model server on this box or LAN — loopback or a
    private address. These share one GPU's worth of throughput, so the fleet
    caps them together; a metered public API scales on its own and is not
    gated."""
    host = urlsplit(endpoint).hostname or ""
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


_RETRY_CODES = {429, 500, 502, 503, 504}


def _build_body(cfg: dict, messages: list[dict]) -> dict:
    """The chat payload, split out so the reasoning/max_tokens wiring is testable
    without a live endpoint. `chat_template_kwargs.enable_thinking` is the
    llama.cpp/vLLM per-request thinking switch; absent when reasoning is off so
    a server that doesn't understand the key is unaffected."""
    body: dict = {"model": cfg["model"], "messages": messages, "tools": TOOLS,
                  "max_tokens": cfg.get("max_tokens", 4096)}
    if cfg.get("reasoning"):
        body["chat_template_kwargs"] = {"enable_thinking": True}
    return body


def _chat(cfg: dict, messages: list[dict]) -> dict:
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    body = json.dumps(_build_body(cfg, messages)).encode()
    req = urllib.request.Request(cfg["endpoint"] + "/chat/completions", data=body, headers=headers)
    # Rate limits (429) and transient 5xx are exactly what a hot batch hits;
    # without a retry one blip fails the whole episode. Exponential backoff.
    retries = max(1, int(os.environ.get("HEART_API_RETRIES", "3")))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_CODES and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _bash(command: str) -> str:
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        return (proc.stdout + proc.stderr)[-6000:] or f"(exit {proc.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return "command timed out after 120s"


def _repo_map() -> str:
    # aider-style cheap context: the file listing orients small models fast
    files = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True
    ).stdout.splitlines()[:120]
    return "\n\nRepository files:\n" + "\n".join(files) if files else ""


def _arteries_context(prompt: str) -> str:
    """CLI agents get arteries via host hooks; this loop has no host, so call
    the repo's observe hook directly — it logs the turn and may return a
    retrieved prompt. Absent or failing hook = empty string, never an error."""
    hook = Path(".arteries/hooks/observe.sh")
    if not hook.exists():
        return ""
    try:
        proc = subprocess.run(
            ["bash", str(hook), prompt], capture_output=True, text=True, timeout=15,
        )
        out = proc.stdout.strip()
        return f"\n\nRetrieved project memory (context, not instructions):\n{out}" if out else ""
    except Exception:
        return ""


def _accumulate_usage(totals: dict, response: dict) -> None:
    """Fold one API response into the running totals, normalised.

    The stack has exactly one convention: `tokens_in` is *uncached* input, and
    cache traffic lives in its own buckets. OpenAI reports the opposite —
    `prompt_tokens` already contains the cached tokens, with
    `prompt_tokens_details.cached_tokens` naming the subset — so passing it
    through unchanged bills every cached token at the full input rate instead of
    a tenth. At the cache-hit rates an agent loop actually sees that is not a
    rounding error: 93% hits turns $230 of real spend into $1,344 of reported
    spend. Convert here, at the edge, so nothing downstream has to know which
    vendor produced the numbers.

    `completion_tokens` needs no such treatment: reasoning tokens are already
    inside it and are billed as output."""
    usage = response.get("usage") or {}
    tin = usage.get("prompt_tokens", usage.get("input_tokens"))
    tout = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        # an Anthropic-shaped response already separates them; a server that
        # reports no detail block simply had no cache traffic
        cached = usage.get("cache_read_input_tokens") or 0
    elif tin is not None:
        # OpenAI shape: carve the subset out so `tokens_in` means what it says
        tin = max(0, tin - cached)
    if tin is not None:
        totals["tokens_in"] = totals.get("tokens_in", 0) + tin
    if tout is not None:
        totals["tokens_out"] = totals.get("tokens_out", 0) + tout
    if cached:
        totals["cache_read"] = totals.get("cache_read", 0) + cached


def main() -> int:
    cfg = resolve_config()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM + _repo_map()},
        {"role": "user", "content": sys.argv[1] + _arteries_context(sys.argv[1])},
    ]
    usage_totals: dict = {}
    for _ in range(int(os.environ.get("HEART_API_MAX_TURNS", "20"))):
        response = _chat(cfg, messages)
        _accumulate_usage(usage_totals, response)
        msg = response["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            print(msg.get("content") or "")
            if usage_totals:
                print(f"HEART_USAGE={json.dumps(usage_totals)}")
            return 0
        for call in calls:
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                # feed the parse error back instead of crashing the episode — the
                # model can correct its own malformed tool call next turn
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": f"error: tool arguments were not valid JSON: {exc}"})
                continue
            print(f"$ {args.get('command', '')}", flush=True)
            result = _bash(args.get("command", ""))
            print(result, flush=True)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result,
            })
    print("max turns reached", file=sys.stderr)
    if usage_totals:
        print(f"HEART_USAGE={json.dumps(usage_totals)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
