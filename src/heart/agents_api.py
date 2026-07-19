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

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

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
        or os.environ.get("HEART_LOCAL_ENDPOINT") or "http://127.0.0.1:8000/v1"
    model = cfg.get("model") or os.environ.get("HEART_API_MODEL") \
        or os.environ.get("HEART_LOCAL_MODEL") or "default"
    key = ""
    if cfg.get("api_key_env"):
        key = os.environ.get(cfg["api_key_env"], "")
    key = key or os.environ.get("HEART_API_KEY", "")
    return {"endpoint": endpoint.rstrip("/"), "model": model, "api_key": key}


def _chat(cfg: dict, messages: list[dict]) -> dict:
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    body = json.dumps({
        "model": cfg["model"], "messages": messages, "tools": TOOLS, "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(cfg["endpoint"] + "/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]


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


def main() -> int:
    cfg = resolve_config()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM + _repo_map()},
        {"role": "user", "content": sys.argv[1] + _arteries_context(sys.argv[1])},
    ]
    for _ in range(int(os.environ.get("HEART_API_MAX_TURNS", "20"))):
        msg = _chat(cfg, messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            print(msg.get("content") or "")
            return 0
        for call in calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            print(f"$ {args.get('command', '')}", flush=True)
            result = _bash(args.get("command", ""))
            print(result, flush=True)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result,
            })
    print("max turns reached", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
