"""An allowlist of hosts a sandboxed agent may reach, and nothing else.

`network: "api"` is plain bridge egress: an agent that needs api.anthropic.com
gets the whole internet along with it. The sandbox notes have pointed at "a
proxy allowlist" as the upgrade path since bwrap; this is it.

Put the agent on an --internal network with only this container reachable, and
tell it to proxy. The route out is then exactly the hosts named in ALLOW. It
fails closed by construction: an agent that ignores the proxy variables has no
gateway at all and reaches nothing, rather than quietly going direct.

    docker network create --internal heart-egress
    docker run -d --name egress --restart unless-stopped --network bridge \
      -e ALLOW=api.anthropic.com,host.docker.internal:8001 \
      -v $PWD/contrib/egress-proxy.py:/proxy.py:ro \
      --entrypoint python3 heart-agent:latest /proxy.py
    docker network connect heart-egress egress

    HEART_API_NETWORK=heart-egress HEART_SANDBOX_PROXY=http://egress:8888 \
      heart run task.json --agent claude:sonnet    # task asks for network "api"

`plexus doctor --fix` provisions all of this; the lines above are what it does.

CONNECT is tunnelled after the host check, never intercepted -- no TLS
termination, no certificate to inject, and the proxy sees hostnames rather than
traffic. Plain HTTP is forwarded by absolute-URI so the same allowlist covers a
local model server on http://.

A name in ALLOW covers that host and its subdomains: `anthropic.com` allows
`api.anthropic.com`. Append a port -- `host.docker.internal:8001` -- to allow
only that one; a bare name allows every port on that host, which for the host
alias means the host's Postgres and heart's own server too. Matching is on the name the client asked for, which is the
point -- an agent that resolves a name itself and connects by IP has no route to
do it.

`*` in ALLOW is the web lane: any *public* host on 80 or 443. Public is checked
on the addresses the name resolves to, not on the name, and the connection goes
to the address that was checked -- so a name that resolves to the LAN, the
tailnet (100.64/10), the Docker Desktop VM or loopback is refused, and a second
lookup cannot swap in one that does. IP literals are refused outright: they
skip DNS, and DNS is where the malware filter lives (the web proxy runs with
--dns pointed at Quad9, which answers NXDOMAIN for known-malicious domains).
Named entries still win over `*`, which is how the local model server stays
reachable on a lane that otherwise only reaches the public internet. DENY names
hosts to refuse whatever ALLOW says.

INJECT_PORT turns on the credential injector, the other half of the design.
The agent is given a sentinel credential and a base URL pointing here; this
side checks the sentinel, swaps in the real credential from /secrets/<route>,
and forwards over TLS. The real credential never enters an agent container,
and a request carrying any credential other than the sentinel -- an attacker's
own API key, say, used to upload the repo to the attacker's account through an
allowlisted host -- is refused. That second property only holds if the
injected host is unreachable any other way, so while the injector runs, CONNECT
to an injected host is refused even when ALLOW or `*` would pass it.

The log is worth reading. Claude Code also reaches for mcp-proxy.anthropic.com
and http-intake.logs.us5.datadoghq.com; under plain bridge egress all of that
leaves the machine unremarked, and the episodes pass without it. Every line
carries the client's address, so `docker inspect` names the container behind
it.

ponytail: no auth on the proxy itself, no logging to disk, no upstream
chaining. The network decides who can reach it, and `docker logs` is the log.
"""
import asyncio
import ipaddress
import json
import os
import socket
import ssl
import sys
import time

ALLOW = tuple(h.strip().lower() for h in os.environ.get("ALLOW", "").split(",") if h.strip())
DENY = tuple(h.strip().lower() for h in os.environ.get("DENY", "").split(",") if h.strip())
PORT = int(os.environ.get("PORT", "8888"))
INJECT_PORT = int(os.environ.get("INJECT_PORT", "0") or 0)
SECRETS = os.environ.get("SECRETS_DIR", "/secrets")

#: The one credential an agent container ever holds: this prefix plus the box's
#: random seed. Worth nothing away from this proxy, and worth something here
#: only to a container that was handed it -- a run whose seats were withheld
#: can reach the injector but not guess its way through. Shaped like a real
#: OAuth token so a client that sanity-checks the prefix accepts it.
SENTINEL_PREFIX = "sk-ant-oat01-heart-"

#: Codex's stand-in. Codex reads claims out of its tokens before it sends
#: anything, so the sentinel has to parse as a JWT: this one is unsigned, says
#: nothing true, and expires in 2100. The box's seed is its third segment.
#: Must match heart.sandbox.CODEX_HEAD.
CODEX_HEAD = ("eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0."
              "eyJleHAiOjQxMDI0NDQ4MDAsImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6"
              "eyJjaGF0Z3B0X2FjY291bnRfaWQiOiJoZWFydC1zZW50aW5lbCJ9fQ")

#: route (the first path segment on the injector) -> (upstream host, the
#: sentinel that route accepts, where the real credential lives). The Claude
#: token is a file the operator saved; Codex's is its own auth.json, read live
#: so a refresh the host CLI does is picked up without a restart.
ROUTES = {"anthropic": "api.anthropic.com", "chatgpt": "chatgpt.com"}


def sentinels() -> dict[str, str]:
    """route -> the stand-in this box's agents are handed, derived from the
    random seed in /secrets/sentinel (heart.sandbox.sentinels makes the same
    two). Read per request, so a rotated seed needs no restart. No seed, no
    sentinel: the injector then accepts nothing."""
    try:
        seed = open(os.path.join(SECRETS, "sentinel"), encoding="utf-8").read().strip()
    except OSError:
        return {}
    return {"anthropic": f"{SENTINEL_PREFIX}{seed}", "chatgpt": f"{CODEX_HEAD}.{seed}"} if seed else {}
_SECRET_FILES = {"anthropic": "anthropic", "chatgpt": "/codex/auth.json"}

WEB_PORTS = (80, 443)

#: Bytes a client may send to web-lane hosts per window. Counted client->upstream
#: only, and only for hosts `*` let through -- the model's own traffic (the
#: injector, the named local model) is prompts full of repo and is not a leak.
#: Browsing is small going out: a TLS hello and a request line per page, a few
#: KB. A repo, a dataset or a database dump is not. This stops bulk; a small
#: repo compresses under any cap that still lets an agent browse, which is why
#: a sensitive repo belongs on the api lane instead.
#: ponytail: keyed on client IP, so a new container that inherits an address
#: inside the window inherits its spend; key on container id if that bites.
UPLOAD_BUDGET = int(os.environ.get("UPLOAD_BUDGET", str(1024 * 1024)) or 0)
UPLOAD_WINDOW = int(os.environ.get("UPLOAD_WINDOW", "600"))
_spend: dict[str, tuple[float, int]] = {}


def charge(peer: str, n: int) -> bool:
    """Record n bytes sent by peer; False once the window's budget is gone."""
    now = time.monotonic()
    start, used = _spend.get(peer, (now, 0))
    if now - start > UPLOAD_WINDOW:
        start, used = now, 0
    _spend[peer] = (start, used + n)
    return not UPLOAD_BUDGET or used + n <= UPLOAD_BUDGET


def _named(host: str, port: int, entries) -> bool:
    for entry in entries:
        name, _, want = entry.partition(":")
        if want and want != str(port):
            continue
        if host == name or host.endswith("." + name):
            return True
    return False


def permitted(host: str, port: int) -> bool:
    """Is this host:port on the list?

    An entry may name a port -- `host.docker.internal:8001` -- and then only
    that port is reachable. This matters more than it looks: the host alias is
    on the list so an agent can call a model server on :8001, and a bare name
    also hands it every other port the host has open. Postgres on 5432 and
    heart's own server on 8000 were both reachable by CONNECT through a proxy
    whose job was to stop exactly that.

    A bare name still allows any port, because that is what the existing
    deployments are configured with and a silent narrowing would read as the
    network being broken. Name the port; the log line says which one was used.

    `*` passes the name check here for any web port; whether the host is
    actually public is decided at connect time, in _resolve_public.
    """
    host = host.lower().rstrip(".")
    if _named(host, port, DENY):
        return False
    if INJECT_PORT and host in _injected_hosts():
        return False
    if _named(host, port, (e for e in ALLOW if e != "*")):
        return True
    if "*" in ALLOW and port in WEB_PORTS:
        try:
            ipaddress.ip_address(host.strip("[]"))
            return False  # an IP literal skips the DNS filter
        except ValueError:
            return True
    return False


def _web_only(host: str, port: int) -> bool:
    """Permitted by `*` alone rather than by a named entry -- the case that
    must be pinned to a public address."""
    host = host.lower().rstrip(".")
    return not _named(host, port, (e for e in ALLOW if e != "*"))


def _public(addr: str) -> bool:
    return ipaddress.ip_address(addr).is_global


async def _resolve_public(host: str, port: int) -> str | None:
    """The address to connect to for a web-lane host, or None if any address
    the name resolves to is not public.

    All of them, not the first: a name with one public and one private record
    would otherwise reach the private one whenever the resolver reorders.
    """
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM)
    except OSError:
        return None
    # v4 first: an agent network rarely has a v6 route out
    addrs = sorted({info[4][0] for info in infos}, key=lambda a: ":" in a)
    if not addrs or not all(_public(a) for a in addrs):
        return None
    return addrs[0]


async def _open(host: str, port: int):
    """Connect upstream, or (None, reason). Web-lane hosts are pinned to the
    public address that was checked; named hosts connect as asked."""
    target = host
    if _web_only(host, port):
        target = await _resolve_public(host, port)
        if target is None:
            return None, f"{host} does not resolve to a public address"
    try:
        return await asyncio.open_connection(target, port), ""
    except OSError as exc:
        return None, f"connect failed: {exc}"


async def _pipe(reader, writer, meter: str = ""):
    try:
        while chunk := await reader.read(65536):
            if meter and not charge(meter, len(chunk)):
                print(f"cut {meter}: upload budget of {UPLOAD_BUDGET} bytes "
                      f"per {UPLOAD_WINDOW}s spent", flush=True)
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def _splice(reader, writer, up_r, up_w, meter: str = ""):
    """Both directions; `meter` names the client whose outbound bytes count
    against the upload budget, or "" for traffic that does not."""
    await asyncio.gather(_pipe(reader, up_w, meter), _pipe(up_r, writer),
                         return_exceptions=True)


#: Carried in every refusal so heart can tell "the allowlist stopped this" from
#: "the agent failed". Without it a too-narrow ALLOW scores every episode
#: `no_change` at reward 0.0 -- measured -- and a batch teaches the model it
#: cannot work, from runs that never reached a model. A token rather than a
#: sentence because an agent echoing the phrase should not be able to trip it by
#: accident.
DENIED_MARKER = "HEART_EGRESS_DENIED"


async def _deny(writer, reason: str):
    reason = f"{DENIED_MARKER} {reason}"
    writer.write(f"HTTP/1.1 403 Forbidden\r\nContent-Length: {len(reason)}\r\n"
                 f"Connection: close\r\n\r\n{reason}".encode())
    await writer.drain()
    writer.close()


def _peer(writer) -> str:
    info = writer.get_extra_info("peername")
    return info[0] if info else "?"


async def _handle(reader, writer):
    peer = _peer(writer)
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=30)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
        writer.close()
        return
    try:
        method, target, _ = head.decode("latin-1").split(None, 2)
    except ValueError:
        writer.close()
        return

    if method.upper() == "CONNECT":
        host, _, port = target.rpartition(":")
        if not host:
            host, port = target, ""
        host = host.strip("[]")
        try:
            port = int(port or 443)
        except ValueError:
            return await _deny(writer, "bad CONNECT target\n")
        if not permitted(host, port):
            print(f"deny CONNECT {host}:{port} from {peer}", flush=True)
            return await _deny(writer, f"{host}:{port} is not in the sandbox allowlist\n")
        # drain the rest of the request head before tunnelling
        try:
            while (await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=30)) != b"\r\n":
                pass
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return
        meter = peer if _web_only(host, port) else ""
        if meter and not charge(peer, 0):
            print(f"deny CONNECT {host}:{port} from {peer}: upload budget spent", flush=True)
            return await _deny(writer, "upload budget for web hosts is spent\n")
        upstream, why = await _open(host, port)
        if upstream is None:
            print(f"deny CONNECT {host}:{port} from {peer}: {why}", flush=True)
            return await _deny(writer, f"{why}\n")
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        print(f"allow CONNECT {host}:{port} from {peer}", flush=True)
        return await _splice(reader, writer, *upstream, meter=meter)

    # plain HTTP arrives as an absolute URI: GET http://host:port/path HTTP/1.1
    if not target.startswith("http://"):
        return await _deny(writer, "only proxied requests are accepted\n")
    rest = target[len("http://"):]
    authority, slash, path = rest.partition("/")
    host, _, port = authority.partition(":")
    port = int(port or 80)
    if not permitted(host, port):
        print(f"deny {method} {host}:{port} from {peer}", flush=True)
        return await _deny(writer, f"{host}:{port} is not in the sandbox allowlist\n")
    meter = peer if _web_only(host, port) else ""
    if meter and not charge(peer, 0):
        print(f"deny {method} {host}:{port} from {peer}: upload budget spent", flush=True)
        return await _deny(writer, "upload budget for web hosts is spent\n")
    upstream, why = await _open(host, port)
    if upstream is None:
        print(f"deny {method} {host}:{port} from {peer}: {why}", flush=True)
        return await _deny(writer, f"{why}\n")
    up_r, up_w = upstream
    up_w.write(f"{method} /{path} HTTP/1.1\r\n".encode("latin-1"))
    await up_w.drain()
    print(f"allow {method} {host}:{port} from {peer}", flush=True)
    await _splice(reader, writer, up_r, up_w, meter=meter)


# --- credential injector ----------------------------------------------------

# Hop-by-hop and framing headers are rebuilt here; the credential headers are
# replaced. Anything else the client sent goes through untouched.
_DROP = {"host", "authorization", "x-api-key", "connection", "keep-alive",
         "proxy-connection", "proxy-authorization", "te", "upgrade",
         "content-length", "transfer-encoding", "chatgpt-account-id",
         # a session cookie is a credential too: an attacker's would log the
         # request into the attacker's account as surely as their API key
         "cookie"}


def _secret_doc(route: str) -> tuple[str, dict]:
    try:
        raw = open(os.path.join(SECRETS, _SECRET_FILES.get(route, route)),
                   encoding="utf-8").read().strip()
    except OSError:
        return "", {}
    if raw.startswith("{"):
        try:
            return "", json.loads(raw)
        except ValueError:
            return "", {}
    return raw, {}


def _injected_hosts() -> set[str]:
    """Upstreams whose seat this proxy holds. Only those lose their CONNECT
    route: a Codex still authenticating from a mounted file must keep reaching
    chatgpt.com until its seat is injected too."""
    return {host for route, host in ROUTES.items() if secret(route)}


def secret(route: str) -> str:
    """The real credential for a route, read on every request.

    Every request, not once at start: a token saved or rotated on the host is
    live on the next call without restarting the proxy. The file is the bare
    token (what `claude setup-token` prints), a Claude credentials JSON, or
    Codex's auth.json.
    """
    raw, doc = _secret_doc(route)
    return raw or ((doc.get("claudeAiOauth") or {}).get("accessToken")
                   or (doc.get("tokens") or {}).get("access_token") or "")


def account(route: str) -> str:
    """The ChatGPT account id Codex sends beside its token, or ""."""
    return ((_secret_doc(route)[1].get("tokens") or {}).get("account_id") or "")


def sentinel_only(headers: list[tuple[str, str]], sentinel: str) -> bool:
    """True when the request carries the sentinel and no other credential.

    Both headers are checked because the two Anthropic auth styles use
    different ones. Any value that is not the sentinel is a credential this
    proxy did not issue, and forwarding it would make the injector a clean
    path for exactly the foreign-key traffic it exists to stop.
    """
    seen = False
    for name, value in headers:
        name, value = name.lower(), value.strip()
        if name == "authorization":
            if value != f"Bearer {sentinel}":
                return False
            seen = True
        elif name == "x-api-key":
            if value != sentinel:
                return False
            seen = True
    return seen


def rewrite(method: str, path: str, headers: list[tuple[str, str]], body: bytes,
            upstream: str, real: str, account_id: str = "") -> bytes:
    """The request as it leaves for the vendor: real credential, one request,
    then close -- so a second request pipelined behind this one on the same
    connection is never forwarded with whatever credential it carries."""
    out = [f"{method} {path} HTTP/1.1", f"Host: {upstream}"]
    out += [f"{n}: {v}" for n, v in headers if n.lower() not in _DROP]
    # an API key goes in x-api-key; an OAuth seat token is a bearer
    out.append(f"x-api-key: {real}" if real.startswith("sk-ant-api")
               else f"Authorization: Bearer {real}")
    if account_id:
        out.append(f"chatgpt-account-id: {account_id}")
    out += [f"Content-Length: {len(body)}", "Connection: close"]
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body


async def _read_body(reader, headers) -> bytes:
    h = {n.lower(): v.strip() for n, v in headers}
    if "chunked" in h.get("transfer-encoding", "").lower():
        body = b""
        while True:
            size = int((await reader.readuntil(b"\r\n")).split(b";")[0], 16)
            if size == 0:
                while (await reader.readuntil(b"\r\n")) != b"\r\n":
                    pass  # trailers
                return body
            body += await reader.readexactly(size)
            await reader.readexactly(2)
    return await reader.readexactly(int(h.get("content-length", "0") or 0))


async def _inject(reader, writer):
    peer = _peer(writer)
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        lines = head.decode("latin-1").split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        headers = [tuple(p.strip() for p in line.split(":", 1))
                   for line in lines[1:] if ":" in line]
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError,
            asyncio.LimitOverrunError, ValueError):
        writer.close()
        return
    route, _, rest = target.lstrip("/").partition("/")
    upstream = ROUTES.get(route)
    if not upstream:
        print(f"deny inject {route!r} from {peer}: no such route", flush=True)
        return await _deny(writer, f"no injector route {route!r}\n")
    want = sentinels().get(route)
    if not want:
        print(f"deny inject {route} from {peer}: no sentinel seed on this proxy", flush=True)
        return await _deny(writer, "this proxy has no sandbox sentinel\n")
    if not sentinel_only(headers, want):
        where = f"{method} {upstream}/{rest.split('?')[0]} from {peer}"
        if not any(n.lower() in ("authorization", "x-api-key") for n, _ in headers):
            # Codex probes a plugin server with no credential at all. Nothing is
            # forwarded and nothing is wrong with the allowlist, so this is a
            # plain 401 -- no egress marker, which heart would read as a
            # misconfigured proxy and stop the run over.
            print(f"refuse inject {where}: no credential", flush=True)
            writer.write(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        print(f"deny inject {where}: foreign credential", flush=True)
        return await _deny(writer, "only the sandbox credential is accepted here\n")
    real = secret(route)
    if not real:
        print(f"deny inject {route} from {peer}: no secret mounted", flush=True)
        return await _deny(writer, f"no credential for {route} on this proxy\n")
    try:
        body = await asyncio.wait_for(_read_body(reader, headers), timeout=120)
        up_r, up_w = await asyncio.open_connection(
            upstream, 443, ssl=ssl.create_default_context(), server_hostname=upstream)
    except (OSError, ValueError, asyncio.TimeoutError, asyncio.IncompleteReadError):
        writer.close()
        return
    up_w.write(rewrite(method, "/" + rest, headers, body, upstream, real, account(route)))
    await up_w.drain()
    print(f"inject {method} {upstream}/{rest.split('?')[0]} from {peer}", flush=True)
    await _pipe(up_r, writer)  # response only: nothing more is read from the agent
    up_w.close()


async def main():
    if not ALLOW and not INJECT_PORT:
        sys.exit("ALLOW is empty: refusing to start a proxy that permits nothing "
                 "-- an agent would fail with no explanation")
    servers = [await asyncio.start_server(_handle, "0.0.0.0", PORT)]
    print(f"egress proxy on :{PORT}, allowing {', '.join(ALLOW) or '(nothing)'}"
          + (f", denying {', '.join(DENY)}" if DENY else ""), flush=True)
    if INJECT_PORT:
        servers.append(await asyncio.start_server(_inject, "0.0.0.0", INJECT_PORT))
        have = [r for r in ROUTES if secret(r)]
        print(f"credential injector on :{INJECT_PORT}, routes with a secret: "
              f"{', '.join(have) or '(none)'}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
