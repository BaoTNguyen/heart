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
      -e ALLOW=api.anthropic.com,host.docker.internal \
      -v $PWD/contrib/egress-proxy.py:/proxy.py:ro \
      --entrypoint python3 heart-agent:latest /proxy.py
    docker network connect heart-egress egress

    HEART_API_NETWORK=heart-egress HEART_SANDBOX_PROXY=http://egress:8888 \
      heart run task.json --agent claude:sonnet    # task asks for network "api"

CONNECT is tunnelled after the host check, never intercepted -- no TLS
termination, no certificate to inject, and the proxy sees hostnames rather than
traffic. Plain HTTP is forwarded by absolute-URI so the same allowlist covers a
local model server on http://.

A name in ALLOW covers that host and its subdomains: `anthropic.com` allows
`api.anthropic.com`. Matching is on the name the client asked for, which is the
point -- an agent that resolves a name itself and connects by IP has no route to
do it.

This replaces the raw TCP forwarder that used to live beside it: a local model
on http:// goes through the same allowlist as a vendor API on https://, so
there is one container and one list rather than two of each. Verified both
ways -- `allow POST host.docker.internal` for llama-server, `allow CONNECT
api.anthropic.com` for the Claude CLI, both episodes passing.

The log is worth reading. Claude Code also reaches for mcp-proxy.anthropic.com
and http-intake.logs.us5.datadoghq.com; under plain bridge egress all of that
leaves the machine unremarked, and the episodes pass without it.

ponytail: no auth, no logging to disk, no upstream chaining. Add those when
something needs them. The allowlist is the whole feature.
"""
import asyncio
import os
import sys

ALLOW = tuple(h.strip().lower() for h in os.environ.get("ALLOW", "").split(",") if h.strip())
PORT = int(os.environ.get("PORT", "8888"))


def permitted(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in ALLOW)


async def _pipe(reader, writer):
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def _splice(reader, writer, up_r, up_w):
    await asyncio.gather(_pipe(reader, up_w), _pipe(up_r, writer),
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


async def _handle(reader, writer):
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
        host, _, port = target.partition(":")
        if not permitted(host):
            print(f"deny CONNECT {host}", flush=True)
            return await _deny(writer, f"{host} is not in the sandbox allowlist\n")
        # drain the rest of the request head before tunnelling
        try:
            while (await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=30)) != b"\r\n":
                pass
            up_r, up_w = await asyncio.open_connection(host, int(port or 443))
        except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):
            writer.close()
            return
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        print(f"allow CONNECT {host}", flush=True)
        return await _splice(reader, writer, up_r, up_w)

    # plain HTTP arrives as an absolute URI: GET http://host:port/path HTTP/1.1
    if not target.startswith("http://"):
        return await _deny(writer, "only proxied requests are accepted\n")
    rest = target[len("http://"):]
    authority, slash, path = rest.partition("/")
    host, _, port = authority.partition(":")
    if not permitted(host):
        print(f"deny {method} {host}", flush=True)
        return await _deny(writer, f"{host} is not in the sandbox allowlist\n")
    try:
        up_r, up_w = await asyncio.open_connection(host, int(port or 80))
    except OSError:
        writer.close()
        return
    up_w.write(f"{method} /{path} HTTP/1.1\r\n".encode("latin-1"))
    await up_w.drain()
    print(f"allow {method} {host}", flush=True)
    await _splice(reader, writer, up_r, up_w)


async def main():
    if not ALLOW:
        sys.exit("ALLOW is empty: refusing to start a proxy that permits nothing "
                 "-- an agent would fail with no explanation")
    server = await asyncio.start_server(_handle, "0.0.0.0", PORT)
    print(f"egress proxy on :{PORT}, allowing {', '.join(ALLOW)}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
