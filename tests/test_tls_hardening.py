import ssl
from pathlib import Path

import heart.sandbox as sb


def _seeded_env(monkeypatch, tmp_path):
    # primed like tests/test_sandbox.py: a sentinel seed under XDG_CONFIG_HOME
    monkeypatch.setenv("HEART_WS_ROOT", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "heart" / "secrets").mkdir(parents=True)
    (tmp_path / "heart" / "secrets" / "sentinel").write_text("s33d")
    monkeypatch.setenv("HEART_SANDBOX_INJECT", "chatgpt")


def test_bundle_refused_without_system_roots(monkeypatch, tmp_path, capsys):
    _seeded_env(monkeypatch, tmp_path)
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"-----BEGIN CERTIFICATE-----\nCA\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("HEART_SANDBOX_INJECT_TLS_PORT", "8443")
    monkeypatch.setenv("HEART_SANDBOX_CA_CERT", str(ca))
    monkeypatch.setattr(sb, "_system_roots", lambda: None)

    # HEART_WS_ROOT=tmp_path puts the bundle in the shared parent: compare, don't assume absent
    bundle = tmp_path.parent / "heart-sentinel" / "ca-bundle.pem"
    before = bundle.exists() and bundle.stat().st_ino

    env = sb.inject_env("proxy")
    assert "SSL_CERT_FILE" not in env
    assert env["HEART_CODEX_BASE"].startswith("http://")
    assert "no system CA bundle" in capsys.readouterr().err
    assert not [m for m in sb.codex_sentinel_mounts() if m.target == sb.CA_BUNDLE_IN_CONTAINER]
    assert (bundle.exists() and bundle.stat().st_ino) == before


def test_bundle_atomic_replace(monkeypatch, tmp_path):
    monkeypatch.setenv("HEART_WS_ROOT", str(tmp_path / "ws"))
    roots = tmp_path / "roots.pem"
    roots.write_bytes(b"ROOTS\n")
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"PROXY-CA\n")
    monkeypatch.setattr(sb, "_system_roots", lambda: roots)

    first = sb._ca_bundle(ca)
    ino = first.stat().st_ino
    second = sb._ca_bundle(ca)
    assert second == first
    body = second.read_bytes()
    assert b"ROOTS" in body and b"PROXY-CA" in body
    assert second.stat().st_ino != ino
    assert not list(second.parent.glob("*.tmp"))


def test_bundle_found_via_default_verify_paths(monkeypatch, tmp_path):
    cafile = tmp_path / "cafile.pem"
    cafile.write_bytes(b"ROOTS\n")
    monkeypatch.setattr(sb, "_ROOTS_CANDIDATES", (str(tmp_path / "absent.pem"),))
    real = ssl.get_default_verify_paths()
    monkeypatch.setattr(ssl, "get_default_verify_paths",
                        lambda: real._replace(cafile=str(cafile)))
    assert sb._system_roots() == cafile


def _proxy_module():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "contrib" / "egress-proxy.py"
    spec = importlib.util.spec_from_file_location("egress_proxy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _garbage_cert(tmp_path):
    (tmp_path / "proxy.pem").write_text("not a certificate\n")
    (tmp_path / "proxy.key").write_text("not a key\n")


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_proxy_garbage_cert_tls_off(monkeypatch, tmp_path, capsys):
    proxy = _proxy_module()
    _garbage_cert(tmp_path)
    monkeypatch.setattr(proxy, "TLS_DIR", tmp_path)
    monkeypatch.setattr(proxy, "INJECT_TLS_PORT", 8443)
    assert proxy.tls_context() is None
    assert "credential injector (TLS) not started:" in capsys.readouterr().out


def test_proxy_garbage_cert_still_serves_plain(monkeypatch, tmp_path):
    import asyncio
    proxy = _proxy_module()
    _garbage_cert(tmp_path)
    port, inject = _free_port(), _free_port()
    monkeypatch.setattr(proxy, "TLS_DIR", tmp_path)
    monkeypatch.setattr(proxy, "SECRETS", str(tmp_path))
    monkeypatch.setattr(proxy, "ALLOW", ("example.com",))
    monkeypatch.setattr(proxy, "PORT", port)
    monkeypatch.setattr(proxy, "INJECT_PORT", inject)
    monkeypatch.setattr(proxy, "INJECT_TLS_PORT", _free_port())

    async def run():
        task = asyncio.create_task(proxy.main())
        try:
            for p in (port, inject):
                for _ in range(100):
                    try:
                        _, w = await asyncio.open_connection("127.0.0.1", p)
                        break
                    except OSError:
                        if task.done():
                            task.result()
                        await asyncio.sleep(0.02)
                else:
                    raise AssertionError(f"nothing listening on {p}")
                w.close()
            assert not proxy.TLS_ON
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_proxy_tls_port_without_cert_keeps_allowlist(monkeypatch, tmp_path):
    proxy = _proxy_module()
    (tmp_path / "anthropic").write_text("sk-ant-oat01-real\n")
    monkeypatch.setattr(proxy, "SECRETS", str(tmp_path))
    monkeypatch.setattr(proxy, "TLS_DIR", tmp_path / "tls")
    monkeypatch.setattr(proxy, "ALLOW", ("api.anthropic.com",))
    monkeypatch.setattr(proxy, "INJECT_PORT", 0)
    monkeypatch.setattr(proxy, "INJECT_TLS_PORT", 8443)
    assert "api.anthropic.com" in proxy._injected_hosts()
    assert proxy.tls_context() is None
    assert proxy.permitted("api.anthropic.com", 443)


def test_empty_override_ws_root(monkeypatch, tmp_path):
    from heart import env
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
    monkeypatch.setenv("HEART_WS_ROOT", "")
    got = env._ws_root()
    assert got == Path("")
    assert not str(got).startswith(str(tmp_path))
    monkeypatch.delenv("HEART_WS_ROOT")
    assert env._ws_root() == env.vascular_paths.path("cache", "heart", "ws")
    assert tmp_path in env._ws_root().parents


def test_empty_override_stats_path(monkeypatch, tmp_path):
    from heart import routing
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
    monkeypatch.setenv("HEART_ROUTE_STATS", "")
    got = routing._stats_path()
    assert got == Path("")
    assert not str(got).startswith(str(tmp_path))
    monkeypatch.delenv("HEART_ROUTE_STATS")
    assert routing._stats_path() == routing.vascular_paths.path("state", "heart", "route_stats.json")
    assert tmp_path in routing._stats_path().parents
