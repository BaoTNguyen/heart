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
