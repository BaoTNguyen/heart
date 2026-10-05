import importlib.util
from pathlib import Path


def _proxy_module():
    path = Path(__file__).resolve().parent.parent / "contrib" / "egress-proxy.py"
    spec = importlib.util.spec_from_file_location("egress_proxy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_proxy_tls_context_without_cert_logs_and_returns_none(monkeypatch, tmp_path, capsys):
    proxy = _proxy_module()
    monkeypatch.setattr(proxy, "INJECT_TLS_PORT", 8443)
    monkeypatch.setattr(proxy, "TLS_DIR", tmp_path)
    assert proxy.tls_context() is None
    assert "proxy.pem" in capsys.readouterr().out


def test_proxy_tls_context_off_is_silent(monkeypatch, capsys):
    proxy = _proxy_module()
    monkeypatch.setattr(proxy, "INJECT_TLS_PORT", 0)
    assert proxy.tls_context() is None
    assert capsys.readouterr().out == ""


def _sandbox_env(monkeypatch, tmp_path):
    import heart.sandbox as sb
    monkeypatch.setenv("HEART_WS_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
    (tmp_path / "config" / "heart" / "secrets").mkdir(parents=True)
    (tmp_path / "config" / "heart" / "secrets" / "sentinel").write_text("s33d")
    monkeypatch.setenv("HEART_SANDBOX_INJECT", "chatgpt")
    monkeypatch.delenv("HEART_SANDBOX_INJECT_TLS_PORT", raising=False)
    monkeypatch.delenv("HEART_SANDBOX_CA_CERT", raising=False)
    return sb


def test_sandbox_inject_without_tls_port_stays_http(monkeypatch, tmp_path):
    sb = _sandbox_env(monkeypatch, tmp_path)
    env = sb.inject_env("egress-web")
    assert env["HEART_CODEX_BASE"].startswith("http://")
    assert "SSL_CERT_FILE" not in env


def test_sandbox_inject_tls_uses_https_and_mounts_the_ca_bundle(monkeypatch, tmp_path):
    sb = _sandbox_env(monkeypatch, tmp_path)
    marker = b"-----BEGIN CERTIFICATE-----\nHEART-TEST-MARKER\n-----END CERTIFICATE-----\n"
    (tmp_path / "ca.pem").write_bytes(marker)
    monkeypatch.setenv("HEART_SANDBOX_INJECT_TLS_PORT", "8443")
    monkeypatch.setenv("HEART_SANDBOX_CA_CERT", str(tmp_path / "ca.pem"))
    env = sb.inject_env("egress-web")
    assert env["HEART_CODEX_BASE"].startswith("https://egress-web:8443/")
    assert env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"].startswith("https://egress-web:8443/")
    assert env["SSL_CERT_FILE"] == sb.CA_BUNDLE_IN_CONTAINER
    mounts = sb.codex_sentinel_mounts()
    (bundle,) = [m for m in mounts if m.target == sb.CA_BUNDLE_IN_CONTAINER]
    assert not bundle.writable and marker in Path(bundle.source).read_bytes()
    assert not any(m.source.endswith(".key") for m in mounts)


def test_sandbox_inject_tls_with_missing_ca_stays_http(monkeypatch, tmp_path):
    sb = _sandbox_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HEART_SANDBOX_INJECT_TLS_PORT", "8443")
    monkeypatch.setenv("HEART_SANDBOX_CA_CERT", str(tmp_path / "missing.pem"))
    env = sb.inject_env("egress-web")
    assert env["HEART_CODEX_BASE"].startswith("http://")
    assert "SSL_CERT_FILE" not in env
