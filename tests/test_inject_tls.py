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
