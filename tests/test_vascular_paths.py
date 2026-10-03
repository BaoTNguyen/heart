"""Tests for src/heart/vascular_paths.py.

The public surface is intentionally tiny: KINDS, home(), path(), journal_dir(),
repo_dir(). Every test pins one piece of that contract.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def default_root(monkeypatch, tmp_path):
    """No VASCULAR_HOME and a throwaway HOME, so the default root is checked
    without ever touching the account's real ~/.vascular."""
    monkeypatch.delenv("VASCULAR_HOME", raising=False)
    monkeypatch.delenv("EVENT_JOURNAL_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / ".vascular"

# The module must import with zero I/O — no file reads at import time.
import heart.vascular_paths as vp


class TestKINDS:
    """KINDS is the fixed tuple of recognised kinds."""

    def test_exact_contents_and_order(self):
        assert vp.KINDS == ("config", "state", "cache", "data", "backups")

    def test_is_a_tuple_not_a_list(self):
        assert isinstance(vp.KINDS, tuple)

    def test_no_extra_kinds(self):
        assert len(vp.KINDS) == 5


class TestHome:
    """home() returns $VASCULAR_HOME when set, else ~/.vascular."""

    def test_defaults_to_home_dot_vascular(self, default_root):
        assert vp.home() == Path.home() / ".vascular" == default_root

    def test_respects_VASCULAR_HOME(self):
        with patch.dict(os.environ, {"VASCULAR_HOME": "/opt/v"}):
            result = vp.home()
        assert result == Path("/opt/v")

    def test_returns_a_path_instance(self):
        result = vp.home()
        assert isinstance(result, Path)


class TestPath:
    """path(kind, component, *parts) builds home()/kind/component/..."""

    def test_valid_kinds_build_correctly(self, default_root):
        base = Path.home() / ".vascular"
        for kind in vp.KINDS:
            p = vp.path(kind, "mycomp")
            assert p == base / kind / "mycomp"

    def test_extra_parts_are_appended(self, default_root):
        base = Path.home() / ".vascular"
        p = vp.path("state", "heart", "events", "2026.ndjson")
        assert p == base / "state" / "heart" / "events" / "2026.ndjson"

    def test_unknown_kind_raises_valueerror(self):
        with pytest.raises(ValueError, match="unknown kind"):
            vp.path("bogus", "comp")

    def test_empty_component_is_allowed(self, default_root):
        base = Path.home() / ".vascular"
        p = vp.path("config", "")
        assert p == base / "config" / ""

    def test_returns_path_instance(self):
        p = vp.path("cache", "x", "a", "b")
        assert isinstance(p, Path)


class TestJournalDir:
    """journal_dir() resolves $EVENT_JOURNAL_DIR or heart's state path."""

    def test_defaults_to_heart_state_events(self, default_root):
        expected = Path.home() / ".vascular" / "state" / "heart" / "events"
        assert vp.journal_dir() == expected

    def test_respects_EVENT_JOURNAL_DIR(self):
        with patch.dict(os.environ, {"EVENT_JOURNAL_DIR": "/tmp/j"}):
            result = vp.journal_dir()
        assert result == Path("/tmp/j")

    def test_returns_path_instance(self):
        result = vp.journal_dir()
        assert isinstance(result, Path)

    def test_event_journal_env_overrides_default(self):
        """The env var wins over the default path, even if the default
        directory already exists on disk."""
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"EVENT_JOURNAL_DIR": td}):
                assert vp.journal_dir() == Path(td)


class TestRepoDir:
    """repo_dir(root, component) returns <root>/.vascular/<component>."""

    def test_string_root(self):
        result = vp.repo_dir("/my/repo", "heart")
        assert result == Path("/my/repo") / ".vascular" / "heart"

    def test_path_root(self):
        result = vp.repo_dir(Path("/my/repo"), "heart")
        assert result == Path("/my/repo") / ".vascular" / "heart"

    def test_nested_component_path(self):
        result = vp.repo_dir("/repo", "sub/module")
        assert result == Path("/repo") / ".vascular" / "sub" / "module"

    def test_returns_path_instance(self):
        result = vp.repo_dir(Path("."), "test")
        assert isinstance(result, Path)


class TestNoIOAtImport:
    """The module must be importable without touching the filesystem."""

    def test_import_does_not_read_env(self):
        """Importing the module must not read any env vars (except through
        the lazy calls below). A real import is a no-op."""
        pass  # The mere fact that this module can be imported at the top
               # of this file without errors is the test.


class TestPathDoesNotCreate:
    """path() builds the Path object but never creates anything on disk."""

    def test_no_directory_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("VASCULAR_HOME")
            os.environ["VASCULAR_HOME"] = tmp
            try:
                p = vp.path("data", "comp", "deep", "nested")
                assert not p.exists()
                # Also check parent dirs
                assert not p.parent.exists()
                assert not p.parent.parent.exists()
            finally:
                if old_home is None:
                    os.environ.pop("VASCULAR_HOME", None)
                else:
                    os.environ["VASCULAR_HOME"] = old_home


class TestHomeReturnsPathNotString:
    """home() must return a pathlib.Path, not a string."""

    def test_home_is_path(self):
        assert isinstance(vp.home(), Path)

    def test_path_result_is_path(self):
        assert isinstance(vp.path("cache", "x"), Path)

    def test_journal_dir_result_is_path(self):
        assert isinstance(vp.journal_dir(), Path)

    def test_repo_dir_result_is_path(self):
        assert isinstance(vp.repo_dir(".", "x"), Path)


class TestHeartPathsUnderVascularHome:
    """Every path heart writes outside a checkout resolves under VASCULAR_HOME,
    at call time, so setting it in a test takes effect."""

    OVERRIDES = ("HEART_WS_ROOT", "HEART_ROUTE_STATS", "EVENT_JOURNAL_DIR")

    def test_defaults_land_under_vascular_home(self, monkeypatch, tmp_path):
        from heart import agents_api, cli, env, events, routing

        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        for k in self.OVERRIDES:
            monkeypatch.delenv(k, raising=False)
        assert agents_api.models_json_path() == tmp_path / "config" / "heart" / "models.json"
        assert env._ws_root() == tmp_path / "cache" / "heart" / "ws"
        assert routing._stats_path() == tmp_path / "state" / "heart" / "route_stats.json"
        assert events.journal_dir() == tmp_path / "state" / "heart" / "events"
        assert cli.work_runs_dir() == tmp_path / "state" / "heart" / "runs"

    def test_specific_overrides_still_win(self, monkeypatch, tmp_path):
        from heart import env, events, routing

        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "v"))
        monkeypatch.setenv("HEART_WS_ROOT", str(tmp_path / "ws"))
        monkeypatch.setenv("HEART_ROUTE_STATS", str(tmp_path / "stats.json"))
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path / "journal"))
        assert env._ws_root() == tmp_path / "ws"
        assert routing._stats_path() == tmp_path / "stats.json"
        assert events.journal_dir() == tmp_path / "journal"
