"""heart observe (was heart pulse, A6): the generic timeline and tail are the
pulse CLI's; heart hands off when pulse is installed and reads in-process
when it isn't."""
from unittest.mock import patch

import pytest

from heart import cli


class Exec(Exception):
    pass


def _execv(path, argv):
    raise Exec(argv)


def test_episode_hands_off_to_pulse_events_text():
    with patch.object(cli.shutil, "which", return_value="/bin/pulse"), \
            patch.object(cli.os, "execv", _execv), pytest.raises(Exec) as ran:
        cli.main(["observe", "episode", "ep-1"])
    assert ran.value.args[0] == ["/bin/pulse", "events", "--episode", "ep-1", "--text"]


def test_tail_prints_history_then_hands_off_with_its_filters():
    with patch.object(cli.shutil, "which", return_value="/bin/pulse"), \
            patch.object(cli.pulse_mod, "tail") as history, \
            patch.object(cli.os, "execv", _execv), pytest.raises(Exec) as ran:
        cli.main(["pulse", "tail", "--source", "heart"])  # the old name still works
    assert history.call_args.kwargs["live"] is False
    assert ran.value.args[0] == ["/bin/pulse", "tail", "--source", "heart", "--text"]


def test_without_pulse_the_timeline_is_read_in_process(capsys):
    with patch.object(cli.shutil, "which", return_value=None), \
            patch.object(cli.pulse_mod, "episode_timeline", return_value=["line"]):
        assert cli.main(["observe", "episode", "ep-1"]) == 0
    assert capsys.readouterr().out == "line\n"


def test_the_old_module_name_still_imports():
    # replaying old plexus commits imports heart.pulse (F28)
    from heart.pulse import load_events
    from heart.observe import load_events as current
    assert load_events is current
