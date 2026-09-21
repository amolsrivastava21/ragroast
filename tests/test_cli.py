"""CLI smoke tests: bare invocation shows help (not an error), --version works."""
from __future__ import annotations

import pytest

from ragroast.cli import main


def test_bare_command_shows_help_not_error(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    for cmd in ("demo", "run", "score"):
        assert cmd in out


def test_version_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
