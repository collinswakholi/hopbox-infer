"""Capture diagnostics when the private windowed interpreter has no console."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path



def _log_path() -> Path:
    from hopbox.paths import state_root
    return state_root() / "HopBox.log"


def _open_log():
    return open(_log_path(), "a", encoding="utf-8", buffering=1)


class _LogStream:
    def __init__(self, handle) -> None:
        self._handle = handle

    def write(self, data) -> int:
        try:
            self._handle.write(data)
            self._handle.flush()
        except Exception:
            pass
        return len(data) if isinstance(data, str) else 0

    def flush(self) -> None:
        try:
            self._handle.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        try:
            return self._handle.fileno()
        except Exception:
            return -1


if sys.stdout is None or sys.stderr is None:
    _handle = _open_log()
    if sys.stdout is None:
        sys.stdout = _LogStream(_handle)
    if sys.stderr is None:
        sys.stderr = _LogStream(_handle)


def _excepthook(exc_type, exc, tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        sys.stderr.write(text + "\n")
    except Exception:
        try:
            _log_path().open("a", encoding="utf-8").write(text + "\n")
        except Exception:
            pass


sys.excepthook = _excepthook
