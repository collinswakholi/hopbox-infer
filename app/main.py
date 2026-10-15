"""HopBox in the browser. Run with the isolated build environment:

    ..\\.build-env\\Scripts\\python.exe main.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(os.environ['HOPBOX_APP_ROOT']) if os.environ.get('HOPBOX_APP_ROOT') else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rthook_stdio
from hopbox.web import create_app

HOST = "127.0.0.1"
PORT = int(os.environ.get('HOPBOX_PORT','8765'))


def _browser_executable() -> Path | None:
    """Chrome or Edge, so HopBox can open in a window Exit is able to close."""
    local = os.environ.get("LOCALAPPDATA", "")
    program = os.environ.get("ProgramFiles", "")
    program86 = os.environ.get("ProgramFiles(x86)", "")
    candidates = [
        Path(local) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(program) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(program86) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(program) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(program86) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def open_browser(url: str) -> None:
    """Open an app window titled HopBox. Exit closes that window by its title."""
    exe = _browser_executable()
    if exe is not None:
        subprocess.Popen(
            [str(exe), f"--app={url}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    webbrowser.open(url)


def main() -> int:
    app = create_app()
    url = f"http://{HOST}:{PORT}"
    if os.environ.get('HOPBOX_NO_BROWSER')!='1':
        def when_ready():
            # The native splash remains visible during model initialization.
            # Never open an empty browser while startup is still running.
            ready = app.config['HOPBOX_READY']
            while not ready['ok'] and not ready.get('failed'):
                time.sleep(0.1)
            if ready['ok']:
                time.sleep(0.7)
                open_browser(url)
        threading.Thread(target=when_ready, daemon=True).start()
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
