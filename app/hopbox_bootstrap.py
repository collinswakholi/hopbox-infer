"""Open diagnostics before importing any application or native dependency."""
from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import platform
import runpy
import sys
import traceback


def main() -> None:
    destination = os.environ.get("HOPBOX_STARTUP_LOG")
    if not destination:
        raise RuntimeError("Start the packaged app through HopBox-Infer.exe.")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This handle belongs to the server, so diagnostics continue after the
    # splash process exits. Redirected pipes owned by the splash would not.
    log = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    faulthandler.enable(file=log, all_threads=True)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Windows: {platform.platform()}")
    print(f"CPU: {os.environ.get('PROCESSOR_IDENTIFIER', platform.machine())}")
    print(f"Device request: {os.environ.get('HOPBOX_DEVICE', 'auto')}")
    print("Importing application; native fault diagnostics are enabled.")
    try:
        runpy.run_module("hopbox_entry", run_name="__main__")
    except BaseException:
        traceback.print_exc(file=log)
        log.flush()
        raise


if __name__ == "__main__":
    main()
