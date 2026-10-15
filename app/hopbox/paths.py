"""Locations that stay valid both in a checkout and in a frozen executable."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Read-only application assets; never store settings or caches here."""
    if os.environ.get('HOPBOX_APP_ROOT'):
        return Path(os.environ['HOPBOX_APP_ROOT']).resolve()
    if frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundle_root() -> Path:
    """Folder that holds packaged code and static files."""
    if os.environ.get('HOPBOX_APP_ROOT'):
        return app_root()
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def state_root() -> Path:
    """Writable per-user data, independent of the installation destination."""
    configured = os.environ.get('HOPBOX_STATE_ROOT')
    root = Path(configured) if configured else Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'HopBox-Infer'
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
