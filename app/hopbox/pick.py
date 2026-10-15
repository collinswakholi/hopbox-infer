"""Native Windows folder dialogs."""
from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path

IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SELECT_NAME = "Select this folder"
CREATE_NO_WINDOW = 0x08000000
_ROOT = Path(__file__).resolve().parent
_CS = _ROOT / "native_shell.cs"

_proc_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def interpret_selection(raw: str) -> dict:
    path = Path(str(raw).strip().strip('"'))
    if path.name == SELECT_NAME:
        folder = path.parent
        if not folder.is_dir():
            raise FileNotFoundError(str(folder))
        return {"path": str(folder), "folder": str(folder), "kind": "folder"}
    if path.is_file():
        if path.suffix.lower() not in IMAGES:
            raise ValueError("Choose a photo or a folder.")
        return {"path": str(path), "folder": str(path.parent), "kind": "image"}
    if path.is_dir():
        return {"path": str(path), "folder": str(path), "kind": "folder"}
    if path.parent.is_dir():
        return {"path": str(path.parent), "folder": str(path.parent), "kind": "folder"}
    raise FileNotFoundError(str(path))


def hopbox_dir() -> Path:
    from hopbox.paths import state_root
    return state_root()


def _startupinfo() -> subprocess.STARTUPINFO:
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0
    return info


def _compiler() -> Path:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for name in ("Framework64", "Framework"):
        csc = windows / "Microsoft.NET" / name / "v4.0.30319" / "csc.exe"
        if csc.is_file():
            return csc
    raise RuntimeError("The .NET compiler was not found.")


def ensure_native() -> Path:
    """Compile the folder-picker library. Returns the DLL path."""
    from hopbox.paths import app_root
    built=app_root()/'native/HopBoxNative.dll'
    if built.is_file():return built
    if os.environ.get('HOPBOX_PACKAGED') == '1':
        raise FileNotFoundError('The native folder picker is missing. Reinstall the complete HopBox Infer package.')
    folder = hopbox_dir()
    dll = folder / "HopBoxNative.dll"
    source = _CS.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    stamp = folder / "HopBoxNative.sha256"
    if dll.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
        return dll
    csc = _compiler()
    command = [
        str(csc), "/nologo", "/t:library", f"/out:{dll}", "/optimize+",
        "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", str(_CS),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )
    if completed.returncode != 0:
        message = (completed.stdout or completed.stderr or "Native compile failed.").strip()
        raise RuntimeError(message[-500:])
    stamp.write_text(digest, encoding="utf-8")
    return dll


def cancel_native() -> None:
    with _proc_lock:
        proc = _proc
    if proc is None or proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _run(script: str, extra: dict | None = None, timeout: int = 300) -> str:
    global _proc
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not powershell.is_file():
        raise RuntimeError("Windows PowerShell was not found in the Windows system folder.")
    env = os.environ.copy()
    if extra:
        env.update({key: "" if value is None else str(value) for key, value in extra.items()})
    proc = subprocess.Popen(
        [str(powershell), "-NoProfile", "-STA", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=CREATE_NO_WINDOW,
        startupinfo=_startupinfo(),
    )
    with _proc_lock:
        _proc = proc
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError("The folder window timed out.")
    finally:
        with _proc_lock:
            if _proc is proc:
                _proc = None
    if proc.returncode != 0:
        message = (stderr or stdout or "The folder window did not open.").strip()
        raise RuntimeError(message.splitlines()[-1][:300])
    return (stdout or "").strip()


def _native(method: str, title: str, start: str) -> str:
    if method not in {"PickImport", "PickFolder"}:
        raise ValueError(method)
    dll = ensure_native()
    folder = _existing_dir(start)
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$ProgressPreference = 'SilentlyContinue'\n"
        "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false\n"
        "Add-Type -Path $env:HOPBOX_NATIVE_DLL\n"
        f"$result = [HopBoxNative.Dialogs]::{method}($env:HOPBOX_PICK_TITLE, $env:HOPBOX_PICK_START)\n"
        "if ([string]::IsNullOrEmpty($result)) { exit 0 }\n"
        "[Console]::Out.Write($result)\n"
    )
    return _run(script, {
        "HOPBOX_NATIVE_DLL": str(dll),
        "HOPBOX_PICK_TITLE": title,
        "HOPBOX_PICK_START": folder,
    })


def _existing_dir(start: str) -> str:
    if start:
        folder = Path(start)
        if folder.is_file():
            folder = folder.parent
        if folder.is_dir():
            return str(folder)
    desktop = Path.home() / "Desktop"
    return str(desktop if desktop.is_dir() else Path.home())


def pick_output(start: str = "") -> dict | None:
    chosen = _native("PickFolder", "Select the output folder", start)
    if not chosen:
        return None
    folder = Path(chosen.splitlines()[0].strip())
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    return {"path": str(folder), "folder": str(folder), "kind": "folder"}


def pick_import(start: str = "") -> dict | None:
    text = _native("PickImport", "Import", start)
    if not text:
        return None
    paths = [line.strip() for line in text.splitlines() if line.strip()]
    if not paths:
        return None
    return {"paths": paths, "kind": "selection"}


def pick(purpose: str, start: str = "") -> dict | None:
    if purpose == "output":
        return pick_output(start)
    if purpose == "model":
        return _pick_model(start)
    return pick_import(start)


def _pick_model(start: str = "") -> dict | None:
    initial = ""
    from hopbox.paths import app_root
    folder = Path(start) if start else app_root() / "models"
    if folder.is_file():
        folder = folder.parent
    if folder.is_dir():
        initial = f"$d.InitialDirectory = '{str(folder).replace(chr(39), chr(39) * 2)}'"
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = 'Select a detection model'
$d.Filter = 'Detection models (*.onnx;*.pt)|*.onnx;*.pt'
$d.CheckFileExists = $true
{initial}
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
if ($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {{
  [Console]::Out.WriteLine($d.FileName)
}}
$owner.Dispose()
"""
    chosen = _run(script)
    if not chosen:
        return None
    path = Path(chosen.strip())
    if path.suffix.lower() not in {".onnx",".pt"} or not path.is_file():
        raise ValueError("Choose an ONNX model bundle or a .pt checkpoint.")
    return {"path": str(path), "kind": "model"}
