"""Record the current server and close HopBox browser windows on Exit."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from hopbox.branding import BRANDING
from hopbox.paths import state_root

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _pid_file() -> Path:
    root = state_root()
    return root / "servers.txt"


def register_instance() -> None:
    """Record this server for local diagnostics; never terminate other PIDs."""
    _pid_file().write_text(str(os.getpid()), encoding="utf-8")


def close_hopbox_browser_windows() -> int:
    """Close browser windows whose title is the HopBox page.

    Chrome and Edge app windows use class Chrome_WidgetWin_1 and the page title
    HopBox. A normal single tab is titled "HopBox - Google Chrome" or
    "HopBox - Microsoft Edge". Cursor's own window title contains Cursor, so
    the editor is left alone. A browser view titled exactly HopBox is closed
    even when its process is Cursor.
    """
    if os.name != "nt":
        return 0
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    WM_CLOSE = 0x0010
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    user32.EnumWindows.argtypes = [enum_proc, ctypes.c_void_p]
    user32.EnumWindows.restype = ctypes.c_bool
    user32.EnumChildWindows.argtypes = [ctypes.c_void_p, enum_proc, ctypes.c_void_p]
    user32.EnumChildWindows.restype = ctypes.c_bool
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
    user32.IsWindow.argtypes = [ctypes.c_void_p]
    user32.IsWindow.restype = ctypes.c_bool
    user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    user32.PostMessageW.restype = ctypes.c_bool
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    kernel32.QueryFullProcessImageNameW.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.QueryFullProcessImageNameW.restype = ctypes.c_bool

    def exe_name(pid: int) -> str:
        if pid <= 0:
            return ""
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = ctypes.c_ulong(512)
            buf = ctypes.create_unicode_buffer(512)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return ""
            return Path(buf.value).name
        finally:
            kernel32.CloseHandle(handle)

    def title_matches(title: str) -> bool:
        folded = " ".join(title.split()).casefold()
        if not folded or "cursor" in folded or "hopboxdrop" in folded or "hopbox_v2" in folded:
            return False
        if folded == "hopbox":
            return True
        product = BRANDING['name'].casefold()
        if folded == product or any(folded == product + suffix for suffix in (' - google chrome', ' - chromium', ' - microsoft edge', ' - edge', ' - chrome')):
            return True
        return folded in {
            "hopbox - google chrome",
            "hopbox - chromium",
            "hopbox - microsoft edge",
            "hopbox - edge",
            "hopbox - chrome",
        }

    found: list[int] = []

    def consider(hwnd) -> None:
        if not hwnd:
            return
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0 or length > 400:
            return
        title_buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buf, length + 1)
        title = title_buf.value or ""
        if not title_matches(title):
            return
        class_buf = ctypes.create_unicode_buffer(256)
        if user32.GetClassNameW(hwnd, class_buf, 256) <= 0:
            return
        if "Chrome_WidgetWin" not in (class_buf.value or ""):
            return
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = exe_name(int(pid.value)).casefold()
        if name in {"cursor.exe", "code.exe"} and title.strip().casefold() != "hopbox":
            return
        value = int(hwnd)
        if value not in found:
            found.append(value)

    def visit_child(hwnd, _lparam):
        try:
            consider(hwnd)
        except Exception:
            pass
        return True

    child_callback = enum_proc(visit_child)

    def visit(hwnd, _lparam):
        try:
            consider(hwnd)
            user32.EnumChildWindows(hwnd, child_callback, 0)
        except Exception:
            pass
        return True

    user32.EnumWindows(enum_proc(visit), 0)
    closed = 0
    for hwnd in found:
        pointer = ctypes.c_void_p(hwnd)
        if user32.IsWindow(pointer) and user32.PostMessageW(pointer, WM_CLOSE, 0, 0):
            closed += 1
    return closed
