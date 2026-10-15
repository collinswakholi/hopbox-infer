"""Serialized client for the native color fitting worker."""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable

__version__ = "1.1.0"



class EngineError(RuntimeError):
    """Failed request; reply retains per-image results for partial batches."""
    def __init__(self, message: str, reply: dict | None = None):
        super().__init__(message)
        self.reply = reply


def _root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "refs/neutrals.json").is_file() and (candidate / "models/opt2").is_dir():
            return candidate
    return None


def _json_default(value: Any) -> str:
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class Engine:
    """Serialized NDJSON worker client. Relative paths use the caller's cwd.

    Explicit executable/root override environment and discovery. Progress runs
    synchronously on the calling thread; callbacks must not issue requests.
    A timeout/protocol error kills the worker; the next request starts a fresh
    session, so persist model paths rather than relying on last-fit state.
    """
    def __init__(self, executable: str | Path | None = None, *,
                 root: str | Path | None = None, policy: str = "cpu",
                 timeout: float = 600, progress: Callable[[str], None] | None = None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout, self.policy, self.progress = timeout, policy, progress
        exe = executable or os.environ.get("CCP_ENGINE_EXE") or os.environ.get("COLORCORRECT_EXE")
        located = Path(root or os.environ.get("CCP_NEW_UI_ROOT", "")).resolve() if root or os.environ.get("CCP_NEW_UI_ROOT") else None
        self.root = located or (_root(Path(exe).resolve().parent) if exe else None) or _root(Path(__file__).resolve().parent) or _root(Path.cwd())
        if self.root is None or not (self.root / "refs/neutrals.json").is_file():
            raise FileNotFoundError("Set root= or CCP_NEW_UI_ROOT to the app folder containing models/ and refs/.")
        if exe:
            self.executable = Path(exe).resolve()
        else:
            name = "ccp-engine.exe" if os.name == "nt" else "ccp-engine"
            candidates = [self.root / name, self.root / "engine/target/release" / name,
                          self.root / "dist/win-cpu" / name,
                          self.root / "dist/linux-x64/payload" / name,
                          self.root / "dist/macos-arm64/payload" / name]
            self.executable = next((p for p in candidates if p.is_file()), candidates[0])
        if not self.executable.is_file():
            raise FileNotFoundError(f"Worker not found: {self.executable}. Use an extracted app package or build the engine.")
        self._process: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=80)
        self._readers: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.ready: dict = {}

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in ("LIBTORCH", "LIBTORCH_PATH", "CCP_LIBTORCH_ROOT"):
            env.pop(name, None)
        env["CCP_NEW_UI_ROOT"] = str(self.root)
        env["PATH"] = os.pathsep.join([str(self.executable.parent), str(Path(env.get("SystemRoot", "C:/Windows")) / "System32")])
        return env

    def _start(self) -> None:
        if self._process and self._process.poll() is None:
            return
        self._stop()
        self._lines = queue.Queue()
        self._stderr.clear()
        self._process = subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=self._environment(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        # Capture each process's queue, so a retiring reader cannot poison a new session.
        def read_stdout(pipe, lines):
            try:
                for line in pipe:
                    lines.put(line)
            finally:
                lines.put(None)
        def read_stderr(pipe):
            for line in pipe:
                self._stderr.append(line.rstrip())
        self._readers = [threading.Thread(target=read_stdout, args=(self._process.stdout, self._lines), daemon=True),
                         threading.Thread(target=read_stderr, args=(self._process.stderr,), daemon=True)]
        for reader in self._readers:
            reader.start()
        try:
            self.ready = self._read_reply(self.timeout)
            if self.ready.get("event") != "ready":
                raise EngineError("Worker did not send a ready event", self.ready)
        except BaseException:
            self._stop()
            raise

    def _read_reply(self, timeout: float, *, report_progress: bool = True) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._lines.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError(f"ColorCorrect worker exceeded {timeout:g}s; session terminated") from exc
            if line is None:
                raise EngineError("Worker exited unexpectedly. " + "\n".join(self._stderr))
            try:
                reply = json.loads(line)
            except ValueError as exc:
                raise EngineError(f"Invalid worker JSON: {line[:200]}") from exc
            if not isinstance(reply, dict):
                raise EngineError("Worker returned a non-object reply")
            if reply.get("type") == "progress":
                if self.progress and report_progress:
                    self.progress(str(reply.get("message", "")))
                continue
            return reply

    def request(self, command: str, *, timeout: float | None = None,
                check: bool = True, **parameters: Any) -> dict:
        """Send any documented operation. check=True raises even for partial batches."""
        duration = self.timeout if timeout is None else timeout
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("timeout must be finite and positive")
        if "cmd" in parameters or "command" in parameters:
            raise ValueError("Pass the operation as the first argument, not cmd/command")
        if parameters.get("do_ffc") and not parameters.get("white"):
            raise ValueError("FFC requires an explicit white image (Blank.JPG in the example corpus)")
        nn = parameters.get("nn")
        if isinstance(nn, dict):
            nn = dict(nn)
            if "hidden_layers" in nn:
                layers = nn["hidden_layers"]
                layers = [layers] if type(layers) is int else layers
                if not isinstance(layers, list) or not 1 <= len(layers) <= 4 or any(type(n) is not int or not 4 <= n <= 512 for n in layers):
                    raise ValueError("nn.hidden_layers must be an integer or 1–4 integers, each from 4 to 512")
                nn["hidden_layers"] = layers
            if "optimizer" in nn and nn["optimizer"] not in ("adam", "sgd", "rmsprop"):
                raise ValueError("nn.optimizer must be adam, sgd or rmsprop")
            if "dropout" in nn and (type(nn["dropout"]) not in (int, float) or not 0 <= nn["dropout"] <= 0.8):
                raise ValueError("nn.dropout must be a number from 0 to 0.8")
            if "batch_norm" in nn and not isinstance(nn["batch_norm"], bool):
                raise ValueError("nn.batch_norm must be true or false")
            parameters["nn"] = nn
        payload = json.dumps({"cmd": command, "policy": self.policy, **parameters}, default=_json_default, allow_nan=False)
        with self._lock:
            self._start()
            try:
                self._process.stdin.write(payload + "\n")
                self._process.stdin.flush()
                reply = self._read_reply(duration)
            except BaseException:
                self._stop()
                raise
            if command in ("shutdown", "quit"):
                self._stop()
            failures = [r for r in reply.get("results", []) if r.get("ok") is False]
            if check and (reply.get("ok") is not True or failures):
                raise EngineError(reply.get("error") or f"{len(failures)} image(s) failed", reply)
            return reply

    def fit(self, image, *, output, report, model_out, **options):
        """Fit/calibrate and optionally correct the chart image; all artifact paths explicit."""
        return self.request("run", image=image, output=output, report=report, model_out=model_out, **options)

    def _stop(self):
        process, self._process = self._process, None
        if process:
            if process.poll() is None:
                process.kill()
            process.wait()
            for reader in self._readers:
                reader.join(timeout=2)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe:
                    pipe.close()

    def close(self):
        """Flush deferred previews and stop. In-flight requests finish first."""
        with self._lock:
            try:
                if self._process and self._process.poll() is None:
                    self._process.stdin.write('{"cmd":"shutdown"}\n')
                    self._process.stdin.flush()
                    self._read_reply(min(self.timeout, 10), report_progress=False)
            except (EngineError, OSError, ValueError, TimeoutError):
                # Cleanup must not mask the error that caused context exit.
                pass
            finally:
                self._stop()

    def __enter__(self):
        with self._lock:
            self._start()
        return self

    def __exit__(self, *exc):
        self.close()

