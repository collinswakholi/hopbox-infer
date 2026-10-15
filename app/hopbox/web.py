"""Local web interface for HopBox. The browser is the control panel."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from hopbox.catalog import IMAGE_EXTS, FolderNode, node_dict, scan, scan_selection
from hopbox.branding import BRANDING
from hopbox.instances import close_hopbox_browser_windows, register_instance
from hopbox.paths import bundle_root, state_root
from hopbox.pipeline import APP_ROOT, DEFAULT_ENGINE, RunOptions, Runner

RUNTIME = state_root() / "runtime"
STATIC = bundle_root() / "hopbox" / "static"
# Same screen size and JPEG quality as pipeline._write_preview. The photo route
# only resizes; it does not run chart detection or hop inference.
_PREVIEW_LONG_SIDE = 1920


def _local_request() -> bool:
    host = (request.remote_addr or "").split("%", 1)[0]
    return host in {"127.0.0.1", "::1"}


def _is_within(child: Path, parent: Path) -> bool:
    """True when child is a file strictly inside parent. Case-insensitive on Windows."""
    child_parts = child.parts
    parent_parts = parent.parts
    if len(child_parts) <= len(parent_parts):
        return False
    return all(left.casefold() == right.casefold() for left, right in zip(parent_parts, child_parts))


def _preview_jpeg(path: Path) -> bytes:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return b""
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side > _PREVIEW_LONG_SIDE:
        scale = _PREVIEW_LONG_SIDE / long_side
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return b""
    return encoded.tobytes()


def _node_json(node: FolderNode) -> dict:
    return node_dict(node)


def _safe_dir(path: str) -> Path:
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(path)
    return folder


def _drives() -> list[str]:
    if os.name != "nt":
        root = Path("/")
        return [str(root)] if root.is_dir() else []
    return [f"{letter}:\\" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{letter}:\\").exists()]


def clear_runtime() -> None:
    """Remove preview and job files. Saved results live in the output folder, not here."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for child in list(RUNTIME.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass


def _body_flag(body: dict, key: str) -> bool:
    if key not in body or body.get(key) is None:
        return True
    value = body.get(key)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "off", "no"}
    return bool(value)


def _preview_cache_id(mode: str, body: dict) -> str:
    """Identity for one cached view. Only settings that change that view's pixels are included."""
    try:
        photo = str(Path(str(body.get("image") or "")).expanduser().resolve()).casefold()
    except OSError:
        photo = str(body.get("image") or "").casefold()
    method = str(body.get("model") or "ccm")
    gamma = "1" if _body_flag(body, "do_gc") else "0"
    balance = "1" if _body_flag(body, "do_wb") else "0"
    device = str(body.get("device") or "auto").strip() or "auto"
    try:
        imgsz = str(int(body.get("imgsz") or 2048))
    except (TypeError, ValueError):
        imgsz = "2048"
    try:
        conf = f"{float(body.get('conf') or 0.58):.4f}"
    except (TypeError, ValueError):
        conf = "0.5800"
    if mode == "chart":
        parts = ["chart", photo, device]
    elif mode in {"corrected", "plots"}:
        parts = [mode, photo, method, gamma, balance, device]
    else:
        parts = ["detections", photo, method, gamma, balance, device, imgsz, conf,str(body.get('_detection_model') or '')]
    # A photo can be replaced in place during collection. A path alone must
    # never reuse measurements or correction from its previous contents.
    try:
        stat = Path(str(body.get('image') or '')).stat()
        parts.extend([str(stat.st_size), str(stat.st_mtime_ns)])
    except OSError:
        parts.append('missing')
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _store_preview_cache(mode: str, body: dict, image_file: Path | None, title: str, plots: dict | None) -> None:
    key = _preview_cache_id(mode, body)
    folder = RUNTIME / "preview-cache" / key
    folder.mkdir(parents=True, exist_ok=True)
    if image_file is not None and Path(image_file).is_file():
        (folder / "image.jpg").write_bytes(Path(image_file).read_bytes())
    meta = {"title": title, "view": mode}
    if isinstance(plots, dict):
        meta["plots"] = plots
    try:
        text = json.dumps(meta)
    except (TypeError, ValueError):
        meta.pop("plots", None)
        text = json.dumps(meta)
    (folder / "meta.json").write_text(text, encoding="utf-8")
    caches = sorted(folder.parent.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in caches[96:]:
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)


def _read_preview_cache(mode: str, body: dict) -> dict | None:
    key = _preview_cache_id(mode, body)
    folder = RUNTIME / "preview-cache" / key
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    previews = []
    image = folder / "image.jpg"
    if image.is_file():
        previews.append({"title": str(meta.get("title") or mode), "url": f"/cache/{key}/image.jpg"})
    plots = meta.get("plots") if isinstance(meta.get("plots"), dict) else None
    if mode == "plots":
        if not plots:
            return None
    elif not previews:
        return None
    return {"hit": True, "previews": previews, "plots": plots}


def _prune_runtime(keep: str) -> None:
    if not RUNTIME.exists():
        return
    for child in list(RUNTIME.iterdir()):
        if child.name == keep or child.name == "preview-cache":
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass


def create_app() -> Flask:
    register_instance()
    clear_runtime()
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    runner = Runner()
    work: queue.Queue = queue.Queue()
    cancel = {"id": None}
    shutting = {"down": False}
    ready = {"ok": False, "text": "Loading models", "startup_id": os.environ.get('HOPBOX_STARTUP_ID', ''), "pid": os.getpid()}
    app.config['HOPBOX_READY'] = ready
    imported_roots: list[Path] = []
    allowed_photos: set[str] = set()
    photo_cache: dict[str, tuple[tuple[int, int], bytes]] = {}
    photo_lock = threading.Lock()

    def _remember_tree(payload: dict) -> None:
        """Remember photos from an import so /api/photo cannot read the rest of the disk."""
        roots: list[Path] = []
        files: set[str] = set()

        def walk(node: dict) -> None:
            raw_root = node.get("path")
            if not raw_root:
                return
            try:
                root = Path(str(raw_root)).expanduser().resolve()
            except OSError:
                return
            if root.is_dir():
                roots.append(root)
            for image in node.get("images") or []:
                if not isinstance(image, dict) or not image.get("path"):
                    continue
                try:
                    target = Path(str(image["path"])).expanduser().resolve()
                except OSError:
                    continue
                if target.suffix.lower() not in IMAGE_EXTS or not target.is_file():
                    continue
                if root.is_dir() and not _is_within(target, root):
                    continue
                files.add(str(target).casefold())
            for child in node.get("children") or []:
                if isinstance(child, dict):
                    walk(child)

        if isinstance(payload, dict):
            walk(payload)
        imported_roots[:] = roots
        allowed_photos.clear()
        allowed_photos.update(files)

    def log(job_id: str, text: str) -> None:
        with lock:
            jobs[job_id]["log"].append(text)

    def worker() -> None:
        try:
            runner.warmup()
            ready["ok"] = True
            ready["text"] = f"Models ready. {runner.compute.label}"
            ready["detail"] = runner.compute.label
        except Exception as error:
            ready["text"] = f"Model load failed: {error}"
            ready["failed"] = True
        while True:
            job_id, kind, payload = work.get()
            with lock:
                jobs[job_id]["status"] = "running"
            try:
                options = payload["options"]
                runner.announce(options, lambda text: log(job_id, text))

                def report(done: float, total: int, label: str) -> None:
                    with lock:
                        if job_id in jobs:
                            jobs[job_id]["progress"] = {"done": done, "total": total, "label": label}

                if kind == "hops":
                    kind = "corrected"
                if kind in {"chart", "corrected", "detections"}:
                    run = {
                        "chart": runner.preview_chart,
                        "corrected": runner.preview_corrected,
                        "detections": runner.preview_detections,
                    }[kind]
                    windows = run(
                        Path(payload["image"]),
                        options,
                        lambda text: log(job_id, text),
                        lambda: cancel["id"] == job_id or shutting["down"],
                        report,
                    )
                    media = RUNTIME / job_id
                    media.mkdir(parents=True, exist_ok=True)
                    flat = []
                    stored = []
                    for title, path in windows:
                        src = Path(path)
                        dest = media / f"{len(flat):02d}_{src.name}"
                        dest.write_bytes(src.read_bytes())
                        flat.append({"title": title, "url": f"/media/{job_id}/{dest.name}"})
                        stored.append((title, dest))
                    stopped = cancel["id"] == job_id
                    if not stopped and stored and kind in {"chart", "corrected", "detections"}:
                        title, dest = stored[0]
                        plots = None
                        if kind == "corrected" and isinstance(runner.last_plots, dict):
                            plots = runner.last_plots
                        try:
                            _store_preview_cache(kind, payload, dest, title, plots)
                            if kind == "corrected" and plots:
                                _store_preview_cache("plots", payload, None, title, plots)
                        except OSError:
                            pass
                    with lock:
                        jobs[job_id]["previews"] = flat
                        if kind == "corrected" and runner.last_plots:
                            jobs[job_id]["plots"] = runner.last_plots
                        jobs[job_id]["status"] = "stopped" if stopped else "done"
                elif kind == "batch":
                    runner.batch(
                        payload["folders"],
                        options,
                        lambda text: log(job_id, text),
                        lambda: cancel["id"] == job_id,
                        progress=report,
                    )
                    with lock:
                        jobs[job_id]["status"] = "stopped" if cancel["id"] == job_id else "done"
            except Exception as error:
                log(job_id, str(error))
                log(job_id, traceback.format_exc().strip().splitlines()[-1])
                with lock:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = str(error)
            finally:
                if cancel["id"] == job_id:
                    cancel["id"] = None

    threading.Thread(target=worker, daemon=True).start()

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/closing")
    def closing():
        """A page whose only job is to close the window that opened HopBox."""
        page = (
            "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>HopBox Infer</title></head>"
            "<body><script>window.close();setTimeout(function(){window.close();},150);</script></body></html>"
        )
        return Response(page, mimetype="text/html")

    @app.get("/api/ready")
    def api_ready():
        from hopbox.compute import resolve_choice
        plan=runner.compute if runner.ui_choice=='auto' else resolve_choice(runner.ui_choice)
        return jsonify({**ready,'detail':plan.label})

    @app.get('/api/about')
    def about():
        return jsonify(BRANDING)

    @app.get("/api/browse")
    def browse():
        raw = request.args.get("path") or str(Path.home() / "Desktop")
        try:
            folder = _safe_dir(raw)
        except FileNotFoundError:
            return jsonify({"error": "That folder does not exist."}), 404
        parent = str(folder.parent) if folder.parent != folder else ""
        dirs = []
        files = []
        try:
            children = sorted(folder.iterdir(), key=lambda item: item.name.lower())
        except OSError as error:
            return jsonify({"error": f"Cannot open that folder ({error})."}), 400
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir():
                dirs.append({"name": child.name, "path": str(child), "kind": "folder"})
            elif child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                files.append({"name": child.name, "path": str(child), "kind": "image"})
        return jsonify({"path": str(folder), "parent": parent, "dirs": dirs, "files": files, "drives": _drives()})

    def _import_tree(folder: Path):
        if folder.is_file():
            folder = folder.parent
        try:
            folder = _safe_dir(str(folder))
        except FileNotFoundError:
            return jsonify({"error": "That folder does not exist."}), 404
        tree = scan(folder)
        if tree is None:
            return jsonify({"error": "No photos in that folder."}), 404
        payload = _node_json(tree)
        _remember_tree(payload)
        return jsonify(payload)

    @app.post("/api/import")
    def import_folder():
        body = request.get_json(force=True) or {}
        try:
            paths = body.get("paths")
            if isinstance(paths, list) and paths:
                tree = scan_selection([Path(item) for item in paths])
                if tree is None:
                    return jsonify({"error": "No photos in that selection."}), 404
                payload = _node_json(tree)
                _remember_tree(payload)
                return jsonify(payload)
            return _import_tree(Path(body.get("path", "")))
        except FileNotFoundError:
            return jsonify({"error": "That folder does not exist."}), 404

    pick_lock = threading.Lock()

    @app.post("/api/pick")
    def api_pick():
        if not pick_lock.acquire(blocking=False):
            return jsonify({"error": "A folder window is already open."}), 409
        try:
            body = request.get_json(force=True) or {}
            from hopbox.pick import pick

            chosen = pick(body.get("purpose") or "import", body.get("start") or "")
            if chosen is None:
                return jsonify({"cancelled": True})
            return jsonify(chosen)
        except Exception as error:
            return jsonify({"error": str(error)}), 400
        finally:
            pick_lock.release()

    def _flag(body: dict, key: str) -> bool:
        if key not in body or body.get(key) is None:
            return True
        value = body.get(key)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "off", "no"}
        return bool(value)

    def _choice_imgsz(body: dict) -> tuple[str, int]:
        choice = str(body.get("device") or "auto").strip() or "auto"
        try:
            imgsz = int(body.get("imgsz") or 2048)
        except (TypeError, ValueError):
            imgsz = 2048
        return choice, imgsz if imgsz >= 32 else 2048

    @app.get("/api/vram")
    def vram():
        from hopbox.compute import device_catalog, plan_text, resolve_choice

        choice = (request.args.get("device") or "auto").strip() or "auto"
        try:
            text = plan_text(choice, runner.compute)
            batch_plan = runner.compute if choice == "auto" else resolve_choice(choice)
        except Exception:
            text = f"{runner.compute.label}. Batch {runner.compute.batch} is a safe start."
            batch_plan = runner.compute
            choice = "auto"
        return jsonify({
            "text": text,
            "batch": batch_plan.batch,
            "devices": device_catalog(),
            "active": choice,
        })

    def _backend_payload(since: int) -> dict:
        lines, cursor = runner.notes_since(since)
        return {
            "lines": lines,
            "cursor": cursor,
            "model": _model_state(),
        }

    changing={"active":False}
    def detection_identity():
        model=runner.names_model or runner.yolo
        return json.dumps([str(runner.model_path),getattr(model,'metadata',{}).get('sha256'),runner.mapping],sort_keys=True)
    def model_mutation(view):
        @wraps(view)
        def guarded():
            with lock:
                if changing["active"] or any(job['status'] in ('queued','running') for job in jobs.values()):
                    return jsonify({'error':'Finish or stop the current operation before changing settings.'}),409
                changing["active"]=True
            try:return view()
            finally:
                with lock:changing["active"]=False
        return guarded

    @app.get("/api/backend")
    def backend_state():
        try:
            since = int(request.args.get("since") or 0)
        except (TypeError, ValueError):
            since = 0
        return jsonify(_backend_payload(since))

    @app.post("/api/backend")
    @model_mutation
    def backend_set():
        with lock:
            if any(job['status'] in ('queued','running') for job in jobs.values()):
                return jsonify({'error':'Finish or stop the current job before changing compute.'}),409
        from hopbox.compute import resolve_choice

        body = request.get_json(force=True) or {}
        choice, imgsz = _choice_imgsz(body)
        try:
            since = int(body.get("since") or 0)
        except (TypeError, ValueError):
            since = 0
        try:
            resolve_choice(choice)
            status = runner.ensure_backend(choice, imgsz)
        except Exception as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({**status, **_backend_payload(since)})

    def _options(body: dict, job_id: str) -> RunOptions:
        from hopbox.compute import resolve_choice

        output = body.get("output") or str(RUNTIME / job_id)
        choice, imgsz = _choice_imgsz(body)
        if body.get("imgsz"):
            try:
                imgsz = int(body.get("imgsz"))
            except (TypeError, ValueError):
                imgsz = 2048
        plan = runner.compute if choice=='auto' else resolve_choice(choice)
        return RunOptions(
            output_dir=Path(output),
            cc_models=[body.get("model") or "ccm"],
            batch_size=max(1,min(16,int(body.get("batch_size") or 1))),
            imgsz=imgsz,
            conf=float(body.get("conf") or 0.58),
            iou=0.75,
            device=str(plan.yolo),
            device_choice=choice,
            do_gc=_flag(body, "do_gc"),
            do_wb=_flag(body, "do_wb"),
            qr_mode=body.get("qr_mode") or "qr_ocr",
            save_hops=bool(body.get("save_hops", True)),
            engine_root=DEFAULT_ENGINE,
        )

    def _named(model) -> dict[int, str] | None:
        names = getattr(model, "names", None) if model is not None else None
        if not isinstance(names, dict) or not names:
            return None
        return {int(key): str(value) for key, value in names.items()}

    def _model_state() -> dict:
        from hopbox.classmap import with_indices

        pending = runner.pending_model
        if pending is not None:
            path = Path(pending["path"])
            classes = list(pending.get("classes") or [])
            return {
                "path": str(path),
                "name": path.name,
                "ready": False,
                "needs_map": True,
                "classes": classes,
                "count": len(pending.get("sources") or classes),
                "default": False,
            }
        path = runner.model_path
        classes = list(runner.mapping)
        names = _named(getattr(runner, "names_model", None) or runner.yolo)
        if names:
            classes = with_indices(classes, names)
        return {
            "path": str(path),
            "name": path.name,
            "ready": runner.model_ready,
            "needs_map": False,
            "classes": classes,
            "count": len(classes),
            "default": path.name.lower() == "hops_fp32.onnx",
        }

    @app.get("/api/model")
    def model_state():
        return jsonify(_model_state())

    @app.post("/api/model")
    @model_mutation
    def load_model():
        with lock:
            if any(job['status'] in ('queued','running') for job in jobs.values()):
                return jsonify({'error':'Finish or stop the current job before loading another model.'}),409
        from hopbox.inference.model_adapter import Model

        from hopbox.classmap import MapStore, default_rows, is_default_names, suggest_rows, with_indices

        body = request.get_json(force=True) or {}
        raw = body.get("path", "")
        path = Path(raw)
        if path.suffix.lower() not in {".onnx",".pt"} or not path.is_file():
            return jsonify({"error": "Choose an ONNX model bundle or a .pt checkpoint."}), 400
        try:
            if path.suffix.lower()=='.pt':
                from hopbox.inference.model_import import convert_checkpoint
                path=convert_checkpoint(path,int(body.get('imgsz') or 2048))
            model = Model(path)
        except Exception as error:
            return jsonify({"error": f"Could not load that model ({error})."}), 400
        names = {int(key): str(value) for key, value in model.names.items()}
        sources = [names[key] for key in sorted(names)]
        store = MapStore(state_root() / "models" / "classmap.json")
        saved = None if is_default_names(sources) else store.find(str(path.resolve()), sources)
        rows = default_rows() if is_default_names(sources) else saved
        if rows is None:
            suggested = suggest_rows(names)
            runner.pending_model = {
                "yolo": model,
                "path": path.resolve(),
                "sources": sources,
                "classes": suggested,
            }
            runner.model_ready = False
            return jsonify({
                "needs_map": True,
                "path": str(path.resolve()),
                "name": path.name,
                "count": len(sources),
                "classes": suggested,
            })
        choice, imgsz = _choice_imgsz(body)
        snap = runner._snapshot()
        runner.install_model(model, path.resolve(), with_indices(rows, names))
        try:
            since = int(body.get("since") or 0)
        except (TypeError, ValueError):
            since = 0
        status = runner.ensure_backend(choice, imgsz, rollback=snap)
        return jsonify({"needs_map": False, **_model_state(), **status, **_backend_payload(since)})

    @app.post("/api/model/map")
    @model_mutation
    def save_model_map():
        with lock:
            if any(job['status'] in ('queued','running') for job in jobs.values()):
                return jsonify({'error':'Finish or stop the current job before changing classes.'}),409
        from hopbox.classmap import MapStore, normalize_rows, with_indices

        body = request.get_json(force=True) or {}
        pending = runner.pending_model
        if pending is None:
            sources = [row["source"] for row in runner.mapping]
            path = runner.model_path
            model = runner.names_model or runner.yolo
        else:
            sources = pending["sources"]
            path = pending["path"]
            model = pending["yolo"]
        rows = normalize_rows(body.get("rows") or [], sources)
        names = _named(model)
        if names:
            rows = with_indices(rows, names)
        if not rows:
            return jsonify({"error": "The class list is empty."}), 400
        MapStore(state_root() / "models" / "classmap.json").save(str(Path(path).resolve()), sources, rows)
        choice, imgsz = _choice_imgsz(body)
        snap = runner._snapshot()
        runner.install_model(model, Path(path).resolve(), rows)
        try:
            since = int(body.get("since") or 0)
        except (TypeError, ValueError):
            since = 0
        status = runner.ensure_backend(choice, imgsz, rollback=snap)
        return jsonify({"needs_map": False, **_model_state(), **status, **_backend_payload(since)})

    @app.post('/api/model/cancel')
    @model_mutation
    def cancel_model():
        if runner.pending_model is not None:
            runner.pending_model['yolo'].close()
            runner.pending_model=None
            runner.model_ready=True
        return jsonify(_model_state())

    def _enqueue(kind: str, body: dict) -> tuple:
        if not ready['ok']:
            return jsonify({'error': ready['text']}), 409
        if shutting["down"]:
            return jsonify({"error": "HopBox Infer is closing."}), 409
        if not runner.model_ready:
            return jsonify({"error": "Name the classes in this detection model first."}), 409
        try:
            options = _options(body, "pending")
        except Exception as error:
            return jsonify({"error": str(error)}), 400
        job_id = uuid.uuid4().hex[:12]
        with lock:
            if changing['active']:
                return jsonify({'error':'Wait for model loading or the compute change to finish.'}),409
            running = any(job["status"] in {"queued", "running"} for job in jobs.values())
            if running:
                return jsonify({"error": "A job is already running."}), 409
            # Completed jobs retain only small metadata, but unbounded records
            # still grow during long sessions. Keep recent job URLs valid.
            while len(jobs) >= 32:
                oldest = next(iter(jobs))
                del jobs[oldest]
            jobs[job_id] = {
                "id": job_id, "status": "queued", "log": [], "previews": [], "error": "",
                "progress": {"done": 0, "total": 1, "label": "Starting"},
            }
        _prune_runtime(job_id)
        if kind in {"chart", "corrected", "detections", "hops"}:
            body = dict(body)
            body["output"] = str(RUNTIME / job_id)
            body['_detection_model']=detection_identity()
        options.output_dir = Path(body.get("output") or options.output_dir)
        work.put((job_id, kind, {"options": options, **body}))
        return jsonify({"id": job_id})

    @app.post("/api/preview-cache")
    def preview_cache():
        """Return a saved preview without running detection or color fitting."""
        if not _local_request():
            abort(403)
        body = request.get_json(force=True) or {}
        mode = body.get("mode")
        body['_detection_model']=detection_identity()
        if mode not in {"chart", "corrected", "detections", "plots"} or not body.get("image"):
            return jsonify({"hit": False})
        found = _read_preview_cache(mode, body)
        if not found:
            return jsonify({"hit": False})
        return jsonify(found)

    @app.post("/api/preview")
    def preview():
        body = request.get_json(force=True) or {}
        if not body.get("image"):
            return jsonify({"error": "Choose a photo first."}), 400
        mode = body.get("mode")
        if mode not in {"chart", "corrected", "detections", "hops"}:
            return jsonify({"error": "Unknown preview."}), 400
        return _enqueue("corrected" if mode == "hops" else mode, body)

    @app.post("/api/run")
    def run_batch():
        body = request.get_json(force=True) or {}
        if not body.get("folders"):
            return jsonify({"error": "Check at least one folder that contains photos."}), 400
        return _enqueue("batch", body)

    @app.get("/api/jobs/<job_id>")
    def job(job_id: str):
        with lock:
            found = jobs.get(job_id)
            if not found:
                return jsonify({"error": "Unknown job."}), 404
            return jsonify(found)

    @app.post("/api/jobs/<job_id>/stop")
    def stop(job_id: str):
        with lock:
            found = jobs.get(job_id)
            if found is None:
                return jsonify({'error': 'Unknown job.'}), 404
            if found['status'] not in {'queued', 'running'}:
                return jsonify({'ok': True})
            cancel['id'] = job_id
            found['log'].append('Stopping after the current step')
        return jsonify({"ok": True})

    @app.post("/api/exit")
    def exit_app():
        """Stop the current job, delete preview cache, then close the server."""
        shutting["down"] = True
        from hopbox.inference.model_import import cancel_export
        cancel_export()
        with lock:
            active = [job_id for job_id, job in jobs.items() if job["status"] in {"queued", "running"}]
        for job_id in active:
            cancel["id"] = job_id
            log(job_id, "Exit requested. Stopping after the current step")

        def _close() -> None:
            while True:
                with lock:
                    busy = changing['active'] or any(job["status"] in {"queued", "running"} for job in jobs.values())
                if not busy:
                    break
                time.sleep(0.2)
            time.sleep(0.35)
            try:
                runner.close()
            except Exception:
                pass
            try:
                if os.environ.get('HOPBOX_NO_BROWSER')!='1':
                    close_hopbox_browser_windows()
            except Exception:
                pass
            clear_runtime()
            os._exit(0)

        threading.Thread(target=_close, name="hopbox-exit", daemon=True).start()
        return jsonify({"ok": True})

    @app.get("/api/photo")
    def photo():
        """JPEG of one imported photo, scaled for the stage. No detection or color work."""
        if not _local_request():
            abort(403)
        raw = request.args.get("path") or ""
        if not raw or "\x00" in raw:
            abort(404)
        try:
            target = Path(raw).expanduser().resolve()
        except OSError:
            abort(404)
        if target.suffix.lower() not in IMAGE_EXTS or not target.is_file():
            abort(404)
        key = str(target).casefold()
        if key not in allowed_photos or not any(_is_within(target, root) for root in imported_roots):
            abort(404)
        try:
            stat = target.stat()
        except OSError:
            abort(404)
        token = (getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)), stat.st_size)
        with photo_lock:
            hit = photo_cache.get(key)
            blob = hit[1] if hit and hit[0] == token else b""
        if not blob:
            try:
                blob = _preview_jpeg(target)
            except OSError:
                abort(404)
            if not blob:
                abort(404)
            with photo_lock:
                photo_cache[key] = (token, blob)
                while len(photo_cache) > 16:
                    photo_cache.pop(next(iter(photo_cache)))
        return Response(
            blob,
            mimetype="image/jpeg",
            headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/media/<job_id>/<name>")
    def media(job_id: str, name: str):
        folder = (RUNTIME / job_id).resolve()
        target = (folder / name).resolve()
        if folder not in target.parents and target != folder:
            abort(404)
        if not target.is_file():
            abort(404)
        return send_from_directory(folder, name)

    @app.get("/cache/<key>/image.jpg")
    def cached_preview(key: str):
        if not _local_request():
            abort(403)
        if not re.fullmatch(r"[0-9a-f]{24}", key):
            abort(404)
        folder = (RUNTIME / "preview-cache" / key).resolve()
        target = (folder / "image.jpg").resolve()
        if target.parent != folder or not target.is_file():
            abort(404)
        return send_from_directory(folder, "image.jpg")

    return app
