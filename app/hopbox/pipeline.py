"""One-photo trials and folder batches. Models live under the app folder."""
from __future__ import annotations

import gc
import json
import shutil
import threading
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hopbox.classmap import default_rows, display_for, measure_names, role_source
from hopbox.compute import quick_plan
from hopbox.inference.fit_geometry import chart_window, shift_handshake
from hopbox.features import crop_class, draw_hops, draw_named, get_ids, hop_features
from hopbox.paths import app_root
from hopbox.qr_read import read_qr

APP_ROOT = app_root()
_PREVIEW_LONG_SIDE = 1920
DEFAULT_ENGINE = APP_ROOT / "resources"
# Scientific model runs consistently in FP32 across CPU and GPU.



def _method_name(model: str) -> str:
    return {"ccm": "CCM", "rpcc2": "RPCC2", "rpcc3": "RPCC3"}.get(str(model), str(model).upper())


def _is_memory_error(error: BaseException) -> bool:
    """True when the process or the GPU cannot allocate the next photo."""
    text = f"{type(error).__name__} {error}".lower()
    return isinstance(error, MemoryError) or any(phrase in text for phrase in (
        "out of memory", "not enough memory", "bad allocation", "bad_alloc",
        "e_outofmemory", "0x8007000e", "insufficient memory",
        "failed to allocate", "unable to allocate",
    ))


def release_workspace() -> None:
    """Collect released Python arrays; native sessions own their device pools."""
    gc.collect()



def _release_result(result) -> None:
    """Drop the full-frame image and masks once traits and drawings are saved."""
    for name in ("orig_img", "masks", "boxes", "polygons", "scores", "classes"):
        if not hasattr(result, name):
            continue
        try:
            setattr(result, name, None)
        except Exception:
            pass




def _write_preview(path: Path, image) -> None:
    """Save a screen-sized JPEG. Full-resolution frames stay in the batch output."""
    view = image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side > _PREVIEW_LONG_SIDE:
        scale = _PREVIEW_LONG_SIDE / long_side
        view = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), view, [int(cv2.IMWRITE_JPEG_QUALITY), 86])


@dataclass
class RunOptions:
    output_dir: Path
    cc_models: list[str] = field(default_factory=lambda: ["ccm", "rpcc2", "rpcc3"])
    batch_size: int = 1
    imgsz: int = 2048
    conf: float = 0.58
    iou: float = 0.75
    device: str = "cpu"
    device_choice: str = "auto"
    do_gc: bool = True
    do_wb: bool = True
    qr_mode: str = "qr_ocr"
    save_hops: bool = True
    engine_root: Path = DEFAULT_ENGINE




def draw_chart(image: np.ndarray, detection: dict) -> np.ndarray:
    canvas = image.copy()
    scale = max(canvas.shape[0], canvas.shape[1]) / 1000.0
    thin = max(2, int(round(2 * scale)))
    thick = max(4, int(round(4 * scale)))
    for quad in detection.get("patch_obbs") or []:
        pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 220, 0), thin, cv2.LINE_AA)
    corners = detection.get("corners") or {}
    ordered = [corners.get(name) for name in ("tl", "tr", "br", "bl")]
    if all(point is not None for point in ordered):
        pts = np.asarray(ordered, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 0, 255), thick, cv2.LINE_AA)
    label = f"{detection.get('chart_type')}  {float(detection.get('score', 0)):.2f}"
    origin = (int(40 * scale), int(80 * scale))
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thin, cv2.LINE_AA)
    return canvas


def _class_count(result, class_name: str) -> int:
    ids = get_ids(result, class_name)
    if result.boxes is None or len(ids) == 0:
        return 0
    return int(np.isin(result.classes, ids).sum())


def patch_colors(report: dict) -> list[dict]:
    rows = []
    for patch in report.get("patches") or []:
        if not isinstance(patch, dict):
            continue

        def color(role: str) -> list[float]:
            values = patch.get(f"{role}_display_srgb") or patch.get(f"{role}_srgb") or []
            return [float(value) for value in list(values)[:3]]

        rows.append({
            "name": str(patch.get("name") or ""),
            "reference": color("reference"),
            "measured": color("measured"),
            "corrected": color("corrected"),
        })
    return rows


def _step(progress, done: float, label: str) -> None:
    if progress is not None:
        progress(float(done), 1, label)


def _metric(report: dict, *keys) -> float:
    node = report
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return float("nan")
        node = node[key]
    try:
        value = float(node)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


class Runner:
    def __init__(self):
        self.yolo = None
        self.detector = None
        self._engine = None
        self._engine_root: Path | None = None
        self.compute = quick_plan()
        self.model_path = APP_ROOT / "models" / "portable" / "hops_fp32.onnx"
        self.mapping = default_rows()
        self.model_ready = True
        self.pending_model = None
        self.last_plots = None
        self._weights: dict[tuple, object] = {}
        self._detectors: dict[str, object] = {}
        self._hop_tuned: set[tuple] = set()
        self._hop_lock = threading.Lock()
        self.names_model = None
        self._notes: list[tuple[int, str]] = []
        self._note_seq = 0
        self._notes_lock = threading.Lock()
        self.ui_choice = "auto"
        self.ui_imgsz = 2048

    def close(self):
        if self._engine is not None:self._engine.close()
        self._engine=None
        for model in self._weights.values():model.close()
        self._weights.clear()
        for detector in self._detectors.values():
            for session in detector.sessions.values():session.close()
        self._detectors.clear()
        from hopbox.qr_read import close_reader
        close_reader()

    def note(self, text: str) -> None:
        with self._notes_lock:
            self._note_seq += 1
            self._notes.append((self._note_seq, text))
            if len(self._notes) > 300:
                self._notes = self._notes[-200:]

    def notes_since(self, seq: int) -> tuple[list[str], int]:
        with self._notes_lock:
            lines = [text for number, text in self._notes if number > int(seq)]
            cursor = self._notes[-1][0] if self._notes else int(seq)
        return lines, cursor

    def _snapshot(self) -> dict:
        return {
            "yolo": self.yolo,
            "model_path": self.model_path,
            "mapping": list(self.mapping),
            "model_ready": self.model_ready,
            "pending_model": self.pending_model,
            "weights": dict(self._weights),
            "names_model": self.names_model,
        }

    def _restore(self, snap: dict | None) -> None:
        if not snap:
            return
        self.yolo = snap["yolo"]
        self.model_path = snap["model_path"]
        self.mapping = list(snap["mapping"])
        self.model_ready = snap["model_ready"]
        self.pending_model = snap["pending_model"]
        self._weights = dict(snap["weights"])
        self.names_model = snap["names_model"]

    def install_model(self, model, path, mapping):
        for previous in self._weights.values(): previous.close()
        self.model_path=Path(path).resolve()
        self.mapping=list(mapping)
        self.model_ready=True
        self.pending_model=None
        self._weights={}
        self.names_model=model
        self.yolo=model


    def ensure_backend(self, choice, imgsz, rollback=None):
        from hopbox.compute import resolve_choice
        plan=self.compute if not choice or choice=='auto' else resolve_choice(choice)
        self.ui_choice=choice or 'auto'
        self.ui_imgsz=int(imgsz or 2048)
        if self.ui_imgsz<32 or self.ui_imgsz>4096: raise ValueError('Image size must be between 32 and 4096')
        if self.pending_model is not None or not self.model_ready:
            return {'ok':True,'needs_map':True}
        try:
            self._yolo(plan.yolo,self.ui_imgsz)
        except Exception:
            self._restore(rollback)
            raise
        return {'ok':True,'needs_map':False,'device':plan.yolo}



    def announce(self, options, log):
        log(f'Batch size {options.batch_size}; image size {options.imgsz}; confidence {options.conf:.0%}.')

    def warmup(self):
        self.compute=quick_plan()
        device=self.compute.yolo
        try:
            self._warm_hop(device,self.ui_imgsz)
            self._detector(device).forward(np.zeros((128,192,3),np.uint8),1280)
        except Exception as error:
            if device=='cpu': raise
            self.note(f'GPU startup failed ({error}); using CPU.')
            self.compute=type(self.compute)('cpu','CPU · FP32',1)
            self._warm_hop('cpu',self.ui_imgsz)
            self._detector('cpu').forward(np.zeros((128,192,3),np.uint8),1280)
        self._color_engine(DEFAULT_ENGINE)

    def _warm_hop(self, device, imgsz, batch=1, log=None):
        key=(str(device),int(imgsz))
        if key in self._hop_tuned:return
        with self._hop_lock:
            self._yolo(device,imgsz).predict(np.zeros((32,48,3),np.uint8),imgsz=imgsz)
            self._hop_tuned.add(key)

    def _yolo(self, device, imgsz=None):
        from hopbox.inference.model_adapter import Model
        key=(str(device),str(self.model_path))
        if key not in self._weights:
            for previous in self._weights.values():previous.close()
            self._weights.clear()
            self._weights[key]=Model(self.model_path,str(device))
        self.yolo=self._weights[key]
        self.names_model=self.yolo
        return self.yolo


    def _detector(self, device):
        from hopbox.inference.chart import ChartDetector
        from hopbox.compute import runtime_device
        key=str(device)
        if key not in self._detectors:
            for detector in self._detectors.values():
                for session in detector.sessions.values():session.close()
            self._detectors.clear()
            provider,index=runtime_device(device)
            self._detectors[key]=ChartDetector(APP_ROOT/'models/portable',provider,index)
        self.detector=self._detectors[key]
        return self.detector

    def _color_engine(self, root, device=None):
        from hopbox.colorcorrect import Engine
        root=Path(root)
        if self._engine is not None and self._engine_root==root:return self._engine
        if self._engine is not None:self._engine.close()
        engine=Engine(root=root,executable=APP_ROOT/'native/ccp-engine.exe',policy='cpu',timeout=600)
        engine.__enter__()
        self._engine=engine
        self._engine_root=root
        return engine

    def _apply_frame(self, original, saved, device=None):
        from hopbox.inference.rust_color import apply_rust
        return apply_rust(original,saved)

    def _fit(
        self,
        engine,
        original: np.ndarray,
        handshake: str,
        model: str,
        work_dir: Path,
        device: str | None = None,
        do_gc: bool = True,
        do_wb: bool = True,
        log=None,
        image_name: str = "",
    ) -> tuple[np.ndarray, dict]:
        work_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(Path(handshake).read_text(encoding="utf-8"))
        window = chart_window(original, payload)
        if window is None:
            source = work_dir / "frame.png"
            cv2.imwrite(str(source), original, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            fit_handshake = handshake
        else:
            crop, x0, y0 = window
            source = work_dir / "chart.png"
            cv2.imwrite(str(source), crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            fit_handshake = work_dir / "fit_handshake.json"
            fit_handshake.write_text(json.dumps(shift_handshake(payload, x0, y0)), encoding="utf-8")
        reply = engine.fit(
            source,
            handshake=fit_handshake,
            output=work_dir / "corrected.png",
            report=work_dir / "report.json",
            model_out=work_dir / "model.json",
            model=model,
            do_ffc=False,
            do_gc=bool(do_gc),
            do_wb=bool(do_wb),
            do_cc=True,
            cross_validate=True,
            phase2_teacher=False,
            preview_mode="none",
            write_output=False,
        )
        chosen = self.compute.yolo if device is None else str(device)
        saved = json.loads((work_dir / "model.json").read_text(encoding="utf-8"))
        saved["do_gc"] = bool(do_gc)
        saved["do_wb"] = bool(do_wb)
        saved["do_cc"] = True
        corrected = self._apply_frame(original, saved, chosen)
        if corrected.shape != original.shape:
            raise RuntimeError(f"{model} did not return an image of the original size")
        return corrected, reply.get("report_json") or {}

    def _traits(self, result, corrected, model: str, report: dict, image_name: str, qr_mode: str, ocr_dir: Path) -> pd.DataFrame:
        result.orig_img = corrected
        qr_name = role_source(self.mapping, "qr")
        if qr_name:
            qr_crop = crop_class(result, qr_name)
            label = crop_class(result, qr_name, pad=0.9)
            qr_info = read_qr(qr_crop if qr_crop is not None else corrected, qr_mode, ocr_dir, label=label)
        else:
            qr_info = "NA"
        features = hop_features(result, measure_names(self.mapping))
        if features.empty:
            return features
        classes = features.pop("_class") if "_class" in features.columns else None
        count = len(features)
        front = {
            "Image_name": [image_name] * count,
            "ID": list(range(count)),
        }
        if classes is not None and (classes.nunique() > 1 or measure_names(self.mapping) != ["Hops"]):
            front["Class"] = [display_for(self.mapping, name) for name in classes.tolist()]
        front["QR_info"] = [qr_info] * count
        front = pd.DataFrame(front)
        tail = pd.DataFrame({
            "CC_model": [model] * count,
            "CC_deltaE": [_metric(report, "delta_e_2000", "mean_after")] * count,
            "CC_deltaE_holdout": [_metric(report, "validation_delta_e_2000", "mean")] * count,
        })
        return pd.concat([front, features, tail], axis=1)

    def _prepare(self, image_path: Path, options: RunOptions):
        image_path = Path(image_path)
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(image_path)
        out = Path(options.output_dir) / "_preview" / image_path.stem
        out.mkdir(parents=True, exist_ok=True)
        return image_path, original, out

    def _find_chart(self, original, image_path: Path, out: Path, device: str):
        detection = self._detector(device).detect(original, image_path, out / "handshake.json", prefer="classic")
        chart_path = out / "chart.jpg"
        _write_preview(chart_path, draw_chart(original, detection))
        return detection, chart_path

    def preview_chart(self, image_path: Path, options: RunOptions, log, cancelled, progress=None) -> list[tuple[str, Path]]:
        image_path, original, out = self._prepare(image_path, options)
        if cancelled():
            return []
        started = time.perf_counter()
        _step(progress, 0.08, f"Chart · {image_path.name}")
        log(f"Chart detection for {image_path.name}.")
        _step(progress, 0.45, f"Chart · {image_path.name} · reading patches")
        try:
            detection, chart_path = self._find_chart(original, image_path, out, options.device)
        except Exception as error:
            log(f"{image_path.name}: chart not found ({error}).")
            raise
        _step(progress, 1, f"Chart · {image_path.name}")
        patches = len(detection.get("patch_obbs") or [])
        log(
            f"{image_path.name}: chart {detection.get('chart_type')} via {detection.get('recall_source')}, "
            f"score {float(detection.get('score', 0)):.2f}, {patches} patches "
            f"in {time.perf_counter() - started:.1f}s."
        )
        return [(f"Chart — {image_path.name}", chart_path)]

    def preview_corrected(self, image_path: Path, options: RunOptions, log, cancelled, progress=None) -> list[tuple[str, Path]]:
        """The selected color method applied to the whole photo."""
        image_path, original, out = self._prepare(image_path, options)
        if cancelled():
            return []
        model = (options.cc_models or ["ccm"])[0]
        started = time.perf_counter()
        _step(progress, 0.08, f"Corrected · {image_path.name}")
        log(f"Corrected preview for {image_path.name}.")
        _step(progress, 0.3, f"Corrected · {image_path.name} · chart")
        try:
            detection, _chart_path = self._find_chart(original, image_path, out, options.device)
        except Exception as error:
            log(f"{image_path.name}: chart not found ({error}).")
            raise
        patches = len(detection.get("patch_obbs") or [])
        log(
            f"{image_path.name}: chart {detection.get('chart_type')} via {detection.get('recall_source')}, "
            f"score {float(detection.get('score', 0)):.2f}, {patches} patches."
        )
        if cancelled():
            return []
        _step(progress, 0.55, f"Corrected · {image_path.name} · {model.upper()}")
        engine = self._color_engine(options.engine_root, options.device)
        corrected, report = self._fit(
            engine, original, detection["handshake"], model, out / model,
            device=options.device, do_gc=options.do_gc, do_wb=options.do_wb,
            log=log, image_name=image_path.name,
        )
        path = out / "corrected.jpg"
        _write_preview(path, corrected)
        self.last_plots = {
            "image": str(image_path),
            "model": model,
            "patches": patch_colors(report),
        }
        delta = _metric(report, "delta_e_2000", "mean_after")
        _step(progress, 1, f"Corrected · {image_path.name}")
        log(
            f"{image_path.name}: {_method_name(model)} ΔE {delta:.2f}, "
            f"finished in {time.perf_counter() - started:.1f}s."
        )
        return [(f"Corrected — {image_path.name}", path)]

    def preview_hops(self, image_path: Path, options: RunOptions, log, cancelled, progress=None) -> list[tuple[str, Path]]:
        image_path, original, out = self._prepare(image_path, options)
        if cancelled():
            return []
        device = options.device
        started = time.perf_counter()
        _step(progress, 0.08, f"Hops · {image_path.name}")
        log(f"Hop detections for {image_path.name}.")
        _step(progress, 0.4, f"Hops · {image_path.name} · finding cones")
        self._warm_hop(device, options.imgsz, 1, log)
        result = self._yolo(device, options.imgsz).predict(
            source=original, imgsz=options.imgsz, conf=options.conf, iou=options.iou,
        )[0]
        hop_ids = get_ids(result, "Hops")
        hop_count = 0 if result.boxes is None or len(hop_ids) == 0 else int(np.isin(result.classes, hop_ids).sum())
        hop_path = out / "hops.jpg"
        cv2.imwrite(str(hop_path), draw_hops(result), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        _step(progress, 1, f"Hops · {image_path.name}")
        log(f"{image_path.name}: {hop_count} {'hops' if measure_names(self.mapping) == ['Hops'] else 'objects'} in {time.perf_counter() - started:.1f}s.")
        _release_result(result)
        release_workspace()
        return [(f"Hops — {image_path.name}", hop_path)]

    def preview_detections(self, image_path: Path, options: RunOptions, log, cancelled, progress=None) -> list[tuple[str, Path]]:
        """One image with every overlay: the color-chart lattice, QR, hops, and any other class."""
        image_path, original, out = self._prepare(image_path, options)
        if cancelled():
            return []
        device = options.device
        started = time.perf_counter()
        _step(progress, 0.06, f"Detections · {image_path.name}")
        log(f"Detections for {image_path.name}.")
        _step(progress, 0.28, f"Detections · {image_path.name} · hops and QR")
        self._warm_hop(device, options.imgsz, 1, log)
        yolo_started = time.perf_counter()
        result = self._yolo(device, options.imgsz).predict(
            source=original, imgsz=options.imgsz, conf=options.conf, iou=options.iou,
        )[0]
        yolo_s = time.perf_counter() - yolo_started
        if cancelled():
            return []
        _step(progress, 0.62, f"Detections · {image_path.name} · chart")
        hop_count = sum(_class_count(result, name) for name in measure_names(self.mapping))
        qr_name = role_source(self.mapping, "qr") or "QR"
        qr_count = _class_count(result, qr_name)
        log(f"{image_path.name}: {hop_count} {'hops' if measure_names(self.mapping) == ['Hops'] else 'objects'}, {qr_count} QR in {yolo_s:.1f}s.")
        model = (options.cc_models or ["ccm"])[0]
        detection = None
        chart_started = time.perf_counter()
        try:
            detection, _chart_path = self._find_chart(original, image_path, out, device)
            patches = len(detection.get("patch_obbs") or [])
            log(
                f"{image_path.name}: chart {detection.get('chart_type')} via {detection.get('recall_source')}, "
                f"score {float(detection.get('score', 0)):.2f}, {patches} patches "
                f"in {time.perf_counter() - chart_started:.1f}s."
            )
        except Exception as error:
            log(f"{image_path.name}: chart not found ({error}).")
        base = original
        if detection is not None:
            try:
                _step(progress, 0.78, f"Detections · {image_path.name} · color")
                base, _report = self._fit(
                    self._color_engine(options.engine_root, device), original, detection["handshake"], model, out / model,
                    device=device, do_gc=options.do_gc, do_wb=options.do_wb,
                    log=log, image_name=image_path.name,
                )
            except Exception as error:
                log(f"{image_path.name}: color fit failed ({error}). Drawing on the original.")
                base = original
        draw_started = time.perf_counter()
        canvas = draw_chart(base, detection) if detection is not None else base.copy()
        counts = []
        for row in self.mapping:
            if row["role"] == "skip":
                continue
            if row["role"] in {"measure", "draw"}:
                before = canvas
                canvas = draw_hops(result, canvas=canvas, class_name=row["source"], label_prefix=row["display"])
                if canvas is not before or row["role"] == "measure":
                    count = _class_count(result, row["source"])
                    if count and row["role"] == "measure":
                        counts.append(f"{count} {row['display']}")
            else:
                canvas, count = draw_named(result, row["source"], canvas, label=row["display"])
                if count and row["role"] != "qr":
                    counts.append(f"{count} {row['display']}")
                if row["role"] == "qr":
                    counts.insert(0, f"{count} QR")
        path = out / "detections.jpg"
        _write_preview(path, canvas)
        draw_s = time.perf_counter() - draw_started
        _step(progress, 1, f"Detections · {image_path.name}")
        if counts:
            log(f"{image_path.name}: {', '.join(counts)}.")
        log(
            f"{image_path.name}: drew overlays in {draw_s:.1f}s, "
            f"finished in {time.perf_counter() - started:.1f}s."
        )
        _release_result(result)
        release_workspace()
        return [(f"Detections — {image_path.name}", path)]

    def _predict_chunk(self, yolo, paths: list[Path], device: str, options: RunOptions, log) -> list:
        """Hop detections for one batch. A full GPU is retried one photo at a time, then on CPU."""
        try:
            ready = self._predict_paths(yolo, paths, device, options)
        except Exception as error:
            if not _is_memory_error(error):
                raise
            # Retried inference must not retain the failed call's image tensors
            # through its traceback. Leave the exception handler before retrying.
            error.__traceback__ = None
        else:
            # CPU batch arenas can consume most of a small machine's memory.
            # Plain detection results no longer need the inference session;
            # release its arena before chart, OCR and trait processing.
            if str(device) == 'cpu' and len(paths) > 1:
                yolo.close()
                release_workspace()
            return ready
        yolo.close()
        release_workspace()
        if len(paths) > 1:
            split=max(1,len(paths)//2)
            log(f"Not enough memory for {len(paths)} photos. Retrying in smaller batches.")
            return self._predict_chunk(yolo,paths[:split],device,options,log)+self._predict_chunk(yolo,paths[split:],device,options,log)
        if str(device) != "cpu":
            log(f"{paths[0].name}: GPU memory was full. Finding hops on the CPU.")
            from hopbox.inference.model_adapter import Model
            cpu_model = Model(self.model_path,"cpu")
            try:
                return self._predict_paths(cpu_model, paths, "cpu", options)
            except Exception as retry_error:
                if not _is_memory_error(retry_error):
                    raise
                raise MemoryError(f'{paths[0].name}: not enough memory on GPU or CPU. Reduce image size or close other applications.') from retry_error
            finally:
                cpu_model.close()
                release_workspace()
        raise MemoryError(f'{paths[0].name}: not enough memory. Reduce image size or close other applications.')

    def _predict_paths(self, yolo, paths: list[Path], device: str, options: RunOptions) -> list:
        ready = []
        stream = yolo.predict(
            source=[str(path) for path in paths],
            stream=True,
            imgsz=options.imgsz,
            conf=options.conf,
            iou=options.iou,
        )
        for result in stream:
            ready.append(result)
            del result
        release_workspace()
        return ready

    def batch(self, folders: list[Path], options: RunOptions, log, cancelled, progress=None) -> None:
        # Keep intermediates on the local disk even when results are on a
        # network share or VM mapped folder. Cleanup also runs on Stop/error.
        with tempfile.TemporaryDirectory(prefix='hopbox-batch-') as scratch:
            self._batch(folders, options, log, cancelled, progress, Path(scratch))

    def _batch(self, folders, options, log, cancelled, progress, scratch):
        device = options.device
        yolo = self._yolo(device, options.imgsz)
        detector = self._detector(device)
        engine = self._color_engine(options.engine_root, device)
        ocr_dir = APP_ROOT / "models" / "portable" / "ocr"
        plans = []
        for entry in folders:
            if isinstance(entry, dict):
                folder = Path(entry.get("path") or "")
                listed = [Path(item) for item in entry.get("files") or [] if Path(item).is_file()]
            else:
                folder = Path(entry)
                listed = []
            if not listed and folder.is_dir():
                listed = sorted(
                    [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}],
                    key=lambda path: path.name.lower(),
                )
            if listed:
                plans.append((folder, listed))
        total = sum(len(images) for _, images in plans)
        seen = 0

        def tick(done: float, label: str) -> None:
            if progress is not None and total:
                progress(min(float(total), done), total, label)

        if total == 0:
            log("No photos in the selected folders")
            return
        step = max(1,min(16,int(options.batch_size)))
        self._warm_hop(device, options.imgsz, step, log)
        tick(0, f"0 of {total}")
        used_names = set()
        for folder, images in plans:
            key = folder.name.casefold()
            if key in used_names:
                raise ValueError('Selected folders have the same name. Run them separately to avoid overwriting results: ' + folder.name)
            used_names.add(key)
        for folder, images in plans:
            if cancelled():
                log("Stopped before the next folder")
                return
            group = Path(options.output_dir) / folder.name
            group.mkdir(parents=True, exist_ok=True)
            log(f"{folder.name}: {len(images)} photos")
            rows = []
            for start in range(0, len(images), step):
                if cancelled():
                    log("Stopped")
                    return
                chunk = images[start:start + step]
                tick(seen, f"{seen} of {total} · finding hops")
                log(f"Finding hops in photos {start + 1}–{start + len(chunk)}.")
                ready = self._predict_chunk(yolo, chunk, device, options, log)
                for offset, result in enumerate(ready):
                    if cancelled():
                        return
                    base = seen + offset
                    name = Path(result.path).name
                    stem = Path(name).stem
                    original = result.orig_img
                    work = scratch / stem
                    started = time.perf_counter()
                    hop_count = sum(_class_count(result, name) for name in measure_names(self.mapping))
                    qr_name = role_source(self.mapping, "qr") or "QR"
                    qr_count = _class_count(result, qr_name)
                    try:
                        tick(base + 0.2, f"{base + 1} of {total} · {name} · chart")
                        try:
                            detection = detector.detect(original, Path(result.path), work / "handshake.json", prefer="classic")
                        except Exception as error:
                            if _is_memory_error(error):
                                raise MemoryError(f'{name}: not enough memory for chart detection. Use a smaller batch or close other applications.') from error
                            log(f"{name}: chart not found ({error}).")
                            log(f"{name}: finished in {time.perf_counter() - started:.1f}s.")
                            tick(base + 1, f"{base + 1} of {total} · {name} · no chart")
                            continue
                        chart_path = group / "Chart_Detections" / f"{stem}.jpg"
                        chart_path.parent.mkdir(parents=True, exist_ok=True)
                        chart_image = original
                        for model in options.cc_models:
                            if cancelled():
                                return
                            tick(base + 0.45, f"{base + 1} of {total} · {name} · color")
                            try:
                                corrected, report = self._fit(
                                    engine, original, detection["handshake"], model, work / model,
                                    device=device, do_gc=options.do_gc, do_wb=options.do_wb,
                                    log=log, image_name=name,
                                )
                            except Exception as error:
                                if _is_memory_error(error) and str(device) == 'cpu':
                                    raise
                                if not _is_memory_error(error) or str(device) == "cpu":
                                    log(f"{name}: {_method_name(model)} failed ({error}).")
                                    continue
                                release_workspace()
                                log(f"{name}: not enough GPU memory for color. Applying on the CPU.")
                                try:
                                    corrected, report = self._fit(
                                        self._color_engine(options.engine_root, "cpu"),
                                        original, detection["handshake"], model, work / model,
                                        device="cpu", do_gc=options.do_gc, do_wb=options.do_wb,
                                        log=log, image_name=name,
                                    )
                                except Exception as retry_error:
                                    log(f"{name}: {_method_name(model)} failed ({retry_error}).")
                                    continue
                            chart_image = corrected
                            cv2.imwrite(str(chart_path), draw_chart(chart_image, detection), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            tick(base + 0.75, f"{base + 1} of {total} · {name} · traits")
                            table = self._traits(result, corrected, model, report, name, options.qr_mode, ocr_dir)
                            delta = _metric(report, "delta_e_2000", "mean_after")
                            log(f"{name}: {hop_count} {'hops' if measure_names(self.mapping) == ['Hops'] else 'objects'}, {qr_count} QR; {_method_name(model)} ΔE {delta:.2f}.")
                            if table.empty:
                                tick(base + 1, f"{base + 1} of {total} · {name} · no hops")
                                continue
                            picture = corrected
                            if options.save_hops:
                                for source in measure_names(self.mapping):
                                    picture = draw_hops(
                                        result, canvas=picture, class_name=source,
                                        label_prefix=display_for(self.mapping, source),
                                    )
                            hop_path = group / "Image_Predictions" / f"{stem}.jpg"
                            hop_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(hop_path), picture, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            rows.append(table)
                            tick(base + 1, f"{base + 1} of {total} · {name}")
                        if not chart_path.exists():
                            cv2.imwrite(str(chart_path), draw_chart(original, detection), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        log(f"{name}: finished in {time.perf_counter() - started:.1f}s.")
                    except Exception as error:
                        if not _is_memory_error(error):
                            raise
                        release_workspace()
                        raise MemoryError(f'{name}: not enough memory. Reduce the batch size and run again. The folder run is incomplete.') from error
                    finally:
                        _release_result(result)
                        shutil.rmtree(work, ignore_errors=True)
                        release_workspace()
                seen += len(chunk)
                tick(seen, f"{seen} of {total}")
            if rows:
                table = pd.concat(rows, ignore_index=True)
                csv_path = group / "Features.csv"
                table.to_csv(csv_path, index=False)
                log(f"Wrote {len(table)} rows to {csv_path}")
            else:
                log(f"No traits for {folder.name}")
        release_workspace()
        tick(total, f"{total} of {total} · finished")
