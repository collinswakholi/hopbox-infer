"""QR reading, with an optional OCR fallback."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_qr = None
_ocr = None


def _reader():
    global _qr
    if _qr is None:
        from hopbox.paths import app_root
        directory=app_root()/'models/portable/qr'
        class Reader:
            def __init__(self):
                self.detector=cv2.wechat_qrcode_WeChatQRCode(*[str(directory/name) for name in
                    ('detect.prototxt','detect.caffemodel','sr.prototxt','sr.caffemodel')])
            def decode(self,image):
                return self.detector.detectAndDecode(image)[0]
        _qr = Reader()
    return _qr


def _ocr_reader(storage: Path):
    global _ocr
    if _ocr is None:
        from hopbox.inference.ocr import OnnxOCR
        _ocr = OnnxOCR(storage)
    return _ocr


def _clean_line(word) -> str:
    return " ".join(str(word).split())


def _read_records(reader, image: np.ndarray, rotation: list[int] | None = None) -> list[tuple[float, float, str]]:
    """One ONNX OCR pass as (vertical center, horizontal center, text)."""
    kwargs = {"detail": 1, "width_ths": 0.7}
    if rotation:
        kwargs["rotation_info"] = rotation
    records = []
    for item in reader.readtext(image, **kwargs) or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            text = _clean_line(item)
            if text:
                records.append((0.0, float(len(records)), text))
            continue
        box, word = item[0], item[1]
        text = _clean_line(word)
        if not text:
            continue
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            records.append((sum(ys) / len(ys), sum(xs) / len(xs), text))
        except (TypeError, IndexError, ValueError):
            records.append((0.0, float(len(records)), text))
    return records


def _join_reads(records: list[tuple[float, float, str]]) -> str:
    """Every distinct line in the QR zone, top to bottom and then left to right."""
    parts: list[str] = []
    for _top, _left, text in sorted(records, key=lambda item: (item[0], item[1])):
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


def _ocr_text(reader, image: np.ndarray) -> str:
    views = [image]
    short = min(image.shape[:2])
    if 0 < short < 140:
        views.append(cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    records: list[tuple[float, float, str]] = []
    for view in views:
        records.extend(_read_records(reader, view))
    if not any(any(ch.isdigit() for ch in text) for _top, _left, text in records):
        records.extend(_read_records(reader, image, [90, 270]))
    return _join_reads(records)


def read_qr(image: np.ndarray, mode: str = "qr_ocr", ocr_storage: Path | None = None, label: np.ndarray | None = None) -> str:
    """mode is off, qr, or qr_ocr. A successful decode matches the script's string form.

    When the code itself does not decode, OCR reads the printed label beside it.
    A small vision model would reread that same crop more slowly, so the fallback
    stays on the GPU text reader already installed with the app.
    """
    if mode == "off" or image is None or image.size == 0:
        return "NA"
    try:
        info = _reader().decode(image)
    except Exception:
        info = []
    if info is not None and len(info) > 0:
        return str(info)
    if mode != "qr_ocr":
        return "NA"
    source = label if label is not None and label.size else image
    from hopbox.paths import app_root
    storage = ocr_storage or app_root() / "models" / "portable" / "ocr"
    try:
        text = _ocr_text(_ocr_reader(storage), source)
    except Exception:
        return "NA"
    if not text:
        return "NA"
    return "OCR(" + text + ")"


def close_reader():
    global _ocr
    if _ocr is not None:
        for name in ('detect_net','recognize_net'):
            session=getattr(_ocr,name,None)
            if hasattr(session,'close'):session.close()
        _ocr=None
