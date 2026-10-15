"""Hop trait columns, in the same order as the original inference CSV."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from hopbox.inference.science import channel_features


def get_ids(result, class_name: str) -> np.ndarray:
    try:
        return np.array([key for key, name in result.names.items() if name == class_name])
    except Exception:
        return np.array([])


def extract_roi(result, idx: np.ndarray, with_bb: bool = False):
    image = result.orig_img
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    idz = np.where(np.isin(result.classes, idx))[0].tolist()
    images = {}
    polys = {}
    for number, det_id in enumerate(idz):
        bb = result.boxes[det_id].astype(int)
        if with_bb:
            images[number] = image[bb[1]:bb[3], bb[0]:bb[2]]
            polys[number] = None
        else:
            poly = result.polygons[det_id].astype(int)
            mask[:] = 0
            cv2.fillPoly(mask, [poly], 255)
            roi = cv2.bitwise_and(image, image, mask=mask)
            images[number] = roi[bb[1]:bb[3], bb[0]:bb[2]].copy()
            polys[number] = poly
    return images, polys


def crop_class(result, class_name: str, pad: float = 0.0):
    ids = get_ids(result, class_name)
    if len(ids) == 0 or result.boxes is None:
        return None
    hits = np.where(np.isin(result.classes, ids))[0]
    if len(hits) == 0:
        return None
    box = result.boxes[int(hits[0])]
    image = result.orig_img
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    left = max(0, int(np.floor(x1 - dx)))
    top = max(0, int(np.floor(y1 - dy)))
    right = min(width, int(np.ceil(x2 + dx)))
    bottom = min(height, int(np.ceil(y2 + dy)))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _draw_one(image, box, poly, label: str, line_width: int):
    canvas = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), int(line_width * 1.5))
    canvas = cv2.polylines(canvas, [poly], True, (255, 0, 0), line_width)
    return cv2.putText(
        canvas, label, (box[0], max(30, box[1] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), line_width, cv2.LINE_AA,
    )


def draw_named(result, class_name: str, canvas: np.ndarray | None = None, label: str | None = None) -> tuple[np.ndarray, int]:
    """Draw one detector class on a full-resolution image. QR and the chart use this."""
    image = result.orig_img.copy() if canvas is None else canvas
    line_width = max(3, int(round(max(image.shape[:2]) / 700)))
    ids = get_ids(result, class_name)
    if result.boxes is None or len(ids) == 0:
        return image, 0
    hits = np.where(np.isin(result.classes, ids))[0].tolist()
    box_color, poly_color = {
        "QR": ((255, 140, 0), (0, 215, 255)),
        "ColorCard": ((40, 40, 220), (40, 180, 40)),
    }.get(class_name, ((0, 0, 220), (220, 40, 40)))
    for number, det_id in enumerate(hits):
        box = result.boxes[det_id].astype(int)
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, int(line_width * 1.4))
        if bool(result.polygons):
            poly = result.polygons[det_id].astype(int)
            if len(poly) > 1:
                image = cv2.polylines(image, [poly], True, poly_color, line_width, cv2.LINE_AA)
        image = cv2.putText(
            image, f"{label or class_name}_{number}", (box[0], max(28, box[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.8, line_width / 4), (20, 20, 20), line_width, cv2.LINE_AA,
        )
    return image, len(hits)


def draw_hops(
    result,
    line_width: int | None = None,
    canvas: np.ndarray | None = None,
    class_name: str = "Hops",
    label_prefix: str | None = None,
) -> np.ndarray:
    image = result.orig_img.copy() if canvas is None else canvas
    if line_width is None:
        line_width = max(3, int(round(max(image.shape[:2]) / 700)))
    ids = get_ids(result, class_name)
    if result.boxes is None or not result.polygons or len(ids) == 0:
        return image
    prefix = label_prefix or class_name
    hits = np.where(np.isin(result.classes, ids))[0].tolist()
    for number, det_id in enumerate(hits):
        box = result.boxes[det_id].astype(int)
        poly = result.polygons[det_id].astype(int)
        image = _draw_one(image, box, poly, f"{prefix}_ID_{number}", line_width)
    return image


def region_properties(poly: np.ndarray) -> pd.DataFrame:
    cnt = np.array(poly)
    properties = {}
    properties["RP_Area"] = cv2.contourArea(cnt, oriented=False)
    properties["RP_Perimeter"] = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    properties.update({"RP_BB_x": x, "RP_BB_y": y, "RP_BB_w": w, "RP_BB_h": h})
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    for index in range(7):
        properties[f"RP_Hu_{index}"] = hu[index][0]
    properties["RP_Aspect_ratio"] = w / h
    hull = cv2.convexHull(cnt)
    properties["RP_Extent"] = properties["RP_Area"] / (w * h)
    properties["RP_Solidity"] = properties["RP_Area"] / cv2.contourArea(hull)
    properties["RP_Equivalent_diameter"] = np.sqrt(4 * properties["RP_Area"] / np.pi)
    (_cx, _cy), (major, minor), angle = cv2.fitEllipse(cnt)
    properties.update({
        "RP_Orientation": angle,
        "RP_Major_axis_length": major,
        "RP_Minor_axis_length": minor,
    })
    _center, (feret_w, feret_h), _angle = cv2.minAreaRect(cnt)
    properties.update({"RP_Feret_diameter_max": feret_w, "RP_Feret_diameter_min": feret_h})
    return pd.DataFrame(properties, index=[0])


CHANNEL_SUFFIXES = [
    "_Entropy_Smoothness", "_Entropy_Uniformity", "_Entropy_Entropy",
    *[f"_LBP_{index}" for index in range(16)],
    "_Color_Mean", "_Color_Median", "_Color_Std", "_Color_Skew", "_Color_Kurtosis",
    "_GLCM_contrast", "_GLCM_dissimilarity", "_GLCM_correlation",
    "_GLCM_homogeneity", "_GLCM_energy", "_GLCM_ASM",
]

def process_image(img: np.ndarray, poly: np.ndarray) -> pd.DataFrame:
    values = {}
    for index, prefix in enumerate(("Blue", "Green", "Red")):
        values.update(zip((prefix + suffix for suffix in CHANNEL_SUFFIXES), channel_features(img[:, :, index])))
    return pd.concat([region_properties([poly]), pd.DataFrame(values, index=[0])], axis=1)


def hop_features(result, class_names: list[str] | None = None) -> pd.DataFrame:
    names = class_names or ["Hops"]
    frames = []
    for name in names:
        ids = get_ids(result, name)
        images, polys = extract_roi(result, ids)
        if not images:
            continue
        rows = []
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(process_image, images[key], np.array(polys[key])) for key in images]
            for future in futures:
                rows.append(future.result())
        frame = pd.concat(rows, axis=0, ignore_index=True)
        frame["_class"] = name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)
