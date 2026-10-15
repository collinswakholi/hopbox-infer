"""Opt2 color-chart detector, matching the ColorCorrect app.

The desktop app loads these graphs from ``models/portable``:

  best_assemble_fp32_1280.onnx   primary, FP32
  best_assemble_fp32_1536.onnx   recall pass when the primary finds no chart

Preprocess, oriented-box decode, and lattice assembly follow
``engine/crates/ccp_detect`` (``infer.rs``, ``decode.rs``, ``assemble.rs``, ``geom.rs``).
The result is the same handshake JSON the color fitter consumes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ANCHOR_NAMES = ("tl", "tr", "br", "bl", "top_mid", "bottom_mid", "left_mid", "center")
CHART_TYPES = ("classic_4x6", "spyder_4x6", "digital_sg_10x14")
CHART_ROWS = (4, 4, 10)
CHART_COLS = (6, 6, 14)
NUM_CLASSES = 24
OBB_ROW = 29
PRIMARY_SIZE = 1280
RECALL_SIZE = 1536
DEFAULT_CONF = 0.15
DECODE_IOU = 0.7
NMS_CHARTS_IOU = 0.35
DEFAULT_CROP = 0.75
DEFAULT_MAX_DET = 300
DECODE_CAND_CAP = 30000
PROBIOU_CLASS_OFFSET = 7680.0
PRIMARY_GRAPH = "best_assemble_fp32_1280.onnx"
RECALL_GRAPH = "best_assemble_fp32_1536.onnx"


def _cli_chart(chart_type: str) -> str:
    if chart_type == "spyder_4x6":
        return "spyder"
    if chart_type == "digital_sg_10x14":
        return "digital-sg"
    return "classic"


def _rows_cols(chart_type: str) -> tuple[int, int]:
    index = CHART_TYPES.index(chart_type)
    return CHART_ROWS[index], CHART_COLS[index]


@dataclass
class Vec2:
    x: float
    y: float

    def finite(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y)


@dataclass
class Quad:
    top_left: Vec2
    top_right: Vec2
    bottom_right: Vec2
    bottom_left: Vec2

    def points(self) -> list[Vec2]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def finite(self) -> bool:
        return all(point.finite() for point in self.points())


@dataclass
class Det:
    chart_type: str
    anchor_name: str
    score: float
    quad: Quad


@dataclass
class Chart:
    chart_type: str
    score: float
    corners: Quad
    patch_cells: list[Quad] = field(default_factory=list)
    patch_obbs: list[Quad] = field(default_factory=list)


def quad_center(quad: Quad) -> Vec2:
    points = quad.points()
    return Vec2(sum(point.x for point in points) * 0.25, sum(point.y for point in points) * 0.25)


def order_visual_quad(points: list[Vec2]) -> Quad:
    pts = list(points)
    center = Vec2(sum(point.x for point in pts) * 0.25, sum(point.y for point in pts) * 0.25)
    pts.sort(key=lambda point: math.atan2(point.y - center.y, point.x - center.x))
    area = 0.0
    for index, point in enumerate(pts):
        nxt = pts[(index + 1) % 4]
        area += point.x * nxt.y - point.y * nxt.x
    if area < 0.0:
        pts.reverse()
    start = min(range(4), key=lambda index: pts[index].x + pts[index].y)
    ordered = [pts[(start + offset) % 4] for offset in range(4)]
    return Quad(*ordered)


def scale_quad(quad: Quad, crop: float) -> Quad:
    center = quad_center(quad)
    factor = max(crop, 0.02)

    def scale(point: Vec2) -> Vec2:
        return Vec2(center.x + (point.x - center.x) * factor, center.y + (point.y - center.y) * factor)

    return Quad(*(scale(point) for point in quad.points()))


def dist2(a: Vec2, b: Vec2) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def edge_len(a: Vec2, b: Vec2) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def patch_diag(quad: Quad) -> float:
    return edge_len(quad.top_left, quad.bottom_right)


def quad_area(quad: Quad) -> float:
    pts = quad.points()
    area = 0.0
    for index, point in enumerate(pts):
        nxt = pts[(index + 1) % 4]
        area += point.x * nxt.y - point.y * nxt.x
    return 0.5 * area


def face_edge_imbalance(quad: Quad) -> float:
    edges = [
        edge_len(quad.top_left, quad.top_right),
        edge_len(quad.top_right, quad.bottom_right),
        edge_len(quad.bottom_right, quad.bottom_left),
        edge_len(quad.bottom_left, quad.top_left),
    ]
    if min(edges) <= 1e-6:
        return 1.0
    ratio_a = min(edges[0], edges[2]) / max(edges[0], edges[2])
    ratio_b = min(edges[1], edges[3]) / max(edges[1], edges[3])
    return 1.0 - min(ratio_a, ratio_b)


def lattice_plausible(chart: Chart, width: int, height: int) -> bool:
    if not chart.corners.finite() or any(not patch.finite() for patch in chart.patch_obbs):
        return False
    if abs(quad_area(chart.corners)) < 64.0 or face_edge_imbalance(chart.corners) > 0.38:
        return False
    center = quad_center(chart.corners)
    pad_x = 0.25 * max(width, 1)
    pad_y = 0.25 * max(height, 1)
    if center.x < -pad_x or center.y < -pad_y or center.x > width + pad_x or center.y > height + pad_y:
        return False
    span = edge_len(chart.corners.top_left, chart.corners.bottom_right)
    return span < 4.0 * max(width, height)


def aabb_iou(a: Quad, b: Quad) -> float:
    def box(quad: Quad) -> tuple[float, float, float, float]:
        xs = [point.x for point in quad.points()]
        ys = [point.y for point in quad.points()]
        return min(xs), min(ys), max(xs), max(ys)

    ax0, ay0, ax1, ay1 = box(a)
    bx0, by0, bx1, by1 = box(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 1e-9 else inter / denom


def solve_8x8(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    augmented = np.concatenate([matrix, rhs.reshape(8, 1)], axis=1)
    for col in range(8):
        pivot = col + int(np.argmax(np.abs(augmented[col:, col])))
        if abs(augmented[pivot, col]) < 1e-12:
            return None
        if pivot != col:
            augmented[[col, pivot]] = augmented[[pivot, col]]
        augmented[col, col:] /= augmented[col, col]
        for row in range(8):
            if row == col:
                continue
            augmented[row, col:] -= augmented[row, col] * augmented[col, col:]
    return augmented[:, 8].copy()


def homography_4(src: list[Vec2], dst: list[Vec2]) -> np.ndarray | None:
    design = np.zeros((8, 8))
    target = np.zeros(8)
    for index in range(4):
        x, y = src[index].x, src[index].y
        u, v = dst[index].x, dst[index].y
        r0, r1 = index * 2, index * 2 + 1
        design[r0, 0], design[r0, 1], design[r0, 2] = x, y, 1.0
        design[r0, 6], design[r0, 7], target[r0] = -u * x, -u * y, u
        design[r1, 3], design[r1, 4], design[r1, 5] = x, y, 1.0
        design[r1, 6], design[r1, 7], target[r1] = -v * x, -v * y, v
    solved = solve_8x8(design, target)
    if solved is None:
        return None
    return np.array([*solved, 1.0])


def warp(homography: np.ndarray, point: Vec2) -> Vec2:
    z = homography[6] * point.x + homography[7] * point.y + homography[8]
    if abs(z) < 1e-12:
        return point
    return Vec2(
        (homography[0] * point.x + homography[1] * point.y + homography[2]) / z,
        (homography[3] * point.x + homography[4] * point.y + homography[5]) / z,
    )


def anchors_for(rows: int, cols: int) -> list[tuple[str, int, int]]:
    return [
        ("tl", 0, 0),
        ("tr", 0, cols - 1),
        ("br", rows - 1, cols - 1),
        ("bl", rows - 1, 0),
        ("top_mid", 0, cols // 2),
        ("bottom_mid", rows - 1, cols // 2),
        ("left_mid", rows // 2, 0),
        ("center", rows // 2, cols // 2),
    ]


def canonical_center(row: int, col: int, rows: int, cols: int) -> Vec2:
    return Vec2((col + 0.5) / cols, (row + 0.5) / rows)


def median_nth(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def covariance(width: float, height: float, angle: float) -> tuple[float, float, float]:
    aa = (width * width) / 12.0
    bb = (height * height) / 12.0
    c = math.cos(angle)
    s = math.sin(angle)
    return aa * c * c + bb * s * s, aa * s * s + bb * c * c, (aa - bb) * c * s


def probiou(ax, ay, aw, ah, ar, bx, by, bw, bh, br) -> float:
    eps = 1e-7
    a_a, a_b, a_c = covariance(aw, ah, ar)
    b_a, b_b, b_c = covariance(bw, bh, br)
    dx = ax - bx
    dy = ay - by
    den = (a_a + b_a) * (a_b + b_b) - (a_c + b_c) ** 2 + eps
    t1 = (((a_a + b_a) * dy * dy + (a_b + b_b) * dx * dx) / den) * 0.25
    t2 = (((a_c + b_c) * (bx - ax) * dy) / den) * 0.5
    det1 = max(a_a * a_b - a_c * a_c, 0.0)
    det2 = max(b_a * b_b - b_c * b_c, 0.0)
    t3 = math.log(((a_a + b_a) * (a_b + b_b) - (a_c + b_c) ** 2) / (4.0 * math.sqrt(det1 * det2) + eps) + eps) * 0.5
    bd = min(max(t1 + t2 + t3, eps), 100.0)
    hd = max(math.sqrt(max(1.0 - math.exp(-bd) + eps, 0.0)), 0.0)
    return 1.0 - hd


def letterbox_rgb(rgb: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, float, float]:
    height, width = rgb.shape[:2]
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    if width <= 0 or height <= 0 or imgsz <= 0:
        return canvas, 1.0, 0.0, 0.0
    gain = min(imgsz / height, imgsz / width)
    nh = int(round(height * gain))
    nw = int(round(width * gain))
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[top:top + nh, left:left + nw] = np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8)
    return canvas, gain, float(left), float(top)


def preds_to_rows(data: np.ndarray) -> np.ndarray:
    """Match ``preds_to_obb_rows``: accept ``[N,29]``, ``[29,N]``, or a leading batch of 1."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] == OBB_ROW:
        return np.ascontiguousarray(arr.T)
    if arr.ndim == 2 and arr.shape[1] == OBB_ROW:
        return np.ascontiguousarray(arr)
    flat = np.ascontiguousarray(arr).reshape(-1)
    if flat.size >= OBB_ROW and flat.size % OBB_ROW == 0:
        return flat.reshape(-1, OBB_ROW)
    return np.zeros((0, OBB_ROW), np.float32)


def decode_obb(rows: np.ndarray, conf: float, iou: float, max_det: int) -> list[tuple]:
    if len(rows) == 0:
        return []
    scores = rows[:, 4:4 + NUM_CLASSES]
    classes = np.argmax(scores, axis=1).astype(np.int32)
    best = scores[np.arange(len(rows)), classes]
    keep = np.flatnonzero(best >= conf)
    if keep.size == 0:
        return []
    order = keep[np.argsort(-best[keep])]
    if order.size > DECODE_CAND_CAP:
        order = order[:DECODE_CAND_CAP]
    cands = [
        (
            float(rows[index, 0]),
            float(rows[index, 1]),
            float(rows[index, 2]),
            float(rows[index, 3]),
            float(rows[index, 4 + NUM_CLASSES]),
            float(best[index]),
            int(classes[index]),
        )
        for index in order
    ]
    suppressed = [False] * len(cands)
    kept = []
    limit = max(max_det, 1)
    for i, cand in enumerate(cands):
        if suppressed[i]:
            continue
        kept.append(cand)
        if len(kept) >= limit:
            break
        ax = cand[0] + cand[6] * PROBIOU_CLASS_OFFSET
        for j in range(i + 1, len(cands)):
            other = cands[j]
            if suppressed[j] or cand[6] != other[6]:
                continue
            overlap = probiou(
                ax, cand[1], cand[2], cand[3], cand[4],
                other[0] + other[6] * PROBIOU_CLASS_OFFSET, other[1], other[2], other[3], other[4],
            )
            if overlap > iou:
                suppressed[j] = True
    return kept


def dets_from_raw(raw: list[tuple], gain: float, pad_x: float, pad_y: float) -> list[Det]:
    dets = []
    for x, y, w, h, angle, score, cls in raw:
        if cls < 0 or cls >= NUM_CLASSES:
            continue
        type_i, anchor_i = divmod(cls, 8)
        if type_i >= len(CHART_TYPES) or anchor_i >= len(ANCHOR_NAMES):
            continue
        cx = (x - pad_x) / gain
        cy = (y - pad_y) / gain
        ww = w / gain
        hh = h / gain
        c = math.cos(angle)
        s = math.sin(angle)
        vx, vy = (ww / 2.0) * c, (ww / 2.0) * s
        ux, uy = (-hh / 2.0) * s, (hh / 2.0) * c
        quad = order_visual_quad([
            Vec2(cx + vx + ux, cy + vy + uy),
            Vec2(cx + vx - ux, cy + vy - uy),
            Vec2(cx - vx - ux, cy - vy - uy),
            Vec2(cx - vx + ux, cy - vy + uy),
        ])
        dets.append(Det(CHART_TYPES[type_i], ANCHOR_NAMES[anchor_i], score, quad))
    dets.sort(key=lambda det: det.score, reverse=True)
    deduped = []
    for det in dets:
        center = quad_center(det.quad)
        if any(dist2(center, quad_center(prev.quad)) < 9.0 for prev in deduped):
            continue
        deduped.append(det)
    return deduped


def assemble_one(dets: list[Det], chart_type: str, crop: float) -> Chart | None:
    rows, cols = _rows_cols(chart_type)
    specs = anchors_for(rows, cols)
    spec_of = {name: (row, col) for name, row, col in specs}
    best: dict[str, Det] = {}
    for det in dets:
        if det.chart_type != chart_type:
            continue
        previous = best.get(det.anchor_name)
        if previous is None or det.score > previous.score:
            best[det.anchor_name] = det
    if len(best) < 4:
        return None
    pairs = []
    for name in ("tl", "tr", "br", "bl"):
        det = best.get(name)
        if det is not None and name in spec_of:
            row, col = spec_of[name]
            pairs.append((canonical_center(row, col, rows, cols), quad_center(det.quad)))
    if len(pairs) < 4:
        pairs = []
        for name, det in best.items():
            if name not in spec_of:
                continue
            row, col = spec_of[name]
            pairs.append((canonical_center(row, col, rows, cols), quad_center(det.quad)))
            if len(pairs) == 4:
                break
    if len(pairs) < 4:
        return None
    homography = homography_4([item[0] for item in pairs[:4]], [item[1] for item in pairs[:4]])
    if homography is None:
        return None
    corners = order_visual_quad([
        warp(homography, Vec2(0.0, 0.0)),
        warp(homography, Vec2(1.0, 0.0)),
        warp(homography, Vec2(1.0, 1.0)),
        warp(homography, Vec2(0.0, 1.0)),
    ])
    if not corners.finite():
        return None
    cells = []
    obbs = []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = col / cols, (col + 1) / cols
            y0, y1 = row / rows, (row + 1) / rows
            cell = order_visual_quad([
                warp(homography, Vec2(x0, y0)),
                warp(homography, Vec2(x1, y0)),
                warp(homography, Vec2(x1, y1)),
                warp(homography, Vec2(x0, y1)),
            ])
            if not cell.finite():
                return None
            cells.append(cell)
            obbs.append(scale_quad(cell, crop))
    cell_size = 0.5 * (
        0.5 * (edge_len(corners.top_left, corners.top_right) + edge_len(corners.bottom_left, corners.bottom_right)) / cols
        + 0.5 * (edge_len(corners.top_left, corners.bottom_left) + edge_len(corners.top_right, corners.bottom_right)) / rows
    )
    if cell_size > 1e-6:
        named_err = []
        for name, det in best.items():
            if name not in spec_of:
                continue
            row, col = spec_of[name]
            predicted = warp(homography, canonical_center(row, col, rows, cols))
            named_err.append(edge_len(predicted, quad_center(det.quad)) / cell_size)
        med = median_nth(named_err)
        if med is not None and med > 0.45:
            return None
    score = sum(det.score for det in best.values()) / len(best)
    return Chart(chart_type, score, corners, cells, obbs)


def assemble_type(dets: list[Det], chart_type: str, crop: float, width: int, height: int, charts: list[Chart]) -> None:
    if chart_type not in CHART_TYPES:
        return
    rows, cols = _rows_cols(chart_type)
    remaining = [index for index, det in enumerate(dets) if det.chart_type == chart_type]
    if len(remaining) < 4:
        return
    name_counts: dict[str, int] = {}
    diags = []
    by_name: dict[str, list[Vec2]] = {}
    for index in remaining:
        det = dets[index]
        name_counts[det.anchor_name] = name_counts.get(det.anchor_name, 0) + 1
        diags.append(max(patch_diag(det.quad), 1.0))
        by_name.setdefault(det.anchor_name, []).append(quad_center(det.quad))
    dup_names = 0
    seps = []
    for points in by_name.values():
        if len(points) < 2:
            continue
        dup_names += 1
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                seps.append(math.sqrt(dist2(points[i], points[j])))
    pair_sep = median_nth(seps)
    if pair_sep is None:
        pair_sep = -1.0
    med_diag = median_nth(diags) or 1.0
    two_face = dup_names >= 3 and pair_sep > 2.0 * med_diag
    cell_span = math.hypot(rows, cols)

    def is_mid(name: str) -> bool:
        return name in {"center", "top_mid", "bottom_mid", "left_mid"}

    def is_corner(name: str) -> bool:
        return name in {"tl", "tr", "br", "bl"}

    def rank_anchor(det: Det, center: Vec2, radius: float) -> float:
        dist = math.sqrt(dist2(quad_center(det.quad), center))
        soft = max(radius * 0.35, 1.0)
        score = max(det.score, 1e-3)
        inv_r = 1.0 / max(radius, 1.0)
        if two_face and name_counts.get(det.anchor_name, 0) >= 2:
            return score / (dist + soft) - 0.08 * dist * inv_r
        if is_corner(det.anchor_name):
            return (score * score) / (dist + soft) + 0.15 * score
        return score / (dist + soft) - 0.05 * dist * inv_r

    while len(remaining) >= 4:
        seed = remaining[0]
        have_mid = False
        for index in remaining:
            det = dets[index]
            if is_mid(det.anchor_name):
                if not have_mid or det.score > dets[seed].score:
                    seed = index
                    have_mid = True
            elif not have_mid and det.score > dets[seed].score:
                seed = index
        seed_det = dets[seed]
        diag = max(patch_diag(seed_det.quad), 1.0)
        radius = 0.55 * diag * cell_span / 0.7
        if two_face:
            radius = min(radius, 0.82 * pair_sep)
        radius2 = radius * radius
        seed_c = quad_center(seed_det.quad)
        bucket: dict[str, int] = {}
        rank: dict[str, float] = {}
        for index in remaining:
            det = dets[index]
            if dist2(quad_center(det.quad), seed_c) > radius2:
                continue
            value = rank_anchor(det, seed_c, radius)
            if det.anchor_name not in rank or value > rank[det.anchor_name]:
                bucket[det.anchor_name] = index
                rank[det.anchor_name] = value
        if len(bucket) >= 3:
            recentre = Vec2(0.0, 0.0)
            n_c = 0
            for name in ("tl", "tr", "br", "bl"):
                index = bucket.get(name)
                if index is None:
                    continue
                center = quad_center(dets[index].quad)
                recentre.x += center.x
                recentre.y += center.y
                n_c += 1
            if n_c < 2:
                recentre = Vec2(0.0, 0.0)
                n_c = 0
                for index in bucket.values():
                    center = quad_center(dets[index].quad)
                    recentre.x += center.x
                    recentre.y += center.y
                    n_c += 1
            if n_c > 0:
                recentre.x /= n_c
                recentre.y /= n_c
                refined: dict[str, int] = {}
                refined_rank: dict[str, float] = {}
                for index in remaining:
                    det = dets[index]
                    if dist2(quad_center(det.quad), recentre) > radius2:
                        continue
                    value = rank_anchor(det, recentre, radius)
                    if det.anchor_name not in refined_rank or value > refined_rank[det.anchor_name]:
                        refined[det.anchor_name] = index
                        refined_rank[det.anchor_name] = value
                for name, index in bucket.items():
                    new_index = refined.get(name)
                    if new_index is None:
                        refined[name] = index
                    else:
                        old_s = dets[index].score
                        new_s = dets[new_index].score
                        if old_s >= 0.85 and old_s > new_s + 0.12:
                            refined[name] = index
                bucket = refined
        if len(bucket) < 4:
            remaining = [index for index in remaining if index != seed]
            continue
        cluster = [dets[index] for index in bucket.values()]
        used = list(bucket.values())
        accepted = False
        cand = assemble_one(cluster, chart_type, crop)
        if cand is not None and lattice_plausible(cand, width, height):
            cluster_diag = sum(max(patch_diag(det.quad), 1.0) for det in cluster) / len(cluster)
            face_span = patch_diag(cand.corners)
            expected = cluster_diag * cell_span / 0.7
            if face_span <= 2.2 * expected and face_span >= 0.2 * expected:
                charts.append(cand)
                accepted = True
        if accepted:
            used_set = set(used)
            remaining = [index for index in remaining if index not in used_set]
        else:
            remaining = [index for index in remaining if index != seed]


def nms_charts(charts: list[Chart]) -> list[Chart]:
    ordered = sorted(charts, key=lambda chart: chart.score, reverse=True)
    kept = []
    for chart in ordered:
        if any(aabb_iou(chart.corners, prev.corners) >= NMS_CHARTS_IOU for prev in kept):
            continue
        kept.append(chart)
    return kept


def assemble_charts(dets: list[Det], prefer: str, crop: float, width: int, height: int) -> list[Chart]:
    order = list(CHART_TYPES)
    if prefer in {"spyder", "spyder_4x6"}:
        order = ["spyder_4x6", "classic_4x6", "digital_sg_10x14"]
    elif prefer in {"digital-sg", "digital_sg_10x14"}:
        order = ["digital_sg_10x14", "classic_4x6", "spyder_4x6"]
    elif prefer in {"classic", "classic_4x6"}:
        order = ["classic_4x6", "spyder_4x6", "digital_sg_10x14"]
    charts: list[Chart] = []
    for chart_type in order:
        assemble_type(dets, chart_type, crop, width, height, charts)
    if not charts:
        for chart_type in order:
            remapped = [Det(chart_type, det.anchor_name, det.score, det.quad) for det in dets]
            assemble_type(remapped, chart_type, crop, width, height, charts)
    charts = nms_charts(charts)
    if prefer and charts:
        preferred = order[0]
        charts = [chart for chart in charts if chart.chart_type == preferred] + [
            chart for chart in charts if chart.chart_type != preferred
        ]
    return charts


def _xy(point: Vec2) -> list[float]:
    return [point.x if math.isfinite(point.x) else 0.0, point.y if math.isfinite(point.y) else 0.0]


def _quad_points(quad: Quad) -> list[list[float]]:
    return [_xy(point) for point in quad.points()]


def _format_quad(quad: Quad) -> str:
    return ";".join(f"{point.x:.3f},{point.y:.3f}" for point in quad.points())


def handshake_payload(chart: Chart, charts: list[Chart], n_dets: int, crop: float, image_path: Path, device: str, recall_source: str) -> dict:
    rows, cols = _rows_cols(chart.chart_type)

    def face(item: Chart, index: int) -> dict:
        face_rows, face_cols = _rows_cols(item.chart_type)
        return {
            "id": index,
            "chart_type": item.chart_type,
            "cli_chart": _cli_chart(item.chart_type),
            "score": item.score,
            "accepted": True,
            "rows": face_rows,
            "cols": face_cols,
            "quad": _format_quad(item.corners),
            "corners": {
                "tl": _xy(item.corners.top_left),
                "tr": _xy(item.corners.top_right),
                "br": _xy(item.corners.bottom_right),
                "bl": _xy(item.corners.bottom_left),
            },
            "patch_cells": [_quad_points(cell) for cell in item.patch_cells],
            "patch_obbs": [_quad_points(cell) for cell in item.patch_obbs],
        }

    return {
        "ok": True,
        "backend": "onnx",
        "runtime": "onnxruntime",
        "device": device,
        "chart_type": chart.chart_type,
        "cli_chart": _cli_chart(chart.chart_type),
        "score": chart.score,
        "accepted": True,
        "crop": crop,
        "rows": rows,
        "cols": cols,
        "n_dets": n_dets,
        "n_charts": len(charts),
        "selected": 0,
        "use_all": False,
        "recall_source": recall_source,
        "image": str(image_path).replace("\\", "/"),
        "quad": _format_quad(chart.corners),
        "corners": {
            "tl": _xy(chart.corners.top_left),
            "tr": _xy(chart.corners.top_right),
            "br": _xy(chart.corners.bottom_right),
            "bl": _xy(chart.corners.bottom_left),
        },
        "patch_cells": [_quad_points(cell) for cell in chart.patch_cells],
        "patch_obbs": [_quad_points(cell) for cell in chart.patch_obbs],
        "charts": [face(item, index) for index, item in enumerate(charts)],
    }


