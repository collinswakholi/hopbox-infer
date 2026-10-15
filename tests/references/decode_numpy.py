"""NumPy/OpenCV implementation of YOLO segmentation decoding.

The observable contract is tested against Ultralytics 8.4.174; no runtime
imports or objects from that package are used here.
"""
from dataclasses import dataclass, field
from time import perf_counter
import cv2
import numpy as np
from .postprocess import nms


@dataclass
class Detections:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    polygons: list
    image_shape: tuple
    names: dict
    masks: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    orig_img: np.ndarray | None = None

    def counts(self):
        return {name: int(np.sum(self.classes == key)) for key, name in self.names.items()}


def letterbox(bgr, size=2048, rectangular=True, stride=32):
    h, w = bgr.shape[:2]
    target = (size, size) if isinstance(size, int) else tuple(size)
    scale = min(target[0] / h, target[1] / w)
    nw, nh = round(w * scale), round(h * scale)
    dw, dh = target[1] - nw, target[0] - nh
    if rectangular:
        dw, dh = dw % stride, dh % stride
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    image = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (w, h) != (nw, nh) else bgr
    image = cv2.copyMakeBorder(image, top, round(dh / 2 + 0.1), left, round(dw / 2 + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def scale_boxes(boxes, input_shape, original_shape):
    h, w = original_shape[:2]
    ih, iw = input_shape
    gain = min(ih / h, iw / w)
    nh, nw = round(h * gain), round(w * gain)
    px, py = round((iw - nw) / 2 - 0.1), round((ih - nh) / 2 - 0.1)
    out = boxes.copy().astype(np.float32)
    out[:, (0, 2)] = (out[:, (0, 2)] - px) / (nw / w)
    out[:, (1, 3)] = (out[:, (1, 3)] - py) / (nh / h)
    out[:, (0, 2)] = np.clip(out[:, (0, 2)], 0, w)
    out[:, (1, 3)] = np.clip(out[:, (1, 3)], 0, h)
    return out


def content_logits(logits, original_shape):
    h, w = original_shape[:2]; mh, mw = logits.shape
    gain = min(mh / h, mw / w)
    nh, nw = round(h * gain), round(w * gain)
    top, left = round((mh - nh) / 2 - 0.1), round((mw - nw) / 2 - 0.1)
    return np.ascontiguousarray(logits[top:top + nh, left:left + nw])


def reconstruct(logits, box, shape, roi=False):
    h, w = shape[:2]
    x1, y1, x2, y2 = np.clip(np.ceil(box).astype(int), (0, 0, 0, 0), (w, h, w, h))
    content = content_logits(logits, shape)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), np.uint8)
    if not roi:
        resized = cv2.resize(content, (w, h), interpolation=cv2.INTER_LINEAR)
        out = (resized > 0).astype(np.uint8)
        out[:y1] = 0; out[y2:] = 0; out[:, :x1] = 0; out[:, x2:] = 0
        return out
    # Sample the ROI in the original full-image coordinate system. Resizing
    # the prototype crop directly to the box would change the mask geometry.
    ch, cw = content.shape
    xs = (np.arange(x1, x2, dtype=np.float32) + 0.5) * (cw / w) - 0.5
    ys = (np.arange(y1, y2, dtype=np.float32) + 0.5) * (ch / h) - 0.5
    xs = np.clip(xs, 0, cw - 1); ys = np.clip(ys, 0, ch - 1)
    xa, ya = np.floor(xs).astype(int), np.floor(ys).astype(int)
    xb, yb = np.minimum(xa + 1, cw - 1), np.minimum(ya + 1, ch - 1)
    wx, wy = xs - xa, ys - ya
    top = content[ya[:, None], xa] * (1 - wx) + content[ya[:, None], xb] * wx
    bottom = content[yb[:, None], xa] * (1 - wx) + content[yb[:, None], xb] * wx
    binary = ((top * (1 - wy[:, None]) + bottom * wy[:, None]) > 0).astype(np.uint8)
    out = np.zeros((h, w), np.uint8)
    out[y1:y2, x1:x2] = binary
    return out


def polygon(mask):
    contours = [c.reshape(-1, 2) for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
    if not contours:
        return np.zeros((0, 2), np.float32)
    if len(contours) == 1:
        return contours[0].astype(np.float32)
    # Join neighbouring contours at closest points with the same two-pass
    # traversal contract as the reference's all-contours strategy.
    links = [[] for _ in contours]
    for i in range(1, len(contours)):
        distances = ((contours[i - 1][:, None] - contours[i][None]) ** 2).sum(-1)
        a, b = np.unravel_index(np.argmin(distances), distances.shape)
        links[i - 1].append(int(a)); links[i].append(int(b))
    segments = []
    for i, ids in enumerate(links):
        seq = contours[i]
        ordered = ids
        if len(ids) == 2 and ids[0] > ids[1]:
            ordered = ids[::-1]; seq = seq[::-1]
        seq = np.roll(seq, -ordered[0], axis=0)
        seq = np.concatenate((seq, seq[:1]))
        contours[i] = seq
        segments.append(seq if i in (0, len(contours) - 1) else seq[:ordered[1] - ordered[0] + 1])
    for i in range(len(contours) - 2, 0, -1):
        segments.append(contours[i][abs(links[i][1] - links[i][0]):])
    return np.concatenate(segments).astype(np.float32)


def decode(prediction, prototypes, shape, input_shape, names, conf=0.58, iou=0.75, roi=False, mask_backend=None,retain_masks=True):
    t = perf_counter()
    pred = np.asarray(prediction)
    if pred.ndim == 3:
        pred = pred[0]
    nc = len(names)
    if pred.shape[0] == 4 + nc + prototypes.shape[-3]:
        pred = pred.T
    cls = pred[:, 4:4 + nc].argmax(1)
    scores = pred[np.arange(len(pred)), 4 + cls]
    rows = pred[scores > conf]
    cls, scores = cls[scores > conf], scores[scores > conf].astype(np.float32)
    xy, wh = rows[:, :2], rows[:, 2:4] / 2
    boxes = np.concatenate((xy - wh, xy + wh), axis=1).astype(np.float32)
    keep = nms(boxes, scores, cls, iou)
    boxes = scale_boxes(boxes[keep], input_shape, shape)
    coeffs = rows[keep, 4 + nc:].astype(np.float32)
    proto = np.asarray(prototypes, np.float32).reshape(prototypes.shape[-3], -1)
    timings = {"nms": perf_counter() - t}
    t = perf_counter()
    logits = (coeffs @ proto).reshape(-1, *prototypes.shape[-2:])
    if not retain_masks and mask_backend is None:
        valid=[];polys=[];mask_seconds=0.;contour_seconds=0.
        for i,(logit,box) in enumerate(zip(logits,boxes)):
            tick=perf_counter();mask=reconstruct(logit,box,shape,roi)
            present=mask.any();mask_seconds+=perf_counter()-tick
            if present:
                tick=perf_counter();polys.append(polygon(mask));contour_seconds+=perf_counter()-tick
                valid.append(i)
            del mask
        timings.update(masks=mask_seconds,contours=contour_seconds)
        return Detections(boxes[valid],scores[keep][valid],cls[keep][valid],polys,tuple(shape[:2]),names,[],timings)
    masks = mask_backend(logits,boxes,shape) if mask_backend else [reconstruct(logit, box, shape, roi) for logit, box in zip(logits, boxes)]
    valid = [i for i, mask in enumerate(masks) if mask.any()]
    masks = [masks[i] for i in valid]
    timings["masks"] = perf_counter() - t
    t = perf_counter(); polys = [polygon(mask) for mask in masks]
    timings["contours"] = perf_counter() - t
    return Detections(boxes[valid], scores[keep][valid], cls[keep][valid], polys, tuple(shape[:2]), names, masks, timings)
