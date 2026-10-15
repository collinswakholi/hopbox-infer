import numpy as np

def chart_window(bgr: np.ndarray, handshake: dict, margin: int = 8):
    """Crop to the chart. Returns None when the chart already fills the photo."""
    charts = handshake.get("charts") or [handshake]
    if not charts:
        return None
    selected = int(handshake.get("selected") or 0)
    face = charts[min(selected, len(charts) - 1)]
    quads = face.get("patch_obbs") or face.get("patch_cells") or []
    xs, ys = [], []
    for quad in quads:
        for point in quad:
            if isinstance(point, dict):
                xs.append(float(point["x"]))
                ys.append(float(point["y"]))
            else:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    if not xs:
        return None
    height, width = bgr.shape[:2]
    x0 = max(0, int(np.floor(min(xs))) - margin)
    y0 = max(0, int(np.floor(min(ys))) - margin)
    x1 = min(width, int(np.ceil(max(xs))) + margin + 1)
    y1 = min(height, int(np.ceil(max(ys))) + margin + 1)
    if (x1 - x0) * (y1 - y0) > 0.55 * height * width:
        return None
    return bgr[y0:y1, x0:x1], x0, y0

def shift_handshake(handshake: dict, x0: int, y0: int) -> dict:
    import copy
    shifted = copy.deepcopy(handshake)

    def move(point):
        if isinstance(point, dict):
            point["x"] = float(point["x"]) - x0
            point["y"] = float(point["y"]) - y0
            return point
        return [float(point[0]) - x0, float(point[1]) - y0]

    def move_face(face):
        for key in ("patch_obbs", "patch_cells"):
            if key in face:
                face[key] = [[move(point) for point in quad] for quad in face[key]]

    move_face(shifted)
    for face in shifted.get("charts") or []:
        move_face(face)
    return shifted
