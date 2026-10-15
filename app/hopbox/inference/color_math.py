"""Apply a ColorCorrect model.

The fitter still measures the chart. The full photo is corrected here in one
pass, matching the engine: exact gamma and white balance, then the same 33³
color LUT the engine uses on a still image.
The compiled Rust kernel applies this math on the processor.
"""
from __future__ import annotations

import numpy as np

_GS = 33
_LAB_EPS = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


def _xyz_from_xy(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _mat3_inv(matrix: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f, g, h, i = matrix
    ac = e * i - f * h
    bc = -(d * i - f * g)
    cc = d * h - e * g
    dc = -(b * i - c * h)
    ec = a * i - c * g
    fc = -(a * h - b * g)
    gc = b * f - c * e
    hc = -(a * f - c * d)
    ic = a * e - b * d
    det = a * ac + b * bc + c * cc
    inv = 1.0 / det
    return np.array([ac * inv, dc * inv, gc * inv, bc * inv, ec * inv, hc * inv, cc * inv, fc * inv, ic * inv])


def _rgb_to_xyz_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    white = _xyz_from_xy(0.31271, 0.32902)
    red, green, blue = _xyz_from_xy(0.64, 0.33), _xyz_from_xy(0.30, 0.60), _xyz_from_xy(0.15, 0.06)
    primary = np.array([
        red[0], green[0], blue[0],
        red[1], green[1], blue[1],
        red[2], green[2], blue[2],
    ])
    scale = _mat3_inv(primary).reshape(3, 3) @ white
    rgb_to_xyz = np.array([
        scale[0] * red[0], scale[1] * green[0], scale[2] * blue[0],
        scale[0] * red[1], scale[1] * green[1], scale[2] * blue[1],
        scale[0] * red[2], scale[1] * green[2], scale[2] * blue[2],
    ])
    return rgb_to_xyz, _mat3_inv(rgb_to_xyz), white


_RGB_TO_XYZ, _XYZ_TO_RGB, _D65 = _rgb_to_xyz_matrix()


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def _features(linear: np.ndarray, kind: str) -> np.ndarray:
    red = np.maximum(linear[..., 0], 0.0)
    green = np.maximum(linear[..., 1], 0.0)
    blue = np.maximum(linear[..., 2], 0.0)
    if kind == "rpcc3":
        return np.stack([
            red, green, blue,
            np.sqrt(red * green), np.sqrt(red * blue), np.sqrt(green * blue),
            np.cbrt(red * red * green), np.cbrt(green * green * blue), np.cbrt(blue * blue * red),
            np.cbrt(red * green * green), np.cbrt(green * blue * blue), np.cbrt(blue * red * red),
            np.cbrt(red * green * blue),
        ], axis=-1)
    if kind == "rpcc2":
        return np.stack([
            red, green, blue,
            np.sqrt(red * green), np.sqrt(red * blue), np.sqrt(green * blue),
        ], axis=-1)
    return np.stack([red, green, blue], axis=-1)


def _build_lut(correction: dict) -> np.ndarray:
    kind = str(correction.get("kind") or "ccm")
    coeffs = np.asarray(correction["coefficients"], dtype=np.float64).reshape(-1, 3)
    grid = np.arange(_GS, dtype=np.float64) / (_GS - 1)
    red, green, blue = np.meshgrid(grid, grid, grid, indexing="ij")
    encoded = np.stack([red, green, blue], axis=-1)
    features = _features(_srgb_to_linear(encoded), kind)
    # Mesh is (R, G, B). The engine stores (B, G, R).
    linear = np.einsum("...f,fc->...c", features, coeffs)
    lut = _linear_to_srgb(linear).astype(np.float32)
    return np.transpose(lut, (2, 1, 0, 3))


