import numpy as np
_MAX_KNOTS = 8

def _lut_rgb(correction: dict) -> np.ndarray:
    from hopbox.inference.color_math import _build_lut
    built = _build_lut(correction)
    return np.ascontiguousarray(np.transpose(built, (2, 1, 0, 3)), dtype=np.float32)

def _pad(values, length: int) -> np.ndarray:
    row = np.zeros(length, dtype=np.float32)
    if values:
        data = np.asarray(values, dtype=np.float64).reshape(-1)[:length]
        row[:data.size] = data
    return row

def _prepare(saved: dict) -> dict:
    from hopbox.inference.color_math import _D65, _RGB_TO_XYZ, _XYZ_TO_RGB
    model = saved.get('gc_model') or {}
    knots = list(model.get('knots') or [])
    values = list(model.get('values') or [])
    slopes = list(model.get('slopes') or [])
    use_spline = len(knots) >= 2 and len(knots) == len(values) == len(slopes)
    correction = saved.get('correction') or {'kind': 'ccm', 'coefficients': [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    wb = np.ones(3, dtype=np.float32)
    if saved.get('do_wb') and saved.get('wb_diag'):
        wb = np.ascontiguousarray(np.asarray(saved['wb_diag'], dtype=np.float64).reshape(3), dtype=np.float32)
    return {'lut': _lut_rgb(correction), 'rgb': np.ascontiguousarray(np.asarray(_RGB_TO_XYZ, dtype=np.float64).reshape(3, 3), dtype=np.float32), 'inv': np.ascontiguousarray(np.asarray(_XYZ_TO_RGB, dtype=np.float64).reshape(3, 3), dtype=np.float32), 'white': np.ascontiguousarray(np.asarray(_D65, dtype=np.float64).reshape(3), dtype=np.float32), 'wb': wb, 'knots': _pad(knots, _MAX_KNOTS), 'values': _pad(values, _MAX_KNOTS), 'slopes': _pad(slopes, _MAX_KNOTS), 'coeffs': _pad(list(model.get('coeffs') or []), _MAX_KNOTS), 'knot_n': min(len(knots), _MAX_KNOTS) if use_spline else 0, 'coeff_n': min(len(list(model.get('coeffs') or [])), _MAX_KNOTS), 'do_gc': 1 if saved.get('do_gc') and saved.get('gc_model') else 0, 'do_wb': 1 if saved.get('do_wb') and saved.get('wb_diag') else 0, 'do_cc': 1 if saved.get('do_cc', True) else 0, 'use_spline': 1 if use_spline else 0}
