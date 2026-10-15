"""Small Rust color kernel, compiled once when the app is built."""
import ctypes as c
import numpy as np
from .color_parameters import _prepare
from hopbox.paths import app_root

_LIB=None

def apply_rust(image,saved):
    global _LIB
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected an eight-bit BGR image')
    model=saved.get('gc_model') or {}
    if any(len(model.get(n) or [])>8 for n in ('knots','values','slopes','coeffs')):
        raise ValueError('Unsupported gamma model size')
    if _LIB is None:
        lib=c.CDLL(str(app_root()/'native/color_cpu.dll'))
        lib.hopbox_color_cpu.argtypes=[c.c_void_p,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t]+[c.c_int]*6
        lib.hopbox_color_cpu.restype=c.c_int
        _LIB=lib
    data=_prepare(saved)
    if image.shape[0]*image.shape[1]<1024:data['do_cc']=0
    parameters=np.concatenate([data[n].reshape(-1) for n in ('lut','rgb','inv','white','wb','knots','values','slopes','coeffs')]).astype(np.float32)
    src=np.ascontiguousarray(image)
    out=np.empty_like(src)
    code=_LIB.hopbox_color_cpu(src.ctypes.data,out.ctypes.data,src.shape[0]*src.shape[1],parameters.ctypes.data,len(parameters),
            *[data[n] for n in ('do_gc','do_wb','do_cc','use_spline','knot_n','coeff_n')])
    if code:raise RuntimeError(f'Rust color processing failed: {code}')
    return out
