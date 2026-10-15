"""Small native Rust numerical kernels; no compiler in the deployment."""
import ctypes as c
import numpy as np
from hopbox.paths import app_root

_LIBRARY=None

def nms(boxes,scores,classes,threshold=.75,limit=300):
    global _LIBRARY
    boxes=np.ascontiguousarray(boxes,dtype=np.float32)
    scores=np.ascontiguousarray(scores,dtype=np.float32)
    classes=np.ascontiguousarray(classes,dtype=np.int64)
    count=len(scores)
    if boxes.shape!=(count,4) or classes.shape!=(count,):raise ValueError('Invalid NMS dimensions')
    if count==0 or limit<=0:return np.empty(0,np.int64)
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):raise ValueError('Non-finite detection output')
    if _LIBRARY is None:
        library=c.CDLL(str(app_root()/'native/postprocess.dll'))
        library.hopbox_nms.argtypes=[c.c_void_p,c.c_void_p,c.c_void_p,c.c_size_t,c.c_float,c.c_size_t,c.c_void_p]
        library.hopbox_nms.restype=c.c_size_t
        _LIBRARY=library
    output=np.empty(min(count,limit),np.int64)
    kept=_LIBRARY.hopbox_nms(boxes.ctypes.data,scores.ctypes.data,classes.ctypes.data,count,threshold,limit,output.ctypes.data)
    return output[:kept]
