"""Validated Rust scientific kernels, called through a small C interface."""
import ctypes as c
import numpy as np
from hopbox.paths import app_root

_LIB=None
def library():
    global _LIB
    if _LIB is None:
        lib=c.CDLL(str(app_root()/'native/science.dll'))
        lib.hopbox_channel_features.argtypes=[c.c_void_p,c.c_size_t,c.c_size_t,c.c_void_p]
        lib.hopbox_lbp.argtypes=[c.c_void_p,c.c_size_t,c.c_size_t,c.c_void_p]
        lib.hopbox_mask_roi.argtypes=[c.c_void_p]+[c.c_size_t]*8+[c.c_void_p]
        lib.hopbox_closest_contours.argtypes=[c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t,c.c_void_p]
        lib.hopbox_components.argtypes=[c.c_void_p,c.c_size_t,c.c_size_t,c.c_void_p,c.c_void_p,c.c_void_p,c.c_size_t]
        for name in ('hopbox_channel_features','hopbox_lbp','hopbox_mask_roi','hopbox_closest_contours','hopbox_components'):getattr(lib,name).restype=c.c_int
        _LIB=lib
    return _LIB

def channel_features(image):
    if image.dtype!=np.uint8 or image.ndim!=2 or not image.size:raise ValueError('Expected a nonempty eight-bit channel')
    image=np.ascontiguousarray(image);out=np.empty(30,np.float64)
    code=library().hopbox_channel_features(image.ctypes.data,*image.shape,out.ctypes.data)
    if code:raise RuntimeError(f'Scientific kernel failed: {code}')
    return out

def lbp_image(image):
    image=np.ascontiguousarray(image,dtype=np.uint8);out=np.empty_like(image)
    if library().hopbox_lbp(image.ctypes.data,*image.shape,out.ctypes.data):raise RuntimeError('LBP failed')
    return out

def mask_roi(content,shape,bounds):
    content=np.ascontiguousarray(content,dtype=np.float32);h,w=shape[:2];x0,y0,x1,y1=map(int,bounds)
    if x1<=x0 or y1<=y0:return np.zeros((0,0),np.uint8)
    out=np.empty((y1-y0,x1-x0),np.uint8)
    if out.size:
        code=library().hopbox_mask_roi(content.ctypes.data,*content.shape,h,w,x0,y0,x1,y1,out.ctypes.data)
        if code:raise RuntimeError(f'Mask sampling failed: {code}')
    return out

def closest_contours(a,b):
    a=np.ascontiguousarray(a,dtype=np.int32);b=np.ascontiguousarray(b,dtype=np.int32);out=np.empty(2,np.uintp)
    if library().hopbox_closest_contours(a.ctypes.data,len(a),b.ctypes.data,len(b),out.ctypes.data):raise RuntimeError('Contour joining failed')
    return tuple(int(v) for v in out)

def connected_components(image):
    image=np.ascontiguousarray(image!=0,dtype=np.uint8)
    if image.ndim!=2 or not image.size:raise ValueError('Expected a nonempty binary image')
    labels=np.empty(image.shape,np.int32)
    # A 4-connected image has at most ceil(pixel_count/2) foreground regions.
    capacity=(image.size+1)//2+1
    stats=np.empty((capacity,5),np.int32);centroids=np.empty((capacity,2),np.float64)
    count=library().hopbox_components(image.ctypes.data,*image.shape,labels.ctypes.data,stats.ctypes.data,centroids.ctypes.data,capacity)
    if count<0:raise RuntimeError(f'Connected components failed: {count}')
    return count,labels,stats[:count].copy(),centroids[:count].copy()
