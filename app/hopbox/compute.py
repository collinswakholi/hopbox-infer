"""Windows GPU discovery without a neural-network framework."""
from dataclasses import dataclass
import ctypes as c
import os
import uuid

@dataclass
class ComputePlan:
    yolo: str
    label: str
    batch: int

@dataclass
class GpuInfo:
    index: int
    name: str
    discrete: bool
    vendor: int = 0

_GPUS=None

def _method(pointer,index,result,*types):
    table=c.cast(pointer,c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(result,c.c_void_p,*types)(table[index])

def windows_gpus():
    global _GPUS
    if _GPUS is not None:return _GPUS
    _GPUS=[]
    if os.name!='nt':return _GPUS
    class Description(c.Structure):
        _fields_=[('name',c.c_wchar*128),('vendor',c.c_uint),('device',c.c_uint),
                  ('subsystem',c.c_uint),('revision',c.c_uint),('video',c.c_size_t),
                  ('system',c.c_size_t),('shared',c.c_size_t),('luid_low',c.c_uint),
                  ('luid_high',c.c_int),('flags',c.c_uint)]
    factory=c.c_void_p()
    try:
        library=c.WinDLL('dxgi.dll')
        create=library.CreateDXGIFactory1
        create.argtypes=[c.c_void_p,c.POINTER(c.c_void_p)];create.restype=c.c_long
        iid=(c.c_ubyte*16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        if create(c.byref(iid),c.byref(factory))!=0:return _GPUS
        enum=_method(factory,12,c.c_long,c.c_uint,c.POINTER(c.c_void_p))
        for index in range(32):
            adapter=c.c_void_p()
            if enum(factory,index,c.byref(adapter))!=0:break
            try:
                desc=Description()
                if _method(adapter,10,c.c_long,c.POINTER(Description))(adapter,c.byref(desc))==0 and not desc.flags&2:
                    _GPUS.append(GpuInfo(index,desc.name,desc.video>=512*1024**2,desc.vendor))
            finally:_method(adapter,2,c.c_ulong)(adapter)
    except (OSError,AttributeError):
        _GPUS=[]
    finally:
        if factory.value:_method(factory,2,c.c_ulong)(factory)
    return _GPUS

def _providers():
    import onnxruntime as ort
    return ort.get_available_providers()

def device_catalog():
    rows=[{'id':'auto','label':'Auto','enabled':True}]
    if 'DmlExecutionProvider' in _providers():
        for gpu in sorted(windows_gpus(),key=lambda x:(not x.discrete,x.index)):
            rows.append({'id':f'dml:{gpu.index}','label':f'GPU · {gpu.name}','enabled':True})
    try:
        import onnxruntime_ep_nv_tensorrt_rtx
        rows.append({'id':'rtx:0','label':'NVIDIA RTX','enabled':True})
    except ImportError:pass
    rows.append({'id':'cpu','label':'CPU','enabled':True})
    return rows

def resolve_choice(choice):
    text=(choice or 'auto').strip()
    if text=='auto':
        override=os.environ.get('HOPBOX_DEVICE','').strip()
        if override and override!='auto':return resolve_choice(override)
        available=[r['id'] for r in device_catalog()]
        if 'rtx:0' in available:return resolve_choice('rtx:0')
        for gpu in sorted(windows_gpus(),key=lambda x:(not x.discrete,x.index)):
            if f'dml:{gpu.index}' in available:return resolve_choice(f'dml:{gpu.index}')
        return resolve_choice('cpu')
    if text=='cpu':return ComputePlan('cpu','CPU · FP32',1)
    if text.startswith('dml:'):
        index=int(text.split(':')[1]);gpu=next((g for g in windows_gpus() if g.index==index),None)
        if gpu is None or 'DmlExecutionProvider' not in _providers():raise RuntimeError('That GPU is unavailable')
        return ComputePlan(text,f'{gpu.name} · FP32',1)
    if text=='rtx:0' and any(r['id']==text for r in device_catalog()):
        return ComputePlan(text,'NVIDIA RTX · FP32',1)
    raise ValueError(f'Unknown compute device: {text}')

def quick_plan():return resolve_choice('auto')
def plan_text(choice,startup):return (startup if choice=='auto' else resolve_choice(choice)).label

def runtime_device(device):
    if str(device).startswith('dml:'):return 'DmlExecutionProvider',int(str(device).split(':')[1])
    if str(device).startswith('rtx:'):return 'NvTensorRTRTXExecutionProvider',int(str(device).split(':')[1])
    if device=='cpu':return 'CPUExecutionProvider',0
    raise ValueError(f'Unknown runtime device: {device}')

