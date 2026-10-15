"""Explicit ONNX Runtime accelerator selection."""
from pathlib import Path
import json
import os
import numpy as np

_RTX_REGISTERED=False

class OnnxBackend:
    def __init__(self,model,provider='CPUExecutionProvider',device=0,shape=None,profile=None,batch=1,dml_options=None):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        options=ort.SessionOptions()
        options.intra_op_num_threads=24
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        # CPU arenas retain peak allocations between chart/OCR/hop runs. In
        # low-memory offline VMs a second folder could exhaust memory even
        # after the hop session was closed. Release those CPU buffers promptly.
        options.enable_cpu_mem_arena=provider!='CPUExecutionProvider'
        options.enable_mem_pattern=provider not in ('DmlExecutionProvider','CPUExecutionProvider')
        options.log_severity_level=3
        if shape:
            for name,value in zip(('batch','height','width'),(batch,*shape)):
                options.add_free_dimension_override_by_name(name,int(value))
        if profile:
            options.enable_profiling=True
            options.profile_file_prefix=str(profile)
        if provider=='NvTensorRTRTXExecutionProvider':
            global _RTX_REGISTERED
            import onnxruntime_ep_nv_tensorrt_rtx as plugin
            if not _RTX_REGISTERED:
                ort.register_execution_provider_library(plugin.get_ep_name(),plugin.get_library_path())
                _RTX_REGISTERED=True
            devices=[d for d in ort.get_ep_devices() if 'tensorrt' in d.ep_name.lower()]
            if not devices:raise RuntimeError('No compatible RTX accelerator was found')
            if not 0<=device<len(devices):raise ValueError('Unknown RTX device')
            from hopbox.paths import state_root
            cache=state_root()/'cache/rtx'
            cache.mkdir(parents=True,exist_ok=True)
            options.add_provider_for_devices([devices[device]],{'nv_runtime_cache_path':str(cache)})
            providers=None
        else:
            if provider not in ort.get_available_providers():
                raise RuntimeError(f'{provider} is unavailable in this installation')
            if provider=='DmlExecutionProvider' and dml_options is None:
                from hopbox.compute import windows_gpus
                adapter=next((g for g in windows_gpus() if g.index==device),None)
                # Intel UHD 770 metacommands zeroed every batch output after
                # the first on YOLO11s. Generic DML kernels passed parity.
                if os.environ.get('HOPBOX_DML_GENERIC')=='1' or adapter is not None and adapter.vendor==0x8086:
                    dml_options={'disable_metacommands':'true'}
            providers=[provider] if provider=='CPUExecutionProvider' else [(provider,{'device_id':device,**(dml_options or {})})]
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=providers)
        self.session.disable_fallback()
        self.provider=provider
        self.actual_providers=self.session.get_providers()
        if provider=='NvTensorRTRTXExecutionProvider':
            if not any('tensorrt' in p.lower() for p in self.actual_providers):raise RuntimeError('RTX initialization fell back')
        elif self.actual_providers[0]!=provider:
            raise RuntimeError(f'Accelerator initialization fell back to {self.actual_providers}')
        self.input=self.session.get_inputs()[0].name
        self.dtype=np.float16 if self.session.get_inputs()[0].type=='tensor(float16)' else np.float32

    def run(self,tensor):
        if self.session is None:raise RuntimeError('Inference session is closed')
        return self.session.run(None,{self.input:np.ascontiguousarray(tensor,dtype=self.dtype)})

    def close(self):
        self.session=None

    def finish_profile(self):
        if not self.session.get_session_options().enable_profiling:return {}
        path=self.session.end_profiling()
        events=json.loads(Path(path).read_text())
        return {'path':path,'providers':sorted({e.get('args',{}).get('provider') for e in events if e.get('args',{}).get('provider')})}
