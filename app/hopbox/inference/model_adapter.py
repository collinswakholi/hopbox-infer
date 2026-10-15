"""Portable model bundles and plain NumPy detection results."""
from pathlib import Path
from collections import OrderedDict
import cv2
import numpy as np
from .model import load_bundle
from .decode import decode,letterbox
from .backends import OnnxBackend
from hopbox.compute import runtime_device

class Model:
    def __init__(self,path,device='cpu'):
        path=Path(path)
        self.directory=path if path.is_dir() else path.parent
        self.graph,self.names,self.metadata=load_bundle(path,'fp32')
        if path.is_file() and path.resolve()!=self.graph.resolve():
            raise ValueError('Choose the FP32 ONNX graph from a portable model bundle')
        self.device=device
        self.sessions=OrderedDict()

    def _session(self,shape,batch=1):
        key=(batch,*shape)
        if key not in self.sessions:
            provider,index=runtime_device(self.device)
            # A large batch session can retain gigabytes of device allocations.
            # Keep only the active profile; tails must release the full batch.
            if self.sessions:
                _,old=self.sessions.popitem(last=False);old.close()
            self.sessions[key]=OnnxBackend(self.graph,provider,index,shape=shape,batch=batch)
        return self.sessions[key]

    def predict(self,source,imgsz=2048,conf=.58,iou=.75,stream=False):
        sources=source if isinstance(source,(list,tuple)) else [source]
        def generate():
            groups=OrderedDict()
            for position,item in enumerate(sources):
                image=cv2.imread(str(item)) if isinstance(item,(str,Path)) else item
                if image is None:raise FileNotFoundError(item)
                tensor=letterbox(image,size=imgsz,stride=self.metadata.get('stride',32))
                shape=tuple(tensor.shape[-2:])
                groups.setdefault(shape,[]).append((position,item,image,tensor))
            results=[None]*len(sources)
            for shape,items in groups.items():
                # Preserve each photo's rectangular preprocessing. Mixed aspect
                # ratios use separate profiles rather than changing padding.
                tensor=items[0][3] if len(items)==1 else np.concatenate([entry[3] for entry in items])
                raw=self._session(shape,len(items)).run(tensor)
                for index,(position,item,image,_) in enumerate(items):
                    result=decode(*(output[index:index+1] for output in raw),image.shape,shape,self.names,conf=conf,iou=iou,roi=True,retain_masks=False)
                    result.orig_img=image
                    result.path=str(item) if isinstance(item,(str,Path)) else ''
                    results[position]=result
                del raw,tensor
            # Return in source order even when aspect-ratio groups interleave.
            yield from results
        return generate() if stream else list(generate())

    def close(self):
        for session in self.sessions.values():session.close()
        self.sessions.clear()
