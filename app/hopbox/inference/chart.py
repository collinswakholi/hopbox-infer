from . import chart_math as cm
from .backends import OnnxBackend
import cv2
import numpy as np
from pathlib import Path

class ChartDetector:
    def __init__(self, directory, provider='CPUExecutionProvider',device=0):
        self.directory=Path(directory);self.provider=provider;self.device=device
        self.sessions={}

    def forward(self,image,size):
        if size not in self.sessions:
            self.sessions[size]=OnnxBackend(self.directory/f'best_assemble_fp32_{size}.onnx',self.provider,self.device)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        canvas,gain,px,py=cm.letterbox_rgb(rgb,size)
        tensor=np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)/255
        raw=self.sessions[size].run(tensor)[0]
        rows=cm.preds_to_rows(raw)
        decoded=cm.decode_obb(rows,.15 if size==1280 else .05,cm.DECODE_IOU,cm.DEFAULT_MAX_DET)
        return cm.dets_from_raw(decoded,gain,px,py)

    def detect(self,image,path,handshake=None,prefer='classic',force_recall=False):
        h,w=image.shape[:2];size=1536 if force_recall else 1280
        dets=self.forward(image,size)
        charts=cm.assemble_charts(dets,'classic',cm.DEFAULT_CROP,w,h)
        if not charts and size==1280:
            size=1536;dets=self.forward(image,size)
            charts=cm.assemble_charts(dets,'classic',cm.DEFAULT_CROP,w,h)
        if not charts:raise RuntimeError('No chart lattice')
        payload=cm.handshake_payload(charts[0],charts,len(dets),cm.DEFAULT_CROP,path,self.provider,'primary' if size==1280 else 'recall_1536')
        if handshake is not None:
            import json
            target=Path(handshake);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(payload),encoding='utf-8')
            payload['handshake']=str(target)
        return payload
