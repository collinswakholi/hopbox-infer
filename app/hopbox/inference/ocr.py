"""CRAFT + English CRNN ONNX inference with no EasyOCR/Torch imports."""
from pathlib import Path
import json
import math
import cv2
import numpy as np
from PIL import Image
from .backends import OnnxBackend
from .ocr_geometry import craft_utils,imgproc,grouping

class OnnxOCR:
    def __init__(self,directory,provider='CPUExecutionProvider',device=0):
        directory=Path(directory)
        self.detect_net=OnnxBackend(directory/'craft.onnx',provider,device)
        self.recognize_net=OnnxBackend(directory/'english.onnx',provider,device)
        metadata=json.loads((directory/'metadata.json').read_text())
        self.characters=metadata['characters']
        language=metadata.get('lang_char',self.characters)
        self.ignore=[i+1 for i,char in enumerate(self.characters) if char not in language]

    def detect(self,image,width_ths=.7):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        resized,ratio,_=imgproc.resize_aspect_ratio(rgb,2560,cv2.INTER_LINEAR)
        tensor=np.ascontiguousarray(imgproc.normalizeMeanVariance(resized).transpose(2,0,1)[None])
        scores=self.detect_net.run(tensor)[0][0]
        boxes,_,_=craft_utils.getDetBoxes(scores[:,:,0],scores[:,:,1],.7,.4,.4,False)
        boxes=craft_utils.adjustResultCoordinates(boxes,1/ratio,1/ratio)
        flat=[np.asarray(b).astype(np.int32).reshape(-1) for b in boxes]
        horizontal,free=grouping.group_text_box(flat,width_ths=width_ths,add_margin=.1)
        horizontal=[b for b in horizontal if max(b[1]-b[0],b[3]-b[2])>20]
        free=[b for b in free if max(grouping.diff([p[0] for p in b]),grouping.diff([p[1] for p in b]))>20]
        return horizontal,free

    def recognize_line(self,crop,contrast=False):
        if contrast:
            high,low=np.percentile(crop,(90,10))
            if (high-low)/max(10,high+low)<.5:
                crop=np.clip((crop.astype(int)-low+25)*(200/max(10,high-low)),0,255).astype(np.uint8)
        pil=Image.fromarray(crop,'L');w,h=pil.size
        width=max(64,math.ceil(w/h)*64)
        resized_w=min(width,math.ceil(64*w/h))
        pixels=np.asarray(pil.resize((resized_w,64),Image.Resampling.BICUBIC),np.float32)/255
        pixels=(pixels-.5)/.5
        tensor=np.pad(pixels,((0,0),(0,width-resized_w)),mode='edge')[None,None]
        logits=self.recognize_net.run(tensor)[0][0]
        exp=np.exp(logits-logits.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
        if self.ignore:prob[:,self.ignore]=0;prob/=prob.sum(1,keepdims=True)
        ids=prob.argmax(1);confidence=prob.max(1)
        chars=[];previous=-1
        for idx,value in zip(ids,confidence):
            if idx and idx!=previous:
                chars.append(self.characters[int(idx)-1])
            previous=idx
        # Reference confidence includes every nonblank timestep, including
        # repeated characters which CTC removes from the decoded string.
        selected=confidence[ids!=0]
        score=float(np.prod(selected)**(2/math.sqrt(len(selected)))) if len(selected) else 0.
        return ''.join(chars),score

    def readtext(self,image,detail=1,width_ths=.7,rotation_info=None,**unused):
        horizontal,free=self.detect(image,width_ths)
        grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        crops,_=grouping.get_image_list(horizontal,free,grey,model_height=64)
        records=[]
        for box,crop in crops:
            choices=[]
            for rotation in [0]+list(rotation_info or []):
                view=np.rot90(crop,rotation//90) if rotation else crop
                text,score=self.recognize_line(view)
                if score<.1:
                    retry=self.recognize_line(view,True)
                    if retry[1]>score:text,score=retry
                choices.append((text,score))
            text,score=max(choices,key=lambda x:x[1]);records.append((box,text,score))
        return records

    def read_joined(self,image):
        views=[image]
        if 0<min(image.shape[:2])<140:
            views.append(cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC))
        records=[]
        for view in views:records.extend(self.readtext(view))
        if not any(any(c.isdigit() for c in str(record[1])) for record in records):
            records.extend(self.readtext(image,rotation_info=[90,270]))
        return joined_text(records)

def joined_text(records):
    ordered=[]
    for box,text,*_ in records:
        clean=' '.join(str(text).split())
        if clean:
            points=np.asarray(box);ordered.append((float(points[:,1].mean()),float(points[:,0].mean()),clean))
    unique=[]
    for _,__,text in sorted(ordered):
        if text not in unique:unique.append(text)
    return ' '.join(unique)
