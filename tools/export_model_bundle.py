"""Development-only .pt exporter. Not included in the deployed application."""
from pathlib import Path
import argparse,hashlib,json
parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('destination',type=Path);parser.add_argument('--imgsz',type=int,default=2048)
args=parser.parse_args()
import torch
from ultralytics import YOLO
model=YOLO(str(args.checkpoint))
if model.task!='segment':raise ValueError('A segmentation checkpoint is required')
model.model.fuse();network=model.model.cpu().float().eval();head=network.model[-1]
head.export=True;head.format='onnx';head.dynamic=True
args.destination.mkdir(parents=True,exist_ok=True)
path=args.destination/(args.checkpoint.stem+'_fp32.onnx')
torch.onnx.export(network,torch.zeros(1,3,args.imgsz,args.imgsz),str(path),opset_version=17,dynamo=False,
    input_names=['images'],output_names=['predictions','prototypes'],
    dynamic_axes={'images':{0:'batch',2:'height',3:'width'},'predictions':{0:'batch',2:'anchors'},'prototypes':{0:'batch',2:'mask_height',3:'mask_width'}})
metadata={'task':'segment','names':model.names,'stride':int(network.stride.max()),'normalization':'RGB/255','mask_dim':int(head.nm),'recommended_imgsz':args.imgsz,
          'input':'images','outputs':['predictions','prototypes'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Portable model exported to',args.destination)
