"""Compare generic versus optimized DirectML masks on the preserved cohort."""
from pathlib import Path
import gc
import json
import os
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'app'))
from hopbox.inference.model_adapter import Model

photos=sorted((ROOT/'Data/119CANON').glob('*.JPG'))
reference=[]
os.environ.pop('HOPBOX_DML_GENERIC',None)
model=Model(ROOT/'app/models/portable','dml:1')
for path in photos:
    result=model.predict(path)[0]
    result.orig_img=None
    reference.append(result)
model.close();del model;gc.collect()
os.environ['HOPBOX_DML_GENERIC']='1'
model=Model(ROOT/'app/models/portable','dml:1')
records=[]
for path,before in zip(photos,reference):
    after=model.predict(path)[0]
    assert np.array_equal(before.classes,after.classes),path.name
    row={'image':path.name,'detections':len(after.boxes),'minimum_mask_iou':1.,'maximum_area_difference_percent':0.,
         'maximum_score_difference':float(np.max(np.abs(before.scores-after.scores))) if len(after.boxes) else 0.}
    for a,b in zip(before.polygons,after.polygons):
        if np.array_equal(a,b):continue
        points=np.concatenate([a,b]).astype(np.int32)
        low=np.maximum(points.min(0),0);high=np.minimum(points.max(0)+1,before.image_shape[::-1])
        masks=[]
        for polygon in (a,b):
            mask=np.zeros((high[1]-low[1],high[0]-low[0]),np.uint8)
            cv2.fillPoly(mask,[polygon.astype(np.int32)-low],1);masks.append(mask)
        union=np.count_nonzero(masks[0]|masks[1]);intersection=np.count_nonzero(masks[0]&masks[1])
        row['minimum_mask_iou']=min(row['minimum_mask_iou'],intersection/union if union else 1.)
        row['maximum_area_difference_percent']=max(row['maximum_area_difference_percent'],100*abs(np.count_nonzero(masks[0])-np.count_nonzero(masks[1]))/max(1,np.count_nonzero(masks[0])))
    assert row['minimum_mask_iou']>=.999 and row['maximum_area_difference_percent']<=.1 and row['maximum_score_difference']<=.001,row
    records.append(row)
model.close()
out=ROOT/'validation/security-generic-host/mask-parity.json'
out.write_text(json.dumps({'pass':True,'records':records,'note':'Host DirectML comparison; this does not certify the blocked RTX 4090 endpoint.'},indent=2))
print('Generic DirectML masks/counts/classes/scores passed on',len(records),'photos.')
