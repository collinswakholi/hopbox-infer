"""Run real multi-image inference and compare masks, IDs and numeric outputs."""
from pathlib import Path
import argparse,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from hopbox.inference.model_adapter import Model
from hopbox.inference.decode import polygon

paths=sorted((ROOT/'Data/119CANON').glob('*.JPG'))
parser=argparse.ArgumentParser();parser.add_argument('--device',default='dml:1');parser.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8]);args=parser.parse_args()
model=Model(ROOT/'app/models/portable',args.device)
reference=[];records=[]
for batch in args.batches:
    results=[];tick=time.perf_counter()
    for offset in range(0,len(paths),batch):
        results.extend(model.predict(paths[offset:offset+batch]))
    record={'device':args.device,'batch':batch,'seconds':time.perf_counter()-tick,'images':len(results),'counts':[], 'minimum_mask_iou':1.,'maximum_area_difference_percent':0.,'maximum_score_difference':0.}
    if batch==1:reference=results
    for path,expected,result in zip(paths,reference,results):
        if result.path!=str(path) or not np.array_equal(expected.classes,result.classes):raise RuntimeError(f'Batch {batch}: identity/count changed on {path.name}')
        record['counts'].append(result.counts())
        record['maximum_score_difference']=max(record['maximum_score_difference'],float(np.max(np.abs(expected.scores-result.scores))))
        for before,after in zip(expected.polygons,result.polygons):
            points=np.concatenate([before,after]).astype(np.int32)
            x0,y0=np.maximum(points.min(0),0);x1,y1=np.minimum(points.max(0)+1,expected.image_shape[::-1])
            masks=[]
            for points in (before,after):
                mask=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillPoly(mask,[points.astype(np.int32)-[x0,y0]],1);masks.append(mask)
            union=np.count_nonzero(masks[0]|masks[1]);intersection=np.count_nonzero(masks[0]&masks[1])
            iou=intersection/union if union else 1.
            area=abs(np.count_nonzero(masks[0])-np.count_nonzero(masks[1]))/max(1,np.count_nonzero(masks[0]))*100
            record['minimum_mask_iou']=min(record['minimum_mask_iou'],iou)
            record['maximum_area_difference_percent']=max(record['maximum_area_difference_percent'],area)
        result.orig_img=None
    record['minimum_mask_iou']=float(record['minimum_mask_iou']);record['maximum_area_difference_percent']=float(record['maximum_area_difference_percent'])
    record['parity_pass']=bool(record['minimum_mask_iou']>=.999 and record['maximum_area_difference_percent']<=.1)
    records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='counts'}),flush=True)
    (ROOT/('validation/batch-inference-'+args.device.replace(':','-')+('-'+'-'.join(map(str,args.batches)) if args.batches!=[1,2,4,8] else '')+'.json')).write_text(json.dumps(records,indent=2))
model.close()
if not all(row['parity_pass'] for row in records):raise RuntimeError('Batch mask parity failed')
# Mixed aspect ratios, repeated source objects, and empty detections preserve order.
model=Model(ROOT/'app/models/portable','cpu')
images=[np.zeros((64,96,3),np.uint8),np.zeros((96,64,3),np.uint8),np.zeros((64,64,3),np.uint8)]
images.append(images[0])
results=model.predict(images,imgsz=640)
if [r.image_shape for r in results]!=[i.shape[:2] for i in images] or any(len(r.boxes) for r in results):raise RuntimeError('Mixed aspect or empty output regression')
model.close()
print('Batch inference, mixed aspect ratios and empty detections passed',flush=True)
