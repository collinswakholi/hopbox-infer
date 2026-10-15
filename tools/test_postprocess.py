"""Differential NMS checks on saved network outputs and adversarial cases."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from hopbox.inference.postprocess import nms as native

def reference(boxes, scores, classes, threshold=0.75, limit=300):
    # Offsets reproduce the float32 arithmetic used by the reference's
    # class-aware NMS rather than using separate calls for each class.
    shifted = np.asarray(boxes, np.float32) + classes[:, None].astype(np.float32) * 7680.0
    order = np.argsort(-scores, kind="stable")[:30000]
    keep = []
    areas = np.maximum(shifted[:, 2] - shifted[:, 0], 0) * np.maximum(shifted[:, 3] - shifted[:, 1], 0)
    while len(order) and len(keep) < limit:
        i = int(order[0]); keep.append(i)
        others = order[1:]
        lt = np.maximum(shifted[i, :2], shifted[others, :2])
        rb = np.minimum(shifted[i, 2:], shifted[others, 2:])
        extent = np.maximum(rb - lt, 0)
        inter = extent[:, 0] * extent[:, 1]
        union = areas[i] + areas[others] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = others[overlap <= threshold]
    return np.asarray(keep, np.int64)


rng=np.random.default_rng(725)
cases=[]
for count in (0,1,30,1000,4000):
    xy=rng.uniform(-20,2048,(count,2)).astype(np.float32)
    boxes=np.concatenate([xy,xy+rng.uniform(0,500,(count,2)).astype(np.float32)],axis=1)
    scores=rng.integers(0,20,count).astype(np.float32)/20
    classes=rng.integers(0,4,count)
    cases.append((f'random-{count}',boxes,scores,classes))
cases.append(('overlap-ties',np.array([[0,0,10,10],[0,0,10,10],[0,0,10,10],[0,0,0,0]],np.float32),np.ones(4,np.float32),np.array([0,0,1,0])))
for path in sorted((ROOT/'.archive/migration-research-20261013/portable/artifacts/baseline/raw_rect').glob('*.npz')):
    with np.load(path) as data:
        rows=data['predictions'][0].T
        classes=rows[:,4:4+len(json.loads((ROOT/'app/models/portable/hops_fp32.json').read_text())['names'])].argmax(1)
        scores=rows[np.arange(len(rows)),4+classes]
        valid=scores>.58;rows=rows[valid];classes=classes[valid];scores=scores[valid]
        cases.append((path.stem,np.concatenate([rows[:,:2]-rows[:,2:4]/2,rows[:,:2]+rows[:,2:4]/2],axis=1),scores,classes))
records=[]
for name,boxes,scores,classes in cases:
    for threshold in (.0,.5,.75,1.):
        tick=time.perf_counter();expected=reference(boxes,scores,classes,threshold);before=time.perf_counter()-tick
        tick=time.perf_counter();actual=native(boxes,scores,classes,threshold);after=time.perf_counter()-tick
        if not np.array_equal(expected,actual):raise RuntimeError(f'NMS mismatch: {name} at {threshold}')
        records.append({'case':name,'threshold':threshold,'detections':len(actual),'numpy_seconds':before,'rust_seconds':after})
(ROOT/'validation/postprocess.json').write_text(json.dumps({'exact':True,'checks':len(records),'records':records},indent=2))
print(f'Rust NMS exactly matched NumPy on {len(records)} checks',flush=True)
