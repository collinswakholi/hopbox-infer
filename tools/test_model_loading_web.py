"""Custom model loading, cancellation, and model-aware preview cache checks."""
from pathlib import Path
import argparse,json,shutil,time,urllib.request,urllib.error
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8767);parser.add_argument('--default-model',type=Path,default=ROOT/'release/HopBox-Infer/app/models/portable/hops_fp32.onnx');parser.add_argument('--label-image',type=Path,default=ROOT/'validation/potato-model/frame_000038.jpg');parser.add_argument('--checkpoint',type=Path,default=ROOT/'validation/potato-model/best_s_1024.pt');args=parser.parse_args()
for fixture in (args.label_image,args.checkpoint,ROOT/'validation/potato-model/best_s_1024_fp32.onnx'):
    if not fixture.is_file():parser.error('Missing local acceptance fixture: '+str(fixture)+'; supply --label-image / --checkpoint and the custom ONNX fixture. See docs/WORKSPACE.md')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path,body=None):
    call=urllib.request.Request(f'http://127.0.0.1:{args.port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with opener.open(call,timeout=60) as reply:return json.load(reply)
def job(body):
    token=request('/api/preview',body)['id'];deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        found=request('/api/jobs/'+token)
        if found['status']=='done':return found
        if found['status'] in ('error','stopped'):raise RuntimeError(found)
        time.sleep(.25)
    raise TimeoutError('Preview did not finish')
default=args.default_model
custom=ROOT/'validation/potato-model/best_s_1024_fp32.onnx'
request('/api/model',{'path':str(default),'device':'dml:1','imgsz':2048})
missing_converter=False
try:
    request('/api/model',{'path':str(args.checkpoint.resolve()),'imgsz':1024})
except urllib.error.HTTPError as error:
    payload=json.load(error)
    missing_converter=error.code==400 and 'converter' in payload['error']
if not missing_converter or request('/api/model')['name']!='hops_fp32.onnx':raise RuntimeError('Missing converter did not preserve the current model')
copy=ROOT/'validation/unmapped-potato/new_model_fp32.onnx';copy.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(custom,copy);shutil.copy2(custom.with_suffix('.json'),copy.with_suffix('.json'))
pending=request('/api/model',{'path':str(copy),'device':'dml:1','imgsz':1024})
if not pending.get('needs_map'):raise RuntimeError('New class mapping was not requested')
restored=request('/api/model/cancel',{})
if not restored.get('ready') or not restored.get('default'):raise RuntimeError('Class mapping cancel did not restore the prior model')
loaded=request('/api/model',{'path':str(custom),'device':'dml:1','imgsz':1024})
rows=[{'index':0,'source':'tuber','display':'Tuber','role':'measure'}]
request('/api/model/map',{'rows':rows,'device':'dml:1','imgsz':1024})
image=str(args.label_image.resolve())
request('/api/import',{'path':str(Path(image).parent)})
body={'image':image,'mode':'detections','imgsz':1024,'device':'dml:1','model':'ccm'}
preview=job(body)
if not request('/api/preview-cache',body)['hit']:raise RuntimeError('Preview cache did not save')
request('/api/model',{'path':str(default),'device':'dml:1','imgsz':1024})
if request('/api/preview-cache',body)['hit']:raise RuntimeError('Old model detection preview leaked into new model')
request('/api/model',{'path':str(custom),'device':'dml:1','imgsz':1024})
rows[0]['display']='Potato test'
request('/api/model/map',{'rows':rows,'device':'dml:1','imgsz':1024})
if request('/api/preview-cache',body)['hit']:raise RuntimeError('Old class labels leaked into new mapping')
rows[0]['display']='Tuber'
request('/api/model/map',{'rows':rows,'device':'dml:1','imgsz':1024})
request('/api/model',{'path':str(default),'device':'dml:1','imgsz':2048})
record={'missing_converter_preserves_model':True,'mapping_cancel_restores_model':True,'custom_preview_pass':True,'different_model_cache_miss':True,'different_class_mapping_cache_miss':True,'default_model_restored':True,'custom_preview_log':preview['log']}
(ROOT/f'validation/model-loading-web-{args.port}.json').write_text(json.dumps(record,indent=2))
print('Model loading, cancellation, cache isolation and restoration passed',flush=True)
