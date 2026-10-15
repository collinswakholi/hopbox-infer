"""Install and exercise the actual EXE with isolated PATH and private runtimes."""
from pathlib import Path
import argparse, csv, hashlib, json, os, subprocess, time, urllib.request, urllib.error
import psutil

ROOT=Path(__file__).resolve().parents[2]
BRANDING=json.loads((ROOT/'app/hopbox/branding.json').read_text())
VERSION=BRANDING['version']; PACKAGE=BRANDING['package']
parser=argparse.ArgumentParser()
parser.add_argument('--port',type=int,default=8771)
parser.add_argument('--device',default='dml:1')
parser.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8])
parser.add_argument('--gui',action='store_true')
parser.add_argument('--area',type=Path)
parser.add_argument('--package',type=Path,help='Test an existing staged package without reinstalling')
parser.add_argument('--label-image',type=Path,help='Locally supplied custom-model photo')
parser.add_argument('--checkpoint',type=Path,help='Locally supplied custom .pt checkpoint for missing-converter handling')
args=parser.parse_args()
area=args.area.resolve() if args.area else ROOT/'validation'/f'installed-{VERSION}'
target=area/PACKAGE;area.mkdir(exist_ok=True)
if args.package:target=args.package.resolve()
if not target.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Test installation must stay inside the workspace')
if not args.package:subprocess.run([str(ROOT/'release'/f'{PACKAGE}-Setup-{VERSION}.exe'),'--install-dir',str(target),'--no-launch','--quiet'],check=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
env=dict(os.environ)
for key in ('PYTHONPATH','HOPBOX_EXPORTER_PYTHON','HOPBOX_EXPORTER_SCRIPT'):env.pop(key,None)
env.update(PATH=str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32'),PYTHONNOUSERSITE='1',HOPBOX_SKIP_INSTANCE_STOP='1',HOPBOX_PORT=str(args.port),HOPBOX_STATE_ROOT=str(area/'state'),HOPBOX_STARTUP_TRACE=str(area/'startup.txt'))
if args.gui:env.pop('HOPBOX_NO_BROWSER',None)
else:env['HOPBOX_NO_BROWSER']='1'
launcher=subprocess.Popen([str(target/f'{PACKAGE}.exe')],env=env,creationflags=subprocess.CREATE_NO_WINDOW)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path,body=None):
    call=urllib.request.Request(f'http://127.0.0.1:{args.port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with opener.open(call,timeout=60) as reply:return json.load(reply)
def job(endpoint,body):
    tick=time.perf_counter();token=request(endpoint,body)['id'];deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        found=request('/api/jobs/'+token)
        if found['status']=='done':return found,time.perf_counter()-tick
        if found['status'] in ('error','stopped'):raise RuntimeError(found)
        time.sleep(.25)
    raise TimeoutError('Installed EXE processing timed out')
with (ROOT/'release'/f'{PACKAGE}-Setup-{VERSION}.exe').open('rb') as stream:
    installer_hash=hashlib.file_digest(stream,'sha256').hexdigest()
server=None;record={'version':VERSION,'application':BRANDING['name'],'installer_sha256':installer_hash,'pass':False,'batches':[]}
if args.package:record.update(installer_sha256=None,package=str(target),test_mode='existing staged EXE')
try:
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        try:
            ready=request('/api/ready')
            if ready.get('failed'):raise RuntimeError(ready)
            if ready['ok']:break
        except OSError:pass
        time.sleep(.5)
    else:raise RuntimeError('Installed EXE did not become ready')
    server=psutil.Process(ready['pid'])
    if not Path(server.exe()).resolve().is_relative_to(target.resolve()):raise RuntimeError('External interpreter was used')
    if request('/api/about')!=BRANDING:raise RuntimeError('App branding differs from build metadata')
    launcher.wait(timeout=10)
    trace=(area/'startup.txt').read_text()
    if not all(value in trace for value in ('splash-shown','models-ready','splash-closed')):raise RuntimeError('Splash did not complete')
    try:request('/api/jobs/unknown/stop',{})
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
    else:raise RuntimeError('Stopping an unknown job should return 404')
    folder=ROOT/'Data/119CANON';request('/api/import',{'path':str(folder)})
    reference=(ROOT/'validation/installed-1.3-batch-1/119CANON/Features.csv').read_bytes()
    for batch in args.batches:
        output=area/f'batch-{batch}'
        found,seconds=job('/api/run',{'folders':[str(folder)],'output':str(output),'batch_size':batch,'device':args.device,'imgsz':2048,'model':'ccm','qr_mode':'qr_ocr'})
        csvpath=output/'119CANON/Features.csv';blob=csvpath.read_bytes()
        with csvpath.open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        if len(rows)!=110:raise RuntimeError(f'Batch {batch}: missing traits')
        exact=blob==reference
        if args.device=='dml:1' and not exact:raise RuntimeError(f'Batch {batch}: scientific CSV changed from the NVIDIA reference')
        row={'batch':batch,'rows':len(rows),'seconds':seconds,'csv_exact_nvidia_reference':exact,'sha256':hashlib.sha256(blob).hexdigest()}
        record['batches'].append(row);print(json.dumps(row),flush=True)
        (area/'result.json').write_text(json.dumps(record,indent=2))
    # Real multi-folder processing includes independent photos.
    found,seconds=job('/api/run',{'folders':[{'path':str(folder),'files':[str(folder/'IMG_0002.JPG')]},str(ROOT/'Data/Canon_test')],'output':str(area/'multi-folder'),'batch_size':4,'device':args.device,'imgsz':2048,'model':'ccm','qr_mode':'qr_ocr'})
    record['multi_folder_seconds']=seconds
    for name,count in (('119CANON',10),('Canon_test',30)):
        with (area/'multi-folder'/name/'Features.csv').open(newline='') as stream:
            if len(list(csv.DictReader(stream)))!=count:raise RuntimeError('Multi-folder output count changed')
    model_check=[str(ROOT/'.build-env/Scripts/python.exe'),str(ROOT/'development/tools/test_model_loading_web.py'),'--port',str(args.port),'--default-model',str(target/'app/models/portable/hops_fp32.onnx')]
    if args.label_image:model_check+=['--label-image',str(args.label_image)]
    if args.checkpoint:model_check+=['--checkpoint',str(args.checkpoint)]
    subprocess.run(model_check,check=True,timeout=180)
    loaded=[module.path for module in server.memory_maps()]
    forbidden=('torch','ultralytics','libtorch','c10.dll','taichi','numba','llvmlite','skimage')
    if any(any(word in path.lower() for word in forbidden) or any(part.lower() in ('scipy','scipy.libs') for part in Path(path).parts) for path in loaded):raise RuntimeError('Forbidden runtime loaded')
    workers=server.children(recursive=True)
    request('/api/exit',{})
    code=server.wait(timeout=40)
    remaining=[child.pid for child in workers if child.is_running()]
    if code!=0 or remaining:raise RuntimeError('Installed EXE did not shut down cleanly')
    record.update(pass_=True,private_interpreter=str(target/'python/pythonw.exe'),splash_pass=True,loaded_dlls=loaded,model_loading_pass=True,multi_folder_pass=True,shutdown_exit=code,remaining_children=remaining)
    record['pass']=record.pop('pass_')
finally:
    (area/'result.json').write_text(json.dumps(record,indent=2))
    if server is not None and server.is_running():
        try:request('/api/exit',{});server.wait(timeout=40)
        except (OSError,psutil.Error):pass
print('Installed EXE folder batches, independent photos, custom models, startup and shutdown passed.',flush=True)
