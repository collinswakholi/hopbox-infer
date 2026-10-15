"""Exercise the production launcher after a simulated native GPU process crash.

Only a disposable package receives the fault-injection bootstrap. Production
contains no injection flag or alternate startup command.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
import psutil

ROOT=Path(__file__).resolve().parents[1]
package=ROOT/'release/HopBox-Infer'
fixture=ROOT/'.build-tmp/native-recovery-fixture'
area=ROOT/'validation/security-native-recovery'
if not fixture.resolve().is_relative_to(ROOT/'.build-tmp') or fixture.exists():
    raise ValueError('Use a new dedicated fault-injection fixture.')
area.mkdir(exist_ok=True)
shutil.copytree(package,fixture)
archive=fixture/'app/hopbox-code.zip'
with zipfile.ZipFile(archive) as z:
    members={name:z.read(name) for name in z.namelist()}
members['hopbox_bootstrap_original.pyc']=members['hopbox_bootstrap.pyc']
source='''import os, ctypes, runpy
if os.environ.get("HOPBOX_DEVICE") != "cpu":
    stop=ctypes.WinDLL("kernel32",use_last_error=True).ExitProcess
    stop.argtypes=[ctypes.c_uint]
    stop.restype=None
    stop(0xc0000005)
runpy.run_module("hopbox_bootstrap_original",run_name="__main__")
'''
with tempfile.TemporaryDirectory(dir=ROOT/'.build-tmp') as scratch:
    path=Path(scratch)/'hopbox_bootstrap.py';path.write_text(source)
    compiled=path.with_suffix('.pyc')
    py_compile.compile(str(path),cfile=str(compiled),dfile='<test-only native failure>',doraise=True)
    members['hopbox_bootstrap.pyc']=compiled.read_bytes()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,content in members.items():z.writestr(name,content)
inventory=json.loads((fixture/'package-files.json').read_text())
for item in inventory['files']:
    if item['path']=='app/hopbox-code.zip':
        item.update(size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(fixture/'package-files.json').write_text(json.dumps(inventory,indent=2))
input_folder=area/'input';input_folder.mkdir(exist_ok=True)
shutil.copy2(ROOT/'Data/119CANON/IMG_0002.JPG',input_folder/'IMG_0002.JPG')
env=dict(os.environ,HOPBOX_PORT='8788',HOPBOX_DEVICE='dml:1',HOPBOX_NO_BROWSER='1',
         HOPBOX_STATE_ROOT=str(area/'state'),HOPBOX_STARTUP_TRACE=str(area/'startup.txt'))
(area/'startup.txt').unlink(missing_ok=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:8788'+path,
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=30) as response:return json.load(response)

record={'pass':False,'launcher_sha256':hashlib.sha256((package/'HopBox-Infer.exe').read_bytes()).hexdigest(),
        'injection':'Disposable bootstrap exits with 0xC0000005 before imports unless CPU is selected.',
        'production_package_modified':False}
launcher=subprocess.Popen([str(fixture/'HopBox-Infer.exe')],env=env,creationflags=subprocess.CREATE_NO_WINDOW)
server=None
try:
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        try:
            ready=api('/api/ready')
            if ready.get('ok'):break
        except OSError:pass
        time.sleep(.25)
    else:raise TimeoutError('CPU recovery did not become ready')
    assert ready['detail'].startswith('CPU'),ready
    server=psutil.Process(ready['pid'])
    assert Path(server.exe()).resolve().is_relative_to(fixture)
    launcher.wait(timeout=15);assert launcher.returncode==0
    trace=(area/'startup.txt').read_text()
    assert trace.count('cpu-startup-recovery')==1 and '0xC0000005' in trace and 'models-ready' in trace,trace
    api('/api/import',{'path':str(input_folder)})
    token=api('/api/run',{'folders':[str(input_folder)],'output':str(area/'output'),
        'device':'auto','imgsz':2048,'model':'ccm','qr_mode':'qr_ocr','batch_size':1})['id']
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        result=api('/api/jobs/'+token)
        if result['status']=='done':break
        assert result['status'] not in ('error','stopped'),result
        time.sleep(.5)
    else:raise TimeoutError('Recovered CPU folder run timed out')
    rows=list(csv.DictReader((area/'output/input/Features.csv').open(encoding='utf-8-sig',newline='')))
    reference=list(csv.DictReader((ROOT/'validation/installed-1.3-batch-1/119CANON/Features.csv').open(encoding='utf-8-sig',newline='')))
    identity=lambda r:(r['Image_name'],r['ID'],r['QR_info'],r['CC_model'])
    expected=[identity(r) for r in reference if r['Image_name']=='IMG_0002.JPG']
    assert [identity(r) for r in rows]==expected and len(rows)==10
    api('/api/exit',{})
    exit_code=server.wait(timeout=30);assert exit_code==0,exit_code
    record.update(pass_=True,ready=ready,folder_rows=len(rows),identities_match=True,
                  shutdown_exit=exit_code,single_cpu_retry=True,trace=trace.splitlines())
    record['pass']=record.pop('pass_')
finally:
    try:api('/api/exit',{})
    except OSError:pass
    if server is not None:
        try:server.wait(timeout=15)
        except psutil.TimeoutExpired:server.kill();server.wait(timeout=10)
    if launcher.poll() is None:launcher.terminate();launcher.wait(timeout=10)
    (area/'result.json').write_text(json.dumps(record,indent=2))
print('Native GPU-exit recovery, CPU folder processing and clean shutdown passed.')
