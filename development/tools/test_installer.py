"""Exercise installation, upgrades and rollback after a locked staging folder."""
from pathlib import Path
import ctypes as c
from ctypes import wintypes as w
import argparse,json,subprocess,time,zipfile,zlib

ROOT=Path(__file__).resolve().parents[2]
version=json.loads((ROOT/'app/hopbox/branding.json').read_text())['version']
parser=argparse.ArgumentParser();parser.add_argument('--area',type=Path);args=parser.parse_args()
area=args.area.resolve() if args.area else ROOT/'validation'/f'installer-regression-{version}';area.mkdir(exist_ok=True)
target=area/'HopBox-Infer'
if not target.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Installer tests must stay in the workspace')
exe=ROOT/'release'/f'HopBox-Infer-Setup-{version}.exe'
command=[str(exe),'--install-dir',str(target),'--no-launch','--quiet']
records=[]
for phase in ('fresh','upgrade'):
    tick=time.perf_counter();result=subprocess.run(command,creationflags=subprocess.CREATE_NO_WINDOW,timeout=180)
    if result.returncode:raise RuntimeError(f'{phase} failed: '+(area/'HopBox-Infer.installation-error.txt').read_text())
    # Verify every deployed file against the exact ZIP used by the installer.
    with zipfile.ZipFile(ROOT/'release'/f'HopBox-Infer-{version}-Windows.zip') as archive:
        for info in archive.infolist():
            path=area/info.filename
            checksum=0
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2),b''):checksum=zlib.crc32(block,checksum)
            if checksum!=info.CRC:raise RuntimeError('Installed file CRC mismatch: '+info.filename)
    records.append({'phase':phase,'exit':0,'every_file_crc_pass':True,'seconds':time.perf_counter()-tick})
    print('Installer',phase,'and every file CRC passed',flush=True)

# Force the same Directory.Move access-denied failure seen in the screenshot.
# A directory handle without FILE_SHARE_DELETE prevents the final activation.
sentinel=target/'rollback-sentinel.txt';sentinel.write_text('preserve this old installation')
previous=set(area.glob('.hopbox-install-*'))
process=subprocess.Popen(command,creationflags=subprocess.CREATE_NO_WINDOW)
kernel=c.WinDLL('kernel32',use_last_error=True)
kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
kernel.CreateFileW.restype=w.HANDLE
kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
deadline=time.monotonic()+30;stage=None;handle=None
try:
    while time.monotonic()<deadline:
        new=set(area.glob('.hopbox-install-*'))-previous
        if new:
            stage=next(iter(new)).resolve()
            if not stage.is_relative_to(area.resolve()):raise RuntimeError('Unexpected staging path')
            handle=kernel.CreateFileW(str(stage),1,3,None,3,0x02000000,None)
            if handle==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
            break
        if process.poll() is not None:raise RuntimeError('Installer ended before locking test')
        time.sleep(.01)
    else:raise TimeoutError('Staging folder did not appear')
    code=process.wait(timeout=180)
finally:
    if handle is not None:kernel.CloseHandle(handle)
if code!=1 or not sentinel.is_file() or sentinel.read_text()!='preserve this old installation':raise RuntimeError('Rollback did not preserve the old installation')
detail=(area/'HopBox-Infer.installation-error.txt').read_text()
if 'previous installation was restored' not in detail or 'activating' not in detail:raise RuntimeError('Rollback diagnostic was incomplete')
records.append({'phase':'locked_staging','exit':code,'old_installation_restored':True,'quiet_error':True})
(ROOT/'validation'/f'installer-regression-{version}.json').write_text(json.dumps({'checks':records,'pass':True},indent=2))
print('Access-denied activation reproduced; old installation restored; no modal error dialog',flush=True)
