"""Build the validated offline Windows CPU/DirectML app."""
from pathlib import Path
import argparse,hashlib,json,os,py_compile,shutil,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
BRANDING=json.loads((ROOT/'app/hopbox/branding.json').read_text())
VERSION=BRANDING['version']
csc=Path(r'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe')
identity=ROOT/'.build-tmp/Branding.cs';identity.parent.mkdir(exist_ok=True)
fields='\n'.join(f'public const string {k.title()} = {json.dumps(v)};' for k,v in BRANDING.items())
identity.write_text('using System.Reflection;\n'+f'[assembly: AssemblyVersion("{VERSION}.0")]\n[assembly: AssemblyFileVersion("{VERSION}.0")]\n[assembly: AssemblyTitle("{BRANDING["name"]}")]\n[assembly: AssemblyProduct("{BRANDING["name"]}")]\n[assembly: AssemblyCompany("{BRANDING["author"]}")]\n'+ 'public static class Branding {\n'+fields+'\n}\n')
logo=ROOT/'app/hopbox/static/branding/hopbox.png'
icon=ROOT/'app/hopbox/static/branding/hopbox.ico'
def compile_cs(source,target,extra=()):
    subprocess.run([str(csc),'/nologo','/codepage:65001','/target:winexe','/optimize+',f'/win32manifest:{ROOT / "installer/app.manifest"}','/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.Web.Extensions.dll',f'/win32icon:{icon}',f'/resource:{logo},HopBoxLogo',f'/out:{target}',*extra,str(identity),str(source)],check=True)
def compile_ui(core):
    compile_cs(ROOT/'installer/Launcher.cs',core/'HopBox-Infer.exe',[str(ROOT/'installer/StartupDiagnostics.cs')])
    subprocess.run([str(csc),'/nologo','/codepage:65001','/target:library','/optimize+','/r:System.Windows.Forms.dll','/r:System.Drawing.dll',f'/out:{core / "app/native/HopBoxNative.dll"}',str(ROOT/'app/hopbox/native_shell.cs')],check=True)
    manifest=json.loads((core/'application-manifest.json').read_text())
    manifest['windows_ui']={'compiler_codepage':'65001','launcher_source_sha256':hashlib.sha256((ROOT/'installer/Launcher.cs').read_bytes()).hexdigest(),'diagnostics_source_sha256':hashlib.sha256((ROOT/'installer/StartupDiagnostics.cs').read_bytes()).hexdigest(),'folder_picker_source_sha256':hashlib.sha256((ROOT/'app/hopbox/native_shell.cs').read_bytes()).hexdigest()}
    (core/'application-manifest.json').write_text(json.dumps(manifest,indent=2))

def sign_owned(paths):
    """Use an existing authorized certificate; never invent publisher trust."""
    thumbprint=os.environ.get('HOPBOX_SIGN_CERT_THUMBPRINT','').replace(' ','')
    if not thumbprint:return
    if len(thumbprint)!=40 or any(c not in '0123456789abcdefABCDEF' for c in thumbprint):raise ValueError('Invalid signing certificate thumbprint')
    signtool=os.environ.get('HOPBOX_SIGNTOOL') or shutil.which('signtool.exe')
    if not signtool:raise FileNotFoundError('Set HOPBOX_SIGNTOOL to the Windows SDK signtool.exe')
    timestamp=os.environ.get('HOPBOX_TIMESTAMP_URL','')
    if not timestamp.startswith('https://'):raise ValueError('Set an approved HTTPS HOPBOX_TIMESTAMP_URL for signed releases')
    for path in paths:
        subprocess.run([signtool,'sign','/sha1',thumbprint,'/fd','SHA256','/tr',timestamp,'/td','SHA256','/d',BRANDING['name'],str(path)],check=True)
        subprocess.run([signtool,'verify','/pa',str(path)],check=True)

def write_file_inventory(core):
    records=[]
    for path in sorted(core.rglob('*')):
        if not path.is_file() or path.name=='package-files.json':continue
        relative=path.relative_to(core).as_posix()
        records.append({'path':relative,'size':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'required':relative.startswith(('app/models/','app/native/')) or relative in ('app/hopbox-code.zip','python/python312.dll','python/python312.zip','python/pythonw.exe','python/python312._pth'),
            'fast_hash':relative in ('app/hopbox-code.zip','python/python312._pth')})
    (core/'package-files.json').write_text(json.dumps({'version':VERSION,'files':records,'purpose':'Detect missing, changed or quarantined files. Integrity hashes do not establish publisher trust.'},indent=2))

def validate_crt_seed():
    runtime=ROOT/'installer/vc-runtime'
    records=json.loads((runtime/'inventory.json').read_text())
    for item in records:
        path=runtime/item['name']
        if path.parent!=runtime or not path.is_file():raise FileNotFoundError('Missing inventoried Visual C++ runtime: '+item['name'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Visual C++ runtime checksum mismatch: '+item['name'])
    return runtime

def build_installer(output):
    if json.loads((output/'branding.json').read_text())!=BRANDING:raise ValueError('Existing payload identity differs; rebuild the complete release')
    setup_args=['/r:System.IO.Compression.dll','/r:System.IO.Compression.FileSystem.dll','/r:System.Management.dll']
    records=[]
    for name,resource in [('HopBox-Infer','HopBoxPayload')]:
        archive=output/f'{name}-{VERSION}-Windows.zip'
        with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        expected=next(p for p in json.loads((output/'inventory.json').read_text()) if p['name']==name)
        if digest!=expected['sha256']:raise ValueError('Payload checksum mismatch: '+name)
        setup_args.append(f'/resource:{archive},{resource}');records.append({'name':name,'sha256':digest})
    compile_cs(ROOT/'installer/Setup.cs',output/f'HopBox-Infer-Setup-{VERSION}.exe',setup_args)
    sign_owned([output/f'HopBox-Infer-Setup-{VERSION}.exe'])
    (output/'installer-build.json').write_text(json.dumps({'setup_source_sha256':hashlib.sha256((ROOT/'installer/Setup.cs').read_bytes()).hexdigest(),'payloads':records},indent=2))
    checksums=[]
    for file in sorted(output.iterdir()):
        if file.suffix.lower() in ('.exe','.zip'):
            with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            checksums.append(digest+'  '+file.name)
    (output/'SHA256SUMS.txt').write_text(chr(10).join(checksums)+chr(10))

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'release');parser.add_argument('--installer-only',action='store_true');parser.add_argument('--repack',action='store_true');args=parser.parse_args()
if sys.version_info[:2] != (3,12):raise RuntimeError('Build bytecode with Python 3.12, matching the private interpreter')
for source in [ROOT/'app/main.py',*(ROOT/'app/hopbox').rglob('*.py')]:
    compile(source.read_text(encoding='utf-8'),str(source),'exec')
for name in ('ccp-engine.exe','color_cpu.dll','postprocess.dll','science.dll'):
    if not (ROOT/'app/native'/name).is_file():raise FileNotFoundError('Build the Rust component first: '+name)
out=args.output.resolve()
if not out.is_relative_to(ROOT.resolve()) or out==ROOT.resolve():raise ValueError('Build output must be a dedicated folder inside the workspace')
out.mkdir(parents=True,exist_ok=True)
if args.installer_only:
    build_installer(out)
    print('Installer rebuilt from verified existing payloads',flush=True)
    raise SystemExit(0)
core=out/'HopBox-Infer'
if not args.repack:
    if core.exists():raise FileExistsError('Archive the prior release before rebuilding')
    core.mkdir()
    python=core/'python';python.mkdir()
    seed=ROOT/'installer/python-seed'
    for file in seed.iterdir():
        if file.is_file():shutil.copy2(file,python/file.name)
    # Include the C++ standard library needed by ONNX Runtime on clean Windows.
    runtime=validate_crt_seed()
    for name in ('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll'):
        if not (runtime/name).is_file():raise FileNotFoundError('Missing required Visual C++ runtime: '+name)
    for file in runtime.glob('*.dll'):shutil.copy2(file,python/file.name)
    (python/'python312._pth').write_text('python312.zip\n.\nLib/site-packages\n../app/hopbox-code.zip\nimport site\n')
    site=python/'Lib/site-packages';site.parent.mkdir(parents=True,exist_ok=True)
    def skip(directory,names):
        exclude=[]
        for name in names:
            unused=('pip','setuptools','wheel','scipy','scikit_image','skimage','imageio','lazy_loader','networkx','tifffile','pefile','ordlookup')
            if name=='__pycache__' or name in ('tests','test','pefile.py','peutils.py') or name.endswith('.whl') or any(name==p or name.startswith((p+'-',p+'.')) for p in unused):exclude.append(name)
        return exclude
    shutil.copytree(ROOT/'.build-env/Lib/site-packages',site,ignore=skip)
    for group in ('onnxruntime/tools','onnxruntime/transformers','onnxruntime/quantization'):
        target=site/group
        if target.exists():
            if not target.resolve().is_relative_to(site.resolve()):raise ValueError('Unsafe runtime prune path')
            shutil.rmtree(target)
    app=core/'app';app.mkdir()
    shutil.copy2(ROOT/'app/DEPLOYMENT.md',app/'DEPLOYMENT.md')
    # Keep editable source in the checkout. Runtime assets remain ordinary files;
    # zipimport loads only matching Python 3.12 bytecode from the application archive.
    compiled=[]
    (ROOT/'.build-tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bytecode-',dir=ROOT/'.build-tmp') as scratch:
        with zipfile.ZipFile(app/'hopbox-code.zip','w',zipfile.ZIP_DEFLATED) as library:
            sources=[(ROOT/'app/main.py','hopbox_entry.py'),(ROOT/'app/rthook_stdio.py','rthook_stdio.py'),(ROOT/'app/hopbox_bootstrap.py','hopbox_bootstrap.py')]
            sources += [(p,str(p.relative_to(ROOT/'app')).replace('\\','/')) for p in sorted((ROOT/'app/hopbox').rglob('*.py'))]
            for source,relative in sources:
                target=Path(scratch)/relative;target=target.with_suffix('.pyc');target.parent.mkdir(parents=True,exist_ok=True)
                py_compile.compile(str(source),cfile=str(target),dfile='<HopBox-Infer>/'+relative,doraise=True,invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
                library.write(target,relative[:-3]+'.pyc')
                compiled.append({'module':relative,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    shutil.copytree(ROOT/'app/hopbox/static',app/'hopbox/static',ignore=shutil.ignore_patterns('hopbox-source.png','__pycache__'))
    shutil.copy2(ROOT/'app/hopbox/branding.json',app/'hopbox/branding.json')
    (core/'application-manifest.json').write_text(json.dumps({'format':'python312-sourceless-zip','modules':compiled,'note':'Bytecode packaging removes editable source, not the private Python runtime. It is not source encryption.'},indent=2))
    shutil.copy2(ROOT/'BUILD.md',core/'BUILD.md')
    shutil.copytree(ROOT/'app/models/portable',app/'models/portable',ignore=shutil.ignore_patterns('__pycache__','rtx-cache'))
    shutil.copytree(ROOT/'app/resources',app/'resources')
    shutil.copytree(ROOT/'app/native',app/'native',ignore=shutil.ignore_patterns('*.pdb','*.lib','*.exp','color-aot'))
    for file in runtime.glob('*.dll'):shutil.copy2(file,app/'native'/file.name)
    shutil.copy2(runtime/'inventory.json',core/'vc-runtime-inventory.json')
    compile_ui(core)
    installation=ROOT/'installer/HOW_TO_INSTALL.txt'
    shutil.copy2(installation,out/'HOW_TO_INSTALL.txt')
    shutil.copy2(installation,core/'HOW_TO_INSTALL.txt')
    model=app/'models/portable/hops_fp32.json'
    metadata=json.loads(model.read_text());metadata.update(deployment_version=VERSION,inference_precision='fp32')
    model.write_text(json.dumps(metadata,indent=2))
else:
    manifest=json.loads((core/'application-manifest.json').read_text())
    for item in manifest['modules']:
        relative='main.py' if item['module']=='hopbox_entry.py' else item['module']
        source=ROOT/'app'/relative
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['source_sha256']:raise ValueError('Application source changed; use a full build: '+relative)
    shutil.copy2(ROOT/'installer/HOW_TO_INSTALL.txt',core/'HOW_TO_INSTALL.txt')
    shutil.copy2(ROOT/'installer/HOW_TO_INSTALL.txt',out/'HOW_TO_INSTALL.txt')
    shutil.copy2(ROOT/'BUILD.md',core/'BUILD.md')
    shutil.copy2(ROOT/'app/DEPLOYMENT.md',core/'app/DEPLOYMENT.md')
    compile_ui(core)
sign_owned([core/'HopBox-Infer.exe',*(core/'app/native'/name for name in ('ccp-engine.exe','color_cpu.dll','postprocess.dll','science.dll','HopBoxNative.dll'))])
write_file_inventory(core)
packs=[core]
inventory=[]
for pack in packs:
    archive=out/(pack.name+f'-{VERSION}-Windows.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(pack.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(out))
    with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    inventory.append({'name':pack.name,'installed_bytes':sum(p.stat().st_size for p in pack.rglob('*') if p.is_file()),
        'compressed_bytes':archive.stat().st_size,'sha256':digest})
    print(inventory[-1],flush=True)
(out/'inventory.json').write_text(json.dumps(inventory,indent=2))
(out/'branding.json').write_text(json.dumps(BRANDING,indent=2))
build_installer(out)
print('Offline release built',flush=True)
