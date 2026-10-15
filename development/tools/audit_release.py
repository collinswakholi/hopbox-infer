"""Audit installed packages, application imports, model files and PE imports."""
from pathlib import Path
import ast,dis,json,marshal,zipfile
import pefile
ROOT=Path(__file__).resolve().parents[2]
forbidden=('torch','torchvision','torchaudio','ultralytics','easyocr','libtorch','c10','scipy','skimage','scikit_image','taichi','numba','llvmlite')
records=[]
application=ROOT/'release/HopBox-Infer/app'
if any(p.suffix in ('.py','.cs','.ps1') for p in application.rglob('*') if p.is_file()):raise RuntimeError('Editable application/build source was shipped')
def imports(code):
    found=[i.argval for i in dis.get_instructions(code) if i.opname=='IMPORT_NAME']
    for value in code.co_consts:
        if isinstance(value,type(code)):found+=imports(value)
    return found
with zipfile.ZipFile(application/'hopbox-code.zip') as library:
    if library.testzip():raise RuntimeError('Application bytecode archive is corrupt')
    for name in library.namelist():
        if not name.endswith('.pyc'):raise RuntimeError('Unexpected application archive member')
        if any(m.split('.')[0] in forbidden for m in imports(marshal.loads(library.read(name)[16:]))):raise RuntimeError('Forbidden compiled application import: '+name)
for folder in (ROOT/'release/HopBox-Infer',):
    bad=[];native=[];distributions=[]
    for name in ('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll'):
        if not (folder/'python'/name).is_file():bad.append('Missing bundled Visual C++ runtime: '+name)
    for path in folder.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(folder)
        if path.suffix.lower() in ('.pt','.torchscript') or path.suffix.lower()=='.pth' and 'models' in relative.parts:bad.append(str(relative))
        if 'site-packages' in relative.parts:
            top=relative.parts[relative.parts.index('site-packages')+1].lower()
            if any(top==p or top.startswith((p+'-',p+'.')) for p in forbidden):bad.append(str(relative))
        if path.name=='METADATA':
            fields={}
            for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
                if line.startswith(('Name: ','Version: ')):
                    key,value=line.split(': ',1);fields[key]=value
            distributions.append(fields)
        if path.suffix=='.py' and relative.parts[0]=='app':
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                if any(m.split('.')[0] in forbidden for m in modules):bad.append(str(relative))
        if path.suffix.lower() in ('.dll','.pyd','.exe'):
            pe=pefile.PE(str(path),fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT']])
            dependencies=[entry.dll.decode() for name in ('DIRECTORY_ENTRY_IMPORT','DIRECTORY_ENTRY_DELAY_IMPORT') for entry in getattr(pe,name,[])]
            native.append({'file':str(relative),'dependencies':dependencies})
            # NumPy's required OpenBLAS DLL contains "libscipy" in its name;
            # this is not the SciPy package or an import of its extension modules.
            if any(any(word in dependency.lower() for word in forbidden if word!='scipy') for dependency in dependencies):bad.append(str(relative))
            if 'taichi' in path.name.lower() or path.name.lower() in ('tch.dll','libtch.dll'):bad.append(str(relative))
    record={'package':folder.name,'pass':not bad,'forbidden':sorted(set(bad)),'distributions':distributions,'native_dependencies':native}
    records.append(record)
    print(folder.name,record['pass'],len(native),'native binaries',flush=True)
archives=[]
for path in (ROOT/'release').glob('*.zip'):
    with zipfile.ZipFile(path) as archive:
        failed=archive.testzip()
        if failed:raise RuntimeError(f'ZIP CRC failed: {failed}')
        archives.append({'archive':path.name,'crc_pass':True})
(ROOT/'validation/release-audit.json').write_text(json.dumps({'packages':records,'archives':archives},indent=2))
if not all(r['pass'] for r in records):raise RuntimeError('Release contains a forbidden dependency')
