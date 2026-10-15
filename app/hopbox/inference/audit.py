"""Inventory packages, direct DLL imports and modules loaded during a trial."""
from pathlib import Path
import importlib.metadata
import json
import sys

FORBIDDEN=('torch','torchvision','torchaudio','ultralytics','easyocr','libtorch','c10')

def runtime_audit(root=None):
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    bad=[n for n in packages if n.lower().replace('-','_') in FORBIDDEN]
    loaded=[n for n in sys.modules if n.split('.')[0] in FORBIDDEN]
    dlls=[];all_libraries=[]
    try:
        import psutil
        all_libraries=sorted({m.path for m in psutil.Process().memory_maps()})
        dlls=[path for path in all_libraries if any(t in path.lower() for t in FORBIDDEN)]
    except (ImportError,OSError):
        pass
    result={'packages':packages,'forbidden_packages':bad,'forbidden_modules':loaded,'forbidden_loaded_libraries':dlls,'loaded_libraries':all_libraries,
            'clean':not (bad or loaded or dlls)}
    if root:
        result['installed_bytes']=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    return result

def pe_imports(path):
    import pefile
    pe=pefile.PE(str(path),fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT']])
    imports=[e.dll.decode() for field in ('DIRECTORY_ENTRY_IMPORT','DIRECTORY_ENTRY_DELAY_IMPORT') for e in getattr(pe,field,[])]
    return {'file':str(path),'imports':imports,'clean':not any(any(t in d.lower() for t in FORBIDDEN) for d in imports)}

if __name__=='__main__':
    print(json.dumps(runtime_audit(),indent=2))
