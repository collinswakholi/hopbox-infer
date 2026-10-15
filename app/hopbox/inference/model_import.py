"""Optional .pt conversion in an explicitly configured external environment."""
import hashlib,os,subprocess,tempfile,threading
from pathlib import Path
from hopbox.paths import state_root

_LOCK=threading.Lock()
_PROCESS=None

def cancel_export():
    with _LOCK:
        if _PROCESS is not None and _PROCESS.poll() is None:_PROCESS.terminate()

def convert_checkpoint(path,imgsz):
    global _PROCESS
    python=Path(os.environ.get('HOPBOX_EXPORTER_PYTHON',''))
    script=Path(os.environ.get('HOPBOX_EXPORTER_SCRIPT',''))
    if not python.is_file() or not script.is_file():
        raise ValueError('The external .pt converter is not configured. Choose the exported ONNX bundle, or follow the converter setup in DEPLOYMENT.md.')
    path=Path(path).resolve()
    key=hashlib.sha256(path.read_bytes()+script.read_bytes()+str(imgsz).encode()).hexdigest()[:24]
    parent=state_root()/'models/imported';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/key
    graph=destination/(path.stem+'_fp32.onnx')
    if graph.is_file() and graph.with_suffix('.json').is_file():return graph
    with tempfile.TemporaryDirectory(prefix='export-',dir=parent) as scratch:
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
        process=subprocess.Popen([str(python.resolve()),str(script.resolve()),str(path),scratch,'--imgsz',str(imgsz)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        with _LOCK:_PROCESS=process
        try:
            stdout,stderr=process.communicate(timeout=600)
            if process.returncode:raise RuntimeError('Model conversion failed: '+stderr.decode('utf-8',errors='replace')[-1500:])
            artifact=Path(scratch)/(path.stem+'_fp32.onnx')
            if not artifact.is_file() or not artifact.with_suffix('.json').is_file():raise RuntimeError('Converter did not produce a complete ONNX bundle')
            Path(scratch).rename(destination)
        except subprocess.TimeoutExpired:
            process.kill();process.communicate()
            raise RuntimeError('Model conversion timed out') from None
        finally:
            with _LOCK:_PROCESS=None
    return graph
