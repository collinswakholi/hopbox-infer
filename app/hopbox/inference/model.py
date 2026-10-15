"""Portable model bundle loading, replacing runtime .pt conversion."""
from pathlib import Path
import hashlib
import json

def load_bundle(source,precision='fp32'):
    source=Path(source).resolve()
    graph=source/f'hops_{precision}.onnx' if source.is_dir() else source
    if graph.suffix.lower()!='.onnx':raise ValueError('Choose an ONNX model bundle')
    metadata=json.loads(graph.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata.get('task')!='segment' or metadata.get('normalization')!='RGB/255':
        raise ValueError('Unsupported model bundle task or preprocessing contract')
    with graph.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checksum!=metadata['sha256']:
        raise ValueError('Model graph does not match bundle checksum')
    names={int(k):str(v) for k,v in metadata['names'].items()}
    if sorted(names)!=list(range(len(names))):raise ValueError('Class indices must be contiguous from zero')
    return graph,names,metadata
