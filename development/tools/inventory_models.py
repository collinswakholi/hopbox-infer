"""Audit portable graphs in the external exporter environment (requires ONNX)."""
from pathlib import Path
import hashlib, json
import onnx
ROOT=Path(__file__).resolve().parents[2]
records=[]
paths=list((ROOT/'app/models/portable').rglob('*.onnx'))+[ROOT/'validation/potato-model/best_s_1024_fp32.onnx']
for path in paths:
    model=onnx.load(path)
    onnx.checker.check_model(model)
    opsets={op.domain or 'ai.onnx':op.version for op in model.opset_import}
    nodes=sorted({node.op_type for node in model.graph.node})
    assert opsets['ai.onnx']<=20, path
    assert not {'GridSample','DeformConv'}.intersection(nodes), path
    assert not any(t.data_type==onnx.TensorProto.FLOAT16 for t in model.graph.initializer),path
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    records.append({'path':str(path.relative_to(ROOT)),'sha256':sha,'checker_pass':True,'opsets':opsets,
                    'operators':nodes,'fp16_initializers':False,'inputs':[v.name for v in model.graph.input]})
(ROOT/'validation/portable-graph-audit.json').write_text(json.dumps({'pass':True,'graphs':records,'scope':'Structural compatibility; not Radeon execution or driver certification.'},indent=2))
print(len(records),'portable FP32 graphs passed ONNX structural checks.')
