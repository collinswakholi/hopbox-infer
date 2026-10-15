"""Exercise AMD adapter selection without claiming to emulate Radeon hardware."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from hopbox import compute
from hopbox.inference.backends import OnnxBackend

calls = []
class Session:
    def __init__(self, model, sess_options, providers):
        calls.append({'providers': providers, 'sequential': str(sess_options.execution_mode),
                      'memory_pattern': sess_options.enable_mem_pattern, 'cpu_arena': sess_options.enable_cpu_mem_arena})
        self.providers = [p[0] if isinstance(p, tuple) else p for p in providers]
    def disable_fallback(self): pass
    def get_providers(self): return self.providers
    def get_inputs(self):
        from types import SimpleNamespace
        return [SimpleNamespace(name='images', type='tensor(float)')]

gpus = [compute.GpuInfo(0, 'Intel integrated', False, 0x8086),
        compute.GpuInfo(1, 'AMD Radeon test fixture', True, 0x1002)]
with patch.object(compute, '_GPUS', gpus), patch.object(compute, '_providers', return_value=['DmlExecutionProvider','CPUExecutionProvider']), patch.dict('os.environ', {'HOPBOX_DEVICE': ''}):
    assert compute.resolve_choice('auto').yolo == 'dml:1'
    assert compute.runtime_device('dml:1') == ('DmlExecutionProvider',1)
    assert any(row['id'] == 'dml:1' and 'AMD Radeon' in row['label'] for row in compute.device_catalog())
    with patch('onnxruntime.InferenceSession', Session):
        amd = OnnxBackend('fixture.onnx', 'DmlExecutionProvider', 1, shape=(1376,2048))
        intel = OnnxBackend('fixture.onnx', 'DmlExecutionProvider', 0)
    assert calls[0]['providers'] == [('DmlExecutionProvider', {'device_id':1})]
    assert calls[1]['providers'][0][1]['disable_metacommands'] == 'true'
    assert all(not row['memory_pattern'] and row['sequential'].endswith('ORT_SEQUENTIAL') for row in calls)
with patch.object(compute, '_GPUS', []), patch.object(compute, '_providers', return_value=['CPUExecutionProvider']), patch.dict('os.environ', {'HOPBOX_DEVICE': ''}):
    assert compute.resolve_choice('auto').yolo == 'cpu'
    with patch('onnxruntime.InferenceSession', Session):
        OnnxBackend('fixture.onnx', 'CPUExecutionProvider')
    assert calls[-1]['cpu_arena'] is False and calls[-1]['memory_pattern'] is False

record = {'pass':True, 'hardware_test':False,
          'checks':['AMD discrete GPU preferred over integrated Intel','correct DXGI adapter index passed to DirectML',
                    'AMD uses vendor-independent DirectML provider options','Intel workaround remains scoped to Intel',
                    'sequential execution and disabled memory patterns','CPU available without GPU'], 'session_options':calls}
(ROOT/'validation/radeon-adapter-checks.json').write_text(json.dumps(record,indent=2))
print('AMD adapter selection checks passed; no AMD hardware was emulated or tested.')
