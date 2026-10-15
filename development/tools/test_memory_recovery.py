"""Exercise batch recovery and release of arrays retained by failure tracebacks."""
from pathlib import Path
import json
import sys
import weakref
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'app'))
from hopbox.pipeline import Runner, RunOptions

records = []
paths = [Path(f'image-{index}.jpg') for index in range(8)]
for error_factory in (lambda: cv2.error('(-4:Insufficient memory) Failed to allocate 53747712 bytes'),
                      lambda: RuntimeError('ORT allocator: failed to allocate memory')):
    allocations = []
    calls = []
    closed = []

    class RecoveringModel:
        def predict(self, source, **unused):
            assert all(ref() is None for ref in allocations), 'Failed tensors survived into retry'
            calls.append(len(source))
            if len(source) > 2:
                buffer = np.empty((1024, 1024), np.uint8)
                allocations.append(weakref.ref(buffer))
                raise error_factory()
            yield from source

        def close(self):
            closed.append(True)

    runner = Runner.__new__(Runner)
    logs = []
    result = runner._predict_chunk(RecoveringModel(), paths, 'cpu', RunOptions(ROOT / 'validation'), logs.append)
    assert result == [str(path) for path in paths]
    assert calls == [8, 4, 2, 2, 4, 2, 2]
    assert len(closed) == 7 and all(ref() is None for ref in allocations)
    records.append({'error': str(error_factory()), 'attempt_sizes': calls,
                    'failed_arrays_released': True, 'successful_cpu_batch_arenas_released': True,
                    'order_preserved': True, 'logs': logs})

class InvalidModel:
    def predict(self, **unused):
        raise ValueError('Invalid model output')
    def close(self):
        raise AssertionError('Non-memory errors must propagate without retry')

try:
    runner._predict_chunk(InvalidModel(), paths, 'cpu', RunOptions(ROOT / 'validation'), logs.append)
except ValueError:
    pass
else:
    raise AssertionError('Non-memory failure was swallowed')
(ROOT / 'validation/memory-recovery.json').write_text(json.dumps({'pass': True, 'checks': records,
    'non_memory_errors_propagate': True}, indent=2))
print('Memory recovery passed: OpenCV/ORT allocation errors, traceback release, recursive splitting and source order')
