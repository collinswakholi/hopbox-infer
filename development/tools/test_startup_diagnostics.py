"""Check damaged-package reporting and logging before application imports."""
from pathlib import Path
import argparse
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--package', type=Path, default=ROOT/'release/HopBox-Infer')
args = parser.parse_args()
package = args.package.resolve()
if not package.is_relative_to(ROOT):
    raise ValueError('Use a disposable package within the workspace.')
area = ROOT/'validation/security-startup-fix'
area.mkdir(exist_ok=True)
exe = package/'HopBox-Infer.exe'


def diagnose(name, expected):
    report = area/f'{name}.json'
    process = subprocess.run([str(exe), '--diagnose', str(report)], timeout=90,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    result = json.loads(report.read_text(encoding='utf-8-sig'))
    assert result['integrity_pass'] == expected, result
    assert process.returncode == (0 if expected else 1)
    return result


diagnose('intact-package', True)
archive = package/'app/hopbox-code.zip'
held = archive.with_suffix('.test-held')
if held.exists():
    raise FileExistsError('A previous test archive needs to be restored first.')
archive.rename(held)
try:
    missing = diagnose('missing-code-archive', False)
    assert 'Missing file: app/hopbox-code.zip' in missing['integrity_errors']
finally:
    held.rename(archive)
pth = package/'python/python312._pth'
original = pth.read_bytes()
try:
    pth.write_bytes(original+b'\n# damaged configuration test\n')
    changed = diagnose('changed-python-path', False)
    assert 'Changed file size: python/python312._pth' in changed['integrity_errors']
finally:
    pth.write_bytes(original)
env = dict(os.environ, HOPBOX_STARTUP_LOG=str(area/'bootstrap-error.log'))
code = ('import runpy,hopbox_bootstrap; '
        'runpy.run_module=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("bootstrap-error-probe")); '
        'hopbox_bootstrap.main()')
result = subprocess.run([str(package/'python/pythonw.exe'), '-B', '-c', code],
                        env=env, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
assert result.returncode != 0
log = (area/'bootstrap-error.log').read_text(encoding='utf-8')
assert 'Traceback' in log and 'bootstrap-error-probe' in log
diagnose('restored-package', True)
(area/'diagnostics-tests.json').write_text(json.dumps({
    'pass': True, 'missing_code_detected': True, 'changed_path_detected': True,
    'pre_import_exception_logged': True, 'original_files_restored': True,
}, indent=2))
print('Missing/changed files reported; pre-import traceback captured; all files restored.')
