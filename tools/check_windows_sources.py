"""Compile native UI sources without private model/runtime payloads."""
from pathlib import Path
import json, os, subprocess

ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'.build-tmp/ci-compile';output.mkdir(parents=True,exist_ok=True)
if not output.resolve().is_relative_to(ROOT):raise ValueError('Invalid compiler output')
branding=json.loads((ROOT/'app/hopbox/branding.json').read_text())
identity=output/'Branding.cs'
identity.write_text('public static class Branding {\n'+'\n'.join('public const string '+k.title()+' = '+json.dumps(v)+';' for k,v in branding.items())+'\n}\n',encoding='utf8')
csc=Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
base=[str(csc),'/nologo','/codepage:65001','/target:winexe','/optimize+',f'/win32manifest:{ROOT/"installer/app.manifest"}','/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.Web.Extensions.dll',f'/resource:{ROOT/"app/hopbox/static/branding/hopbox.png"},HopBoxLogo']
subprocess.run([*base,f'/out:{output/"SourceOnly-Launcher.exe"}',str(identity),str(ROOT/'installer/Launcher.cs'),str(ROOT/'installer/StartupDiagnostics.cs')],check=True)
subprocess.run([*base,'/r:System.IO.Compression.dll','/r:System.IO.Compression.FileSystem.dll','/r:System.Management.dll',f'/out:{output/"SourceOnly-Setup.exe"}',str(identity),str(ROOT/'installer/Setup.cs')],check=True)
subprocess.run([str(csc),'/nologo','/codepage:65001','/target:library','/optimize+','/r:System.Windows.Forms.dll','/r:System.Drawing.dll',f'/out:{output/"SourceOnly-Picker.dll"}',str(ROOT/'app/hopbox/native_shell.cs')],check=True)
(output/'README.txt').write_text('Compiler checks only. These are not distribution files; setup has no embedded payload.\n')
print('C# launcher, diagnostics, setup and native picker compiled successfully.',flush=True)
