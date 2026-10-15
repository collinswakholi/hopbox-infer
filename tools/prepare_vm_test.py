"""Prepare a fresh offline Windows Sandbox installation test."""
from pathlib import Path
import argparse,hashlib,json,shutil,xml.etree.ElementTree as xml
parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--area',type=Path);parser.add_argument('--package-dir',type=Path,help='Directory containing a downloaded installer and branding.json; defaults to release');parser.add_argument('--interactive',action='store_true');parser.add_argument('--label-image',type=Path,help='Locally supplied custom-model label photo; not distributed');parser.add_argument('--batches',type=int,nargs='+',choices=[1,2,4,8],default=[8,4,2,1]);args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
version=json.loads((ROOT/'app/hopbox/branding.json').read_text())['version']
package_dir=args.package_dir.resolve() if args.package_dir else ROOT/'release'
installer=package_dir/f'HopBox-Infer-Setup-{version}.exe'
if not installer.is_file() or not (package_dir/'branding.json').is_file():parser.error('Package directory must contain the installer and branding.json')
label_image=args.label_image or ROOT/'validation/potato-model/frame_000038.jpg'
if not label_image.is_file():parser.error('Supply --label-image PATH for the custom-model acceptance photo; see docs/WORKSPACE.md')
area=args.area.resolve() if args.area else ROOT/'validation'/f'vm-{version}'
if not area.is_relative_to(ROOT.resolve()):raise ValueError('VM records must stay in the workspace')
if args.gpu:area=area/'gpu'
control=area/'control';results=area/'results'
control.mkdir(parents=True,exist_ok=True);results.mkdir(exist_ok=True)
for name in ('guest.ps1','audit_guest.ps1','check_startup_options.ps1'):
    shutil.copy2(ROOT/'tools/vm'/name,control/name)
for name in ('best_s_1024_fp32.onnx','best_s_1024_fp32.json'):
    shutil.copy2(ROOT/'validation/potato-model'/name,control/name)
shutil.copy2(ROOT/'validation/installed-1.3-batch-1/119CANON/Features.csv',control/'reference.csv')
shutil.copy2(label_image,control/'frame_000038.jpg')
config=xml.Element('Configuration')
for name,value in (('vGPU','Enable' if args.gpu else 'Disable'),('Networking','Disable'),('AudioInput','Disable'),('VideoInput','Disable'),('PrinterRedirection','Disable'),('ClipboardRedirection','Enable'),('MemoryInMB','8192')):
    xml.SubElement(config,name).text=value
folders=xml.SubElement(config,'MappedFolders')
for host,guest,readonly in ((package_dir,r'C:\HopBox-QA\Package',True),(ROOT/'Data/119CANON',r'C:\HopBox-QA\119CANON',True),(control,r'C:\HopBox-QA\Control',True),(results,r'C:\HopBox-QA\Results',False)):
    folder=xml.SubElement(folders,'MappedFolder')
    xml.SubElement(folder,'HostFolder').text=str(host)
    xml.SubElement(folder,'SandboxFolder').text=guest
    xml.SubElement(folder,'ReadOnly').text=str(readonly).lower()
command=xml.SubElement(config,'LogonCommand')
xml.SubElement(command,'Command').text=r'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\HopBox-QA\Control\guest.ps1"'
if args.gpu:command.find('Command').text+=' -Mode GPU'
if args.interactive:command.find('Command').text+=' -Interactive'
command.find('Command').text+=' -Batches '+','.join(map(str,args.batches))
xml.indent(config)
xml.ElementTree(config).write(area/'HopBox-Offline.wsb',encoding='utf-8',xml_declaration=True)
manifest={'prepared':True,'executed':False,'networking':'disabled','vgpu':'enabled' if args.gpu else 'disabled','memory_mib':8192,'batches':args.batches,
          'clipboard':'enabled for entering the destination through UI automation',
          'installer_mode':'interactive' if args.interactive else 'quiet; explicit destination and components',
          'checks':['destination and component selection ('+('wizard' if args.interactive else 'command line')+')','custom Program Files destination with spaces','same-location quiet upgrade','read-only installation with separate writable data','no editable HopBox application source','installed HopBox-Infer.exe startup and browser UI','eleven-photo CPU Run Folders at batches '+','.join(map(str,args.batches)),'custom ONNX mapping/cancel/preview/restore','package and loaded-DLL audit','every CSV field reported','clean shutdown'],
          'result_file':str(results/'vm-result.json'),'input_folders_readonly':True,'writable_host_folder':str(results)}
if args.gpu:manifest['checks']=[s.replace('CPU Run Folders','DirectML GPU Run Folders') for s in manifest['checks']]+['explicit GPU inference; CPU fallback fails the test']
manifest['installer_path']=str(installer)
with installer.open('rb') as stream:manifest['installer_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
(area/'prepared.json').write_text(json.dumps(manifest,indent=2))
mode='DirectML GPU' if args.gpu else 'CPU'
vgpu='enabled; the VM shares a host GPU' if args.gpu else 'disabled'
(area/'README.txt').write_text(f'''HopBox fresh offline Windows Sandbox {mode} EXE test

Windows Sandbox must be enabled. Open HopBox-Offline.wsb.
Networking, audio/video and printer sharing are disabled. Clipboard redirection
is enabled for installer destination entry during optional UI automation.
vGPU is {vgpu}. Installer, input photos and control scripts are read-only.
Only results are shared back to the host with write access.

Installer mode: {manifest['installer_mode']}.
For --interactive, enter C:\\Program Files\\HopBox Infer QA, select core only
for either CPU or GPU, then Install and Finish. Do not open the app
from the wizard; the controller opens its actual EXE next.
Default tests use the explicit command-line destination/components unattended.
Both modes install, upgrade and run from the custom Program Files folder with
spaces, make the install read-only, process eleven photos at the planned batches,
load/map/preview/restore a custom ONNX model, audit dependencies and shut down.
No system Python or standalone Python processing script is used in the guest.

Progress and final results: results/vm-result.json. The guest is disposable.
A shared host vGPU cannot certify physical AMD Radeon hardware.
Preparation alone does not constitute a passed test.
''')
print('Offline VM test prepared:',area/'HopBox-Offline.wsb')
