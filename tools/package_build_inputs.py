"""Package the exact model/runtime seeds required by a source checkout."""
from pathlib import Path
import argparse, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = json.loads((ROOT/'app/hopbox/branding.json').read_text())['version']

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ROOT/'release'); parser.add_argument('--release-id',type=int,help='Draft/release API identity, allowing authenticated downloads before a tag is published'); args=parser.parse_args()
    output=args.output.resolve()
    if output==ROOT or not output.is_relative_to(ROOT): raise ValueError('Use a dedicated output folder inside this checkout')
    output.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'app/models/portable').rglob('*'))+list((ROOT/'installer/python-seed').iterdir())+list((ROOT/'installer/vc-runtime').glob('*.dll'))
    paths=sorted(p for p in paths if p.is_file() and '__pycache__' not in p.parts)
    if any(p.suffix.lower() in ('.pt','.pth','.torchscript') for p in paths): raise ValueError('Only deployment model artifacts belong in build inputs')
    archive=output/f'HopBox-Infer-{VERSION}-Build-Inputs.zip'; records=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT): raise ValueError('Input must be a regular workspace file')
            relative=path.relative_to(ROOT).as_posix(); blob=path.read_bytes()
            member=zipfile.ZipInfo(relative,date_time=(1980,1,1,0,0,0)); member.compress_type=zipfile.ZIP_DEFLATED
            bundle.writestr(member,blob)
            records.append({'path':relative,'size':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    manifest={'version':VERSION,'release_tag':'v'+VERSION,'repository':'collinswakholi/hopbox-infer','asset':archive.name,'size':archive.stat().st_size,'sha256':digest,'purpose':'Exact tested ONNX/QR models and app-local CPython/Microsoft runtime seeds; no training checkpoints or private data.','files':records}
    if args.release_id:manifest['release_id']=args.release_id
    (ROOT/'docs'/f'build-inputs-{VERSION}.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:manifest[key] for key in ('asset','size','sha256')}),flush=True)

if __name__=='__main__':main()
