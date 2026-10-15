"""Verify and stage a local or authenticated GitHub build-input archive."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, stat, tempfile, urllib.request, urllib.error, zipfile

ROOT=Path(__file__).resolve().parents[2]

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlsplit
        if urlsplit(newurl).scheme!='https':raise ValueError('Refusing an insecure download redirect')
        result=super().redirect_request(req,fp,code,msg,headers,newurl)
        if urlsplit(req.full_url).netloc!=urlsplit(newurl).netloc:
            result.remove_header('Authorization')
        return result

def download(manifest, target):
    opener=urllib.request.build_opener(SafeRedirect())
    headers={'Accept':'application/vnd.github+json','User-Agent':'HopBox-build-inputs'}
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:headers['Authorization']='Bearer '+token
    api='https://api.github.com/repos/collinswakholi/hopbox-infer'
    release_path='/releases/'+str(int(manifest['release_id'])) if manifest.get('release_id') else '/releases/tags/'+manifest['release_tag']
    try:
        with opener.open(urllib.request.Request(api+release_path,headers=headers),timeout=60) as response:release=json.load(response)
    except urllib.error.HTTPError as error:
        if error.code==404:raise RuntimeError('Pinned build-input release or asset was not found. Check the release URL, or pass a checksum-matching local --archive PATH.') from None
        if error.code==403:raise RuntimeError('GitHub denied the download or rate-limited this request. Download the published archive in a browser and pass --archive PATH. CI permissions remain read-only.') from None
        raise
    if release['tag_name']!=manifest['release_tag']:raise RuntimeError('Release identity differs from pinned manifest')
    assets=[a for a in release['assets'] if a['name']==manifest['asset']]
    if len(assets)!=1:raise RuntimeError('Expected exactly one matching release asset')
    headers['Accept']='application/octet-stream'
    request=urllib.request.Request(api+'/releases/assets/'+str(int(assets[0]['id'])),headers=headers)
    target.parent.mkdir(parents=True,exist_ok=True)
    with opener.open(request,timeout=120) as response, target.open('wb') as handle:
        while chunk:=response.read(1024*1024):handle.write(chunk)

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--archive',type=Path);group.add_argument('--download',action='store_true');args=parser.parse_args()
    version=json.loads((ROOT/'app/hopbox/branding.json').read_text())['version'];manifest=json.loads((ROOT/'docs'/f'build-inputs-{version}.json').read_text())
    archive=args.archive.resolve() if args.archive else ROOT/'.build-tmp/assets'/manifest['asset']
    if args.download and not archive.is_file():download(manifest,archive)
    with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=manifest['sha256'] or archive.stat().st_size!=manifest['size']:raise ValueError('Build-input archive checksum/size mismatch')
    expected={r['path']:r for r in manifest['files']};verified=0
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.infolist())!=len(expected) or {i.filename for i in bundle.infolist()}!=set(expected):raise ValueError('Archive members differ from the pinned inventory')
        for info in bundle.infolist():
            relative=PurePosixPath(info.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive entry')
            if not info.filename.startswith(('app/models/portable/','installer/python-seed/','installer/vc-runtime/')):raise ValueError('Unexpected build input destination')
            destination=PurePosixPath('development')/relative if relative.parts[0]=='installer' else relative
            target=ROOT.joinpath(*destination.parts)
            if not target.resolve().is_relative_to(ROOT):raise ValueError('Build input escapes checkout')
            for parent in [target,*target.parents]:
                if parent==ROOT:break
                if parent.is_symlink() or parent.is_junction():raise ValueError('Refusing a reparse-point destination')
            record=expected[info.filename]
            if info.file_size!=record['size']:raise ValueError('Member size mismatch')
            if target.is_file():
                with target.open('rb') as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
                if existing!=record['sha256']:raise FileExistsError('Existing build input differs: '+info.filename)
                verified+=1;continue
            target.parent.mkdir(parents=True,exist_ok=True)
            fd,scratch=tempfile.mkstemp(prefix='.hopbox-input-',dir=target.parent)
            try:
                checksum=hashlib.sha256()
                with os.fdopen(fd,'wb') as handle,bundle.open(info) as stream:
                    while chunk:=stream.read(1024*1024):handle.write(chunk);checksum.update(chunk)
                if checksum.hexdigest()!=record['sha256']:raise ValueError('Member checksum mismatch')
                os.replace(scratch,target);verified+=1
            finally:
                if os.path.exists(scratch):os.unlink(scratch)
    print(f'Verified and staged {verified} model/runtime input files.',flush=True)

if __name__=='__main__':main()
