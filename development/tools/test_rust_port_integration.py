"""Validate the complete ports against the preserved 1.2.0 implementations."""
from pathlib import Path
import importlib.util,json,sys,time,warnings
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'app'))
def reference(name,path,package=None):
    spec=importlib.util.spec_from_file_location(name,ROOT/'.archive/rust-port-reference-1.2.0/hopbox'/path)
    module=importlib.util.module_from_spec(spec)
    if package:module.__package__=package
    spec.loader.exec_module(module);return module
old_decode=reference('old_decode','inference/decode.py','hopbox.inference')
old_features=reference('old_features','features.py')
old_craft=reference('old_craft','inference/ocr_geometry/craft_utils.py')
old_aot=reference('old_aot','inference/aot_color.py','hopbox.inference')
from hopbox.inference.decode import decode
from hopbox.features import process_image
from hopbox.inference.ocr_geometry import craft_utils
from hopbox.inference.rust_color import apply_rust
names={int(k):v for k,v in json.loads((ROOT/'app/models/portable/hops_fp32.json').read_text())['names'].items()}
raws=ROOT/'.archive/migration-research-20261013/portable/artifacts/baseline/raw_rect'
if not list(raws.glob('*.npz')):
    raise FileNotFoundError('Real-image migration fixtures are required for this local acceptance test; see docs/WORKSPACE.md. CI uses test_rust_science.py instead.')
records=[];colors=[];texture_before=0.;texture_after=0.
for path in sorted(raws.glob('*.npz')):
    image=cv2.imread(str(ROOT/f'Data/119CANON/{path.stem}.JPG'))
    with np.load(path) as raw:
        expected=old_decode.decode(raw['predictions'],raw['prototypes'],image.shape,(1376,2048),names,roi=True,retain_masks=False)
        actual=decode(raw['predictions'],raw['prototypes'],image.shape,(1376,2048),names,roi=True,retain_masks=False)
    if not np.array_equal(expected.classes,actual.classes) or len(expected.polygons)!=len(actual.polygons) or not all(np.array_equal(a,b) for a,b in zip(expected.polygons,actual.polygons)):raise RuntimeError('Mask/contour changed: '+path.stem)
    record={'image':path.stem,'masks_and_contours_exact':True,'traits':{}}
    expected.orig_img=image
    # Compare identical real cropped pixels; independent from the color backend.
    crops,polys=old_features.extract_roi(expected,np.array([k for k,v in names.items() if v=='Hops']))
    for key,crop in crops.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tick=time.perf_counter();before=old_features.process_image(crop,polys[key]);texture_before+=time.perf_counter()-tick
            tick=time.perf_counter();after=process_image(crop,polys[key]);texture_after+=time.perf_counter()-tick
        if list(before.columns)!=list(after.columns):raise RuntimeError('Trait columns changed')
        for column in before.columns:
            a=float(before[column].iloc[0]);b=float(after[column].iloc[0]);difference=abs(a-b)
            if not np.isclose(a,b,rtol=1e-11,atol=1e-10,equal_nan=True):raise RuntimeError(f'{path.stem}/{key}/{column}: {a}/{b}')
            if np.isfinite(difference):record['traits'][column]=max(record['traits'].get(column,0.),difference)
    records.append(record);print(path.stem,'masks exact; every feature passed',flush=True)
    # All fitted methods: replacement color kernel must respect the original gate.
    for model in ('ccm','rpcc2','rpcc3'):
        saved=json.loads((ROOT/f'.archive/migration-research-20261013/portable/artifacts/color/{path.stem}/{model}-reference.json').read_text());saved.update(do_gc=True,do_wb=True,do_cc=True)
        cpu=apply_rust(image,saved);gpu=old_aot.apply_aot(image,saved,ROOT/'.archive/rust-port-reference-1.2.0/native/color-aot',1)
        difference=np.abs(cpu.astype(np.int16)-gpu.astype(np.int16))
        item={'image':path.stem,'method':model,'max_pixel_difference':int(difference.max()),'changed_values':int(np.count_nonzero(difference))}
        if item['max_pixel_difference']>1:raise RuntimeError('Color helper pixel gate failed')
        colors.append(item)
old_aot.close()
# Test CRAFT grouping on synthetic sparse and dense maps including estimated characters.
rng=np.random.default_rng(864)
for _ in range(12):
    text=cv2.GaussianBlur(rng.random((80,120)).astype(np.float32),(5,5),0)
    links=rng.random((80,120)).astype(np.float32)*.5
    for estimate in (False,True):
        a=old_craft.getDetBoxes_core(text,links,.5,.4,.4,estimate);b=craft_utils.getDetBoxes_core(text,links,.5,.4,.4,estimate)
        if len(a[0])!=len(b[0]) or not all(np.array_equal(x,y) for x,y in zip(a[0],b[0])) or a[2]!=b[2]:raise RuntimeError('CRAFT boxes/order changed')
result={'images':records,'color':colors,'craft_checks':24,'reference_feature_seconds':texture_before,'rust_feature_seconds':texture_after,'pass':True}
(ROOT/'validation/rust-ports/integration.json').write_text(json.dumps(result,indent=2))
print('All Rust numerical ports passed real-image integration',flush=True)
