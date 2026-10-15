"""Differential scientific-kernel tests; Python references are build-only."""
from pathlib import Path
import importlib.util,json,sys,warnings,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'app'))
(ROOT/'validation/rust-ports').mkdir(parents=True,exist_ok=True)
from hopbox.inference.science import channel_features,lbp_image,connected_components,mask_roi,closest_contours
from skimage.feature import local_binary_pattern
from scipy.ndimage import label
spec=importlib.util.spec_from_file_location('reference_features',ROOT/'tests/references/features_numpy.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
spec=importlib.util.spec_from_file_location('reference_decode',ROOT/'tests/references/decode_numpy.py')
decode=importlib.util.module_from_spec(spec);decode.__package__='hopbox.inference';spec.loader.exec_module(decode)
rng=np.random.default_rng(13102026);records=[]
images=[np.zeros((1,1),np.uint8),np.ones((1,1),np.uint8),np.full((50,60),180,np.uint8),np.zeros((40,35),np.uint8),np.arange(256,dtype=np.uint8).reshape(16,16)]
images += [rng.integers(0,256,size=shape,dtype=np.uint8) for shape in ((2,3),(3,2),(7,11),(51,63),(240,310))]
images += [((rng.random((33,44))>.9)*255).astype(np.uint8)]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for index,image in enumerate(images):
        expected_lbp=local_binary_pattern(image,8,1)
        actual_lbp=lbp_image(image)
        if not np.array_equal(actual_lbp,expected_lbp):
            bad=np.argwhere(actual_lbp!=expected_lbp)[0];raise RuntimeError(f'LBP mismatch {index} at {bad}: {actual_lbp[tuple(bad)]} / {expected_lbp[tuple(bad)]}')
        expected=np.concatenate([reference.entropy_feats(image,'X').values[0],reference.lbp_feats(image,'X').values[0],reference.color_feats(image,'X').values[0],reference.glcm_feats(image,'X').values[0]])
        actual=channel_features(image)
        differences=np.abs(expected-actual)
        if not np.allclose(expected,actual,rtol=1e-11,atol=1e-10,equal_nan=True):raise RuntimeError(f'Feature mismatch {index}: {np.c_[expected,actual,differences]}')
        records.append({'case':index,'maximum_absolute_difference':float(np.nanmax(differences)),'lbp_exact':True})
for shape in ((1,1),(1,10),(10,1),(13,17),(150,180)):
    for probability in (0,.05,.5,1):
        image=(rng.random(shape)<probability).astype(np.uint8)
        n,labels,stats,centroids=connected_components(image)
        en,el,es,ec=cv2.connectedComponentsWithStats(image,connectivity=4)
        if n!=en or not np.array_equal(labels,el) or not np.array_equal(stats,es) or not np.allclose(centroids,ec,equal_nan=True):raise RuntimeError(f'Components mismatch {shape}/{probability}')
        sl,sn=label(image)
        if n!=sn+1 or not np.array_equal(sl,labels):raise RuntimeError('Character labeling mismatch')
for shape in ((4000,6000),(6000,4000),(107,253),(2,7)):
    logits=rng.standard_normal((19,31)).astype(np.float32)
    box=np.asarray([.17*shape[1],.14*shape[0],.58*shape[1],.64*shape[0]],np.float32)
    bounds=np.clip(np.ceil(box).astype(int),(0,0,0,0),(shape[1],shape[0],shape[1],shape[0]))
    expected=decode.reconstruct(logits,box,shape,True)
    content=decode.content_logits(logits,shape);actual=mask_roi(content,shape,bounds)
    x0,y0,x1,y1=bounds
    if not np.array_equal(actual,expected[y0:y1,x0:x1]):raise RuntimeError(f'Mask sampling mismatch {shape}')
for _ in range(20):
    a=rng.integers(0,6000,size=(47,2),dtype=np.int32);b=rng.integers(0,6000,size=(31,2),dtype=np.int32)
    expected=np.unravel_index(np.argmin(((a[:,None]-b[None])**2).sum(-1)),(len(a),len(b)))
    if closest_contours(a,b)!=expected:raise RuntimeError('Contour distance mismatch')
for bounds in ((4,2,3,8),(1,5,7,4),(2,2,2,4)):
    if mask_roi(np.ones((4,4),np.float32),(10,10),bounds).size:raise RuntimeError('Empty mask bounds regression')
(ROOT/'validation/rust-ports/kernels.json').write_text(json.dumps({'features':records,'components_checks':20,'mask_checks':4,'contour_checks':20,'pass':True},indent=2))
print('Rust features, LBP, components, mask sampling and contour distances passed',flush=True)
