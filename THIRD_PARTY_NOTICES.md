# Third-party components and model artifacts

Original HopBox application code: Copyright (C) 2026 Collins Wakholi,
GNU AGPL-3.0-only; see [LICENSE](LICENSE). This does not replace upstream licenses.

| Component | Origin / terms | Use in HopBox |
| --- | --- | --- |
| Hop and chart YOLO models | Ultralytics-derived trained graphs; AGPL-3.0 terms apply absent a separately applicable license. [Upstream licensing](https://www.ultralytics.com/license) | Same FP32 ONNX artifacts used in validation. ONNX conversion does not remove upstream terms. |
| OCR geometry and recognition artifacts | [EasyOCR](https://github.com/JaidedAI/EasyOCR), Apache-2.0; [CRAFT](https://github.com/clovaai/CRAFT-pytorch), MIT, Copyright (c) 2019-present NAVER Corp. | CRAFT and English recognition exported to ONNX. Geometry/grouping adapted; connected components replaced with Rust. License texts are in `licenses` and beside the vendored geometry source. |
| WeChat QR detector and super-resolution models | [OpenCV models](https://github.com/opencv/opencv_3rdparty/tree/wechat_qrcode_20210119), [Tencent notice](https://github.com/opencv/opencv_contrib/blob/4.x/modules/wechat_qrcode/LICENSE), Apache-2.0 | Original `.prototxt` / `.caffemodel` assets; CPU QR decoding through OpenCV. |
| Rust color worker crates | Existing `ccp_core`, `ccp_detect`, `ccp_bridge`, `ccp_video` sources declare MIT in their Cargo manifests. | Compiled with `--no-default-features`; optional `tch`, ONNX chart and video paths are not enabled for deployment. Upstream notices and declarations remain. |
| Rust dependencies | Exact versions and upstream checksums in `development/native/ccp-worker/Cargo.lock`; each dependency retains its license. | Scientific/color processing; source available through the crates named in the lockfile. |
| CPython | [PSF license and included notices](https://docs.python.org/3/license.html) | Official Python 3.12.10 x64 embeddable distribution; its `LICENSE.txt` remains in build inputs and the package. |
| ONNX Runtime / DirectML | [ONNX Runtime](https://github.com/microsoft/onnxruntime), MIT; [DirectML terms](https://github.com/microsoft/DirectML) | Native FP32 CPU/DirectML runtime. Package notices remain in the private runtime. |
| OpenCV, NumPy, pandas, Pillow, Flask and supporting Python libraries | Upstream license files remain in the packaged `*.dist-info` metadata and package directories. See `docs/runtime-packages-1.0.0.json`. | Image operations, numerical arrays, tables and local browser interface. The license asset collects the corresponding notices. |
| Microsoft Visual C++ runtime | Microsoft redistribution terms; original Microsoft signatures and `development/installer/vc-runtime/inventory.json` retained. | App-local runtime DLLs. They are not relicensed under AGPL. |

Torch, Ultralytics, EasyOCR's Python package, SciPy, scikit-image, Numba and Taichi
are excluded from the deployed runtime. An external development environment may
use Torch/Ultralytics to convert a `.pt` checkpoint; its dependencies are separate
from the installed application's dependency inventory.

Model asset names, sizes and SHA-256 hashes are listed in
`docs/build-inputs-1.0.0.json`. The custom potato model and private test photos are
not release assets. Training photos are not needed for normal inference.

The 1.0.0 release retains the tested binaries unchanged;
download the accompanying license asset and this source revision with them.
The public validation summary describes remaining hardware and startup limits.
