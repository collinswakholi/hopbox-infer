# HopBox Infer 1.0.0

Open `release/HopBox-Infer/HopBox-Infer.exe`, or use the offline Windows installer.
The package contains a private Python interpreter; no Python installation,
PyTorch, Ultralytics, EasyOCR, Numba, Taichi, SciPy or scikit-image is required.
The distribution's `HOW_TO_INSTALL.txt` explains installation, portable use,
GPU packs, model loading and upgrades.

The app runs ONNX Runtime on DirectML or CPU. Auto prefers a discrete GPU.
The same installer supports NVIDIA, Intel and AMD through DirectML; no separate
vendor pack is needed. Intel and NVIDIA were tested locally; AMD remains
unvalidated on physical hardware. The TensorRT RTX candidate is excluded after
full-folder identifier and numerical parity failures.

The package includes Microsoft Visual C++ runtime DLLs beside its private
interpreter and native worker. ONNX Runtime requires MSVCP140 and MSVCP140_1;
the first fresh VM exposed their absence in 1.3.0. No separate runtime download
or machine-wide installation is needed. Additions are recorded in the release inventory.

Inference now uses FP32 consistently across devices. Image size 2048,
confidence 0.58, IoU 0.75, class roles, original-resolution contours, and
scientific feature formulas are retained. FP32 can produce different boundaries
and measurements from the previous FP16 app; the migration's recorded
comparisons remain in `.archive/migration-research-20261013/portable`.

Color fitting uses the existing Rust worker built without its optional `tch`
feature. Color application now uses the validated Rust CPU kernel on every
device. In the measured comparison it was faster than the Vulkan helper and
removes that helper's compiler runtime. Neural inference still uses the selected
GPU. No runtime compilation is performed.

Rust also performs NMS, original-resolution mask ROI sampling, nearest-point
contour joining, entropy, LBP, color statistics, GLCM and OCR connected-component
labeling. The remaining image geometry operations use compiled OpenCV. The UI,
job orchestration and portable model interface remain in Python.

Import uses the Windows folder picker. Hold Shift while clicking Import folder
to use the existing in-app folder browser instead. Preview Chart, Corrected,
Detections, correction plots, folder processing and class mapping retain the
existing interface. Batch sizes 1, 2, 4 and 8 now control real network batches.
Photos with different rectangular input shapes are grouped separately; input
padding and original-resolution masks are preserved. Memory allocation failures
split the batch before trying single-photo CPU inference. Start with batch 1;
larger batches do not necessarily improve whole-folder speed.

Allocation recovery recognizes OpenCV and ONNX Runtime failures. It releases
failed tensors and the network session before retrying smaller batches, so a
failed large batch does not retain its memory while the smaller retry runs.
Successful CPU batches release their inference memory pool before chart, OCR
and trait processing; the next batch recreates the inference session. This
trades some CPU setup time for lower peak RAM. GPU session reuse is retained.
Chart allocation failures stop the job explicitly instead of skipping photos
as if they had no chart.
Native folder and model pickers use the full Windows PowerShell path, independent
of the application's isolated PATH.

Custom models are ONNX graphs with adjacent JSON metadata containing class
names, stride, preprocessing and the graph checksum. Filenames may differ from
the default model. Use Load model to browse or paste the model path, then assign
class roles. The chosen image size is retained; use 1024 for a model trained at
1024 when appropriate.

Conversion of `.pt` checkpoints runs in an external development environment:

```powershell
venv\Scripts\python.exe tools\export_model_bundle.py C:\Models\custom.pt C:\Models\portable --imgsz 1024
```

To enable the same conversion from Load model, set these variables before
starting HopBox. Both paths must be absolute and point to an external Python
environment with Torch, Ultralytics and ONNX installed, and the exporter script:

```powershell
$env:HOPBOX_EXPORTER_PYTHON = 'C:\ExportTools\venv\Scripts\python.exe'
$env:HOPBOX_EXPORTER_SCRIPT = 'C:\ExportTools\export_model_bundle.py'
& 'C:\HopBox-Infer\HopBox-Infer.exe'
```

The app never imports those frameworks. The converter exits after export;
converted bundles are cached under `%LOCALAPPDATA%/HopBox-Infer/models/imported`. A normal offline
installation accepts ONNX bundles without any converter or Python installation.
If the converter is absent, choosing a `.pt` file displays setup instructions
and leaves the currently loaded model available.

AMD Radeon inference uses DirectML in the standard app, requiring a compatible
DirectX 12 Windows driver. Rust color processing and feature extraction use the
CPU on every platform. Radeon has not been tested on local hardware.

Rebuild with `app/build_exe.ps1` after preparing `.build-env` from
`app/requirements.txt`. See `../BUILD.md` for compiler prerequisites, individual
build commands and clean offline EXE validation.
The wrapper compiles all Rust DLLs and `ccp_bridge` with
`--no-default-features`, preserving the chart handshake and fitting mathematics.

`validation` contains the production regression results, installed-package
checks, dependency inventories and a screenshot of the running interface.
Original training weights and comparison records are preserved outside the
application. Superseded installers and distribution packages have been removed.
The original development environment remains available for external model export
and is excluded from every deployment package.

