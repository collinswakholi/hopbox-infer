# Build HopBox Infer for Windows x64

The installed app is offline and self-contained. A source build needs Windows
x64, Python **3.12.10**, Rust **1.95.0** (MSVC target), Visual Studio C++ Build
Tools/Windows SDK, and the .NET Framework C# compiler at
`C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe`.
The pinned Python 3.12 build emits bytecode compatible with the private runtime.

## Stage the model and runtime inputs

The source repository excludes large models and binary seeds. Obtain
`HopBox-Infer-1.0.0-Build-Inputs.zip` from the draft/released `v1.0.0` assets.
It contains the tested FP32 ONNX/QR models, official CPython embeddable seed and
Microsoft-signed app-local CRT DLLs. Every file and the archive itself are pinned
by SHA-256 in [docs/build-inputs-1.0.0.json](docs/build-inputs-1.0.0.json).

```powershell
py -3.12 tools/prepare_build_inputs.py --archive C:\Downloads\HopBox-Infer-1.0.0-Build-Inputs.zip
```

Or let the script download the pinned archive:

```powershell
py -3.12 tools/prepare_build_inputs.py --download
```

While the release is draft, the download requires private draft-release access through
`GH_TOKEN` / `GITHUB_TOKEN`, or an owner-downloaded local archive. Do not put
credentials in source files. The script verifies checksums, rejects unexpected
members/reparse points and refuses to overwrite differing existing inputs.
It stages `app/models/portable`, `installer/python-seed` and CRT DLLs under
`installer/vc-runtime`. Chart reference JSONs and the CRT inventory are in Git.

## Create the isolated build environment

```powershell
py -3.12 -m venv .build-env
.build-env\Scripts\python.exe -m pip install -r requirements-build.txt
```

`requirements-build.txt` pins all 25 runtime distributions plus `pefile` for the
package audit. `app/requirements.txt` lists direct runtime requirements.
For numerical reference tests, install `requirements-validation.txt` instead.
SciPy/scikit-image and their reference dependencies are excluded from packaging.
Do not install Torch/Ultralytics/EasyOCR/Numba/Taichi into `.build-env`.

Network access is needed to fetch Python wheels, the Rust toolchain/crates and
build inputs unless separately cached. This is distinct from the fully offline
installed app. An offline rebuild additionally needs a wheelhouse, Rust toolchain
and Cargo registry/vendor cache; these are not claimed to be included in the
model/runtime input archive.

## Compile and package

Open **x64 Visual Studio Developer PowerShell** in the checkout:

```powershell
powershell -NoProfile -File tools\build_native.ps1
.build-env\Scripts\python.exe tools\build_release.py
.build-env\Scripts\python.exe tools\audit_release.py
```

The Rust worker is built `--locked --no-default-features`; optional `tch` and
native ONNX chart paths are disabled. The standalone DLLs target baseline x64,
not the host CPU's AVX instruction set. ONNX charts use the established worker
handshake. Original Rust crate MIT declarations and notices are retained.

Packaging refuses an existing `release/HopBox-Infer`. Use a fresh checkout or a
separate stage (`--output .build-tmp/release-next`). It creates the launcher,
installer, portable ZIP, file inventory and checksums. Never overwrite a running
installation. The UI is Python orchestration plus a native C# splash/launcher;
application modules ship as Python 3.12 bytecode, not editable `.py` source.
Third-party runtime Python modules and their license metadata remain.

Reproducibility here means pinned input artifacts, toolchain and settings, with
recorded provenance; .NET executable build identifiers are not guaranteed to be
bit-for-bit reproducible. The draft contains the exact previously tested EXE/ZIP,
not an untested rebuild made solely for publishing.

## Verify a fresh build

```powershell
.build-env\Scripts\python.exe -m pip install -r requirements-validation.txt
New-Item -ItemType Directory -Force validation | Out-Null
.build-env\Scripts\python.exe tools\test_rust_science.py
.build-env\Scripts\python.exe tools\test_postprocess.py
.build-env\Scripts\python.exe tools\test_memory_recovery.py
.build-env\Scripts\python.exe tools\test_gpu_selection.py
.build-env\Scripts\python.exe tools\test_installer.py
& .\release\HopBox-Infer\HopBox-Infer.exe --diagnose
```

These checks run locally on a fresh checkout with build inputs. Hosted Windows
CI remains **contents:read**: it compiles the Rust worker/DLLs and C# source and
runs numerical/memory/selection checks without private model/runtime assets.
Full installer/package/integrity validation of the private draft inputs is
performed locally and recorded. The source-only C# outputs are explicitly
labelled and have no distribution payload.

After release inputs become public, manual dispatch with `retain_distribution`
can build/audit the complete package and retain it as an Actions artifact. It
does not publish a release. That option cannot fetch private draft inputs with
the read-only Actions token; use the owner-downloaded archive locally instead.

Full photo/UI/GPU/VM checks use local datasets, reference CSVs, a custom ONNX
fixture and representative label photo. See `docs/WORKSPACE.md`, the arguments of
`tools/prepare_vm_test.py` / `test_model_loading_web.py`, and
[the recorded validation scope](docs/VALIDATION_1.0.0.md). A VM vGPU shares the
host device; it cannot certify AMD without physical AMD hardware.

## Optional conversion, signing and maintenance

Use `tools/export_model_bundle.py` in a separate Torch/Ultralytics/ONNX environment
for `.pt` conversion. Only the resulting ONNX/JSON bundle goes into the app.
See `app/DEPLOYMENT.md`. Python/GPU compiler environments never ship.

The builder supports an existing authorized signing certificate through
`HOPBOX_SIGN_CERT_THUMBPRINT`, `HOPBOX_SIGNTOOL` and an HTTPS
`HOPBOX_TIMESTAMP_URL`. Signing verifies `/pa`; no certificate or trust root is
created. The tested 1.0.0 artifacts are unsigned. Signing does not guarantee a
security product will accept a file. See `SECURITY_AND_STARTUP.md`.

`--repack` recompiles C# and repacks existing source-verified runtime files;
`--installer-only` rebuilds setup from the checksummed portable payload. These
create new artifact hashes and require renewed acceptance. Python/native source
changes require a full build. Update `app/hopbox/branding.json` to change version,
author, development date or profile link.

See `docs/WORKSPACE.md` for which development folders to keep. Build inputs,
virtual environments, private validation logs and distributions remain ignored.

