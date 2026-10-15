# HopBox Infer

Offline Windows x64 hop segmentation, chart color correction, QR/OCR and
scientific feature extraction by [Collins Wakholi](https://github.com/collinswakholi).
Version **1.0.0**, developed October 14, 2026.

![HopBox Infer logo](app/hopbox/static/branding/hopbox.png)

The app uses ONNX Runtime CPU/DirectML and Rust numerical kernels. It includes
its own private Python runtime and needs no system Python or network connection.
The installed app contains no Torch, Ultralytics, EasyOCR package, SciPy,
scikit-image, Numba or Taichi.

## Download and install

Download **v1.0.0** from the
[latest release](https://github.com/collinswakholi/hopbox-infer/releases/latest).

- Use `HopBox-Infer-Setup-1.0.0.exe` for the installer wizard. Choose the destination
  and shortcut options; Program Files is suggested and elevation is requested
  only when needed.
- Or extract `HopBox-Infer-1.0.0-Windows.zip` and open `HopBox-Infer.exe`. Keep the
  complete folder together.
- Download the accompanying license archive, read `HOW_TO_INSTALL.txt`, and
  verify downloads using `SHA256SUMS.txt` and PowerShell `Get-FileHash`.

Core includes both CPU and DirectML. No vendor-specific GPU pack is required.
Auto prefers a discrete GPU. Physical AMD Radeon execution remains unvalidated;
DirectML selection checks are not hardware certification. The unsigned final
launcher was blocked by Cortex on one managed RTX 4090 machine, although fresh
installation and upgrade passed. See [startup guidance](SECURITY_AND_STARTUP.md).

## Use the app

Import photo folders, select the folders/images to process, preview detections
and correction, then use **Run Folders**. Start with batch 1; larger batches
1/2/4/8 need more memory and do not necessarily improve complete-folder speed.
The app writes traits to `Features.csv` and saves image previews beside results.

The defaults are image size 2048, confidence 0.58 and IoU 0.75. Inference uses
FP32 on CPU/GPU. Rust performs color processing, NMS, mask sampling and scientific
feature kernels. Scientific migration and cross-device differences are explained
in the [validation summary](docs/VALIDATION_1.0.0.md).

Load another segmentation model as an ONNX graph with adjacent JSON class,
preprocessing and checksum metadata. Class roles are assigned in the app.
External `.pt` conversion is optional and requires a separate development
Torch/Ultralytics environment; those packages are never imported by the app.
See [deployment and model instructions](app/DEPLOYMENT.md).

Installations can be read-only. Settings, logs, caches and imported models live
under `%LOCALAPPDATA%/HopBox-Infer`, or `HOPBOX_STATE_ROOT`. The native splash
waits for models, and startup retries once on CPU after an early native GPU
access violation. `--cpu` selects CPU explicitly; `--diagnose` checks all files.

## Build and validate

See [BUILD.md](BUILD.md) for Windows compilers, pinned dependencies, model/runtime
input staging and EXE packaging. The [Windows workflow](.github/workflows/windows.yml)
builds and checks source without publishing releases. Full photo/GPU/VM acceptance
remains separate from hosted CI. [Workspace guide](docs/WORKSPACE.md).

Private datasets, virtual environments, raw endpoint logs, runtime binaries and
large model artifacts are excluded from source control. The release's checksum-pinned
build-input archive supplies the exact deployment models and runtime seeds.
The source license and notices accompany the tested binary assets unchanged.

## Previous Docker version

The older Docker/PyTorch application is preserved at
[`legacy-docker-2026-10-15`](https://github.com/collinswakholi/hopbox-infer/tree/legacy-docker-2026-10-15)
(commit `15d0729`). Its Docker instructions and Git LFS model pointers remain
available there. Existing Docker images were not modified. This Windows release
does not replace Docker Hub's image with a different platform artifact.

## License

Original HopBox source: **GNU AGPL-3.0-only**, Copyright (C) 2026 Collins Wakholi.
See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third-party/model notices](THIRD_PARTY_NOTICES.md). Third-party components and
trained model terms remain applicable.
