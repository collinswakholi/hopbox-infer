# HopBox Infer 1.0.0 — draft

Offline Windows x64 hop segmentation, chart color correction, QR/OCR and trait
extraction by Collins Wakholi. Developed October 14, 2026.

The Windows app replaces the previous Docker/PyTorch distribution with ONNX
Runtime CPU/DirectML and Rust numerical kernels. The installed app contains no
Torch or Ultralytics and needs no system Python or network connection.

Download `HopBox-Infer-Setup-1.0.0.exe` for the destination/component wizard, or
unpack `HopBox-Infer-1.0.0-Windows.zip` and keep its entire folder together.
Download the accompanying `HopBox-Infer-1.0.0-Licenses.zip`, read
`HOW_TO_INSTALL.txt`, and verify the files against `SHA256SUMS.txt`.

The core supports CPU and DirectML through one package. NVIDIA and Intel paths
were exercised during development; physical Radeon execution remains unvalidated.
The managed RTX 4090 machine installed and upgraded correctly, but Cortex blocked
the exact final launcher. The build is unsigned. A final 4090 runtime acceptance
test and Radeon hardware validation are still outstanding.

The launcher has a native splash and persistent startup diagnostics, verifies
the package, and retries once on CPU if a native GPU access violation occurs
before models become ready. `--cpu` explicitly selects CPU; `--diagnose` checks
all package files. No security policy or driver changes are performed.

Models use FP32 with image size 2048, confidence 0.58 and IoU 0.75. All 11-photo
batch 1/2/4/8 runs matched the current NVIDIA reference. FP32 migration and
CPU/GPU differences from older measurements are described in
[the validation summary](VALIDATION_1.0.0.md).

Source is AGPL-3.0-only; third-party components and model terms are retained.
`HopBox-Infer-1.0.0-Build-Inputs.zip` supplies the checksum-pinned models and
private runtime seeds required for a source build. See [BUILD.md](../BUILD.md).
These are build inputs; end users only need the installer or portable package.

The old Docker/PyTorch version is preserved at
[`legacy-docker-2026-10-15`](https://github.com/collinswakholi/hopbox-infer/tree/legacy-docker-2026-10-15).
Existing Docker images were not modified.

This release is deliberately **draft** and is not marked Latest or published.
The installer and portable ZIP are the exact previously tested artifacts.
