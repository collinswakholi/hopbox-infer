# HopBox Infer 1.0.0 validation

This is a curated summary of local records from October 14-15, 2026. Private
photos, SSH details, machine identifiers and raw endpoint logs are not published.
The release contains the tested binaries; publishing source/docs did not rebuild them.

| Check | Result and scope |
| --- | --- |
| Actual Windows EXE, batches 1/2/4/8 | All 11 `119CANON` photos, 110 output rows each; every CSV byte matched the NVIDIA reference. Extra hop on IMG_0002 and QR fallback case on IMG_0003 are included. |
| Independent folders and custom ONNX model | Folder processing, class mapping/cancel/restore, model-aware previews and clean shutdown passed. |
| Fresh offline Windows Sandbox | Installer, chosen Program Files destination, upgrade, read-only install and actual EXE passed without system Python or network. NVIDIA host vGPU run: 122.46 s for 11 photos; CPU run: 271.70 s. These are VM observations, not native hardware benchmarks. |
| CPU / GPU comparison | All 110 identities retained. CPU output exactly matched the earlier CPU reference. CPU vs NVIDIA differed in 36 numeric columns, worst absolute difference 0.000115181 in Green_GLCM_contrast. Do not infer cross-device CSV byte identity. |
| Startup recovery | A disposable test bootstrap produced native exit 0xC0000005; the launcher retried once on CPU and completed real photo processing. Production contains no fault injection switch. Later mid-job native crashes are outside this startup recovery. |
| Installer acceptance | Fresh install, upgrade, selected destination, file inventories and locked-staging rollback passed locally. A managed RTX 4090 machine passed fresh install and upgrade, all 3,303 file hashes. Its launcher execution was blocked by Cortex before diagnosis; no 4090 app run is claimed. |
| Package audit | 139 native binaries audited. No Torch/Ultralytics/EasyOCR/Numba/Taichi/SciPy/scikit-image package or forbidden native dependency. A private Python runtime remains. |
| Maintenance verification | Scientific kernels against SciPy/scikit-image, 68 NMS cases, allocation recovery, GPU selection, ZIP CRC, 36 application source hashes and EXE full integrity passed after cleanup. Real `.pt` export reproduced the existing custom ONNX graph byte for byte; both CPU output arrays were identical. |

The default model settings remain image size 2048, confidence 0.58, IoU 0.75,
FP32, original-resolution mask coordinates and the established feature formulas.
The older Docker/FP16 application is not a byte-identical scientific baseline.
Recalibrate comparisons when migrating existing experiments from it.

Physical AMD Radeon execution has **not** been tested. DirectML adapter selection
was tested, including an AMD fixture; that is not hardware certification. A
Sandbox vGPU shares the host GPU and does not emulate a Radeon.

The optional `--gpu-generic` mode is diagnostic: masks/rendered pixels matched
locally, but two calibration values differed by at most 9.77e-15, so its CSVs
were not byte-identical. It is not enabled by default.

The binaries are unsigned. A confirmed Cortex prevention event affects the
final launcher on the managed 4090 machine. See [startup guidance](../SECURITY_AND_STARTUP.md).

Checksums of the tested artifacts:

```text
852c3096b17e77ace2b5f68f7fabc42d73a17b3ea2bde632cbba9b707f15550b  HopBox-Infer-Setup-1.0.0.exe
b2b765447a61ae03c50c2b1328084b28aa4b5b2cc8e048e0619fdc8c6398facb  HopBox-Infer-1.0.0-Windows.zip
1895abedad1fca6fd1b97d54d3afdb4940de578f87609f7011ae886da4c4cd95  HopBox-Infer.exe
```
