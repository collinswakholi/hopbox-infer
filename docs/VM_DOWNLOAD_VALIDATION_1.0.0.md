# GitHub download and VM validation: 1.0.0

Tested October 15, 2026. A fresh unauthenticated download of the published installer was installed and run in an offline Windows 11 x64 Sandbox. The installed application EXE performed the processing; no source Python processing script was run. **All remaining checks passed.**

- Installer SHA-256: `852c3096b17e77ace2b5f68f7fabc42d73a17b3ea2bde632cbba9b707f15550b`.
- Fresh install and same-location upgrade: exit 0; chosen Program Files folder with spaces.
- Read-only installation, no system Python, no editable application source; 8 GiB configured VM memory; no enabled network adapters.
- GPU: shared NVIDIA GeForce RTX 4070 Ti SUPER through DirectML; explicit GPU inference and loaded-DLL checks.
- Model readiness: 22.67 seconds.

| Device | Requested batch | Photos | Rows | Seconds | Same-device reference |
| --- | --- | --- | --- | --- | --- |
| NVIDIA DirectML | 1 | 11 | 110 | 89.33 | CSV byte-identical |
| NVIDIA DirectML | 2 | 11 | 110 | 95.03 | CSV byte-identical |
| NVIDIA DirectML | 4 | 11 | 110 | 122.23 | CSV byte-identical |
| NVIDIA DirectML | 8 | 11 | 110 | 98.18 | CSV byte-identical |
| CPU | 1 | 11 | 110 | 199.60 | CSV byte-identical |
| CPU | 2 | 11 | 110 | 203.96 | CSV byte-identical |

Batches 4 and 8 completed in the earlier fresh public-download VM. That earlier session was stopped by the user during batch 2. Batches 1 and 2, both CPU runs and the checks below completed in the new session. Timings are observations under shared host load, not controlled native hardware benchmarks. Production memory safeguards remain enabled.

Additional completed checks:

- Custom ONNX class discovery, mapping/cancel, real preview and default-model restoration.
- Full installed-file integrity check; explicit `--cpu` overrides a GPU preference.
- Runtime and loaded-DLL audits: no Torch, Ultralytics, EasyOCR package, SciPy, scikit-image, Numba, llvmlite or Taichi.
- Browser UI HTTP route: 33 probes during CPU processing, zero failures; maximum 1.824 seconds. This checks HTTP responsiveness, not visual interaction.
- Normal shutdown exit 0, explicit CPU startup/shutdown exit 0, no remaining app processes.

CPU and GPU references are separate: their numerical values are not all byte-identical across devices. Physical AMD Radeon hardware and final physical RTX 4090 application execution remain unvalidated. The binaries are unsigned and managed-endpoint security acceptance remains a separate limitation.

Both test VMs were disposed of after their results were saved. Original source, models, photos, CSVs and distinct validation images remain preserved.
