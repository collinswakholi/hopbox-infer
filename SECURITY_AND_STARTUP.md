# Startup and managed Windows installations

The 1.0.0 launcher begins persistent logging before application imports and
reports missing/changed files and native exit codes directly. It retries once on
CPU after native exit `0xC0000005` before models are ready, while preserving the
original failure. It does not recover a later mid-job native crash.

```powershell
& 'C:\Program Files\HopBox-Infer\HopBox-Infer.exe' --cpu
& 'C:\Program Files\HopBox-Infer\HopBox-Infer.exe' --diagnose
```

Keep the complete installation together. Logs and diagnostic reports are under
`%LOCALAPPDATA%\HopBox-Infer`, or `HOPBOX_STATE_ROOT`. Installations can be read-only.

A test machine's original startup failure was a native `nvcuda64.dll` access
violation (`0xC0000005`, NVIDIA driver `32.0.15.9579`), not a missing app folder.
The old generic message was misleading. The original EXE started successfully
on CPU. A disposable fault fixture verified the rebuilt launcher's one-time CPU
recovery followed by real photo processing.

A separate confirmed Cortex prevention event affects the final launcher:

- October 14, 2026, 23:20:38 UTC; provider `TrapsV2`, event 85.
- Description: suspicious executable detected; status `0xC0400055`.
- Exact launcher SHA-256:
  `1895abedad1fca6fd1b97d54d3afdb4940de578f87609f7011ae886da4c4cd95`.
- Fresh installation and upgrade on the managed RTX 4090 machine passed;
  all 3,303 file hashes matched. The subsequent `--diagnose` invocation was
  blocked, exited -1, and produced no report. No 4090 runtime test is claimed.

The generic event does not establish the precise classifier reason. The binaries
are unsigned; author/company metadata does not establish a verified publisher.
An authorized signing workflow is supported by the builder, but no certificate
was available for these tested artifacts. Signing cannot guarantee approval.

CPU recovery does not retry prevention codes or restore quarantined files.
No security settings, exclusions, trust roots or drivers were changed by the
validation tools. Raw endpoint/SSH records stay private. See the
[public validation summary](docs/VALIDATION_1.0.0.md) for test scope and limitations.

`--gpu-generic` disables DirectML vendor metacommands for diagnostics. It is not
the default: two calibration values differed locally by up to 9.77e-15 despite
matching masks/pixels. This mode is unvalidated on the RTX 4090 test machine.
