# HopBox Infer

**Measure hop appearance from your HopBox photos, without an internet connection.**

HopBox Infer helps breeders turn tray photographs into measurements of hop size,
shape, colour and texture. It finds individual hops, uses the colour chart to
correct colour, reads QR codes or printed labels, and saves results in a table
you can open in Excel. Review the detected outlines and sample labels before
using the measurements to compare breeding material.

<img src="app/hopbox/static/branding/hopbox.png" alt="HopBox Infer logo" width="120">

**[Download the Windows installer](https://github.com/collinswakholi/hopbox-infer/releases/download/v1.0.0/HopBox-Infer-Setup-1.0.0.exe)**
&nbsp; | &nbsp;
**[Read or download the illustrated user guide (PDF)](docs/HopBox-Infer-User-Guide.pdf)**

Version **1.0.0** · By [Collins Wakholi](https://github.com/collinswakholi).

## What you need

- A **64-bit Windows PC**. This version has been tested on Windows 11.
- Photos taken using the established HopBox setup, with the colour chart and
  sample label clearly visible. Supported files: JPG, JPEG, PNG, TIF, TIFF and BMP.
- Space for the app and your results. The installer is about **368 MB** and the
  installed app about **613 MB**; allow extra space for installation and photos.
- A graphics processor (GPU) is optional. Processing on the PC's main processor
  (CPU) is included. NVIDIA and Intel GPU paths have
  been tested; physical AMD Radeon execution is still untested.

You do **not** need Python, Docker, PyTorch, Ultralytics or a separate GPU package.
Once downloaded and installed, the app processes photos offline on your PC.

## Install and open

1. Download **`HopBox-Infer-Setup-1.0.0.exe`** using the link above. The green
   **Code > Download ZIP** button on GitHub downloads source code, not the app.
2. Open the installer. Choose the installation folder and shortcut options.
   Use a location permitted by your organisation. Windows may request
   administrator approval for a protected folder such as Program Files.
3. Keep the standard core component: it includes both CPU and GPU support.
   Click **Install**, then open **HopBox Infer**.
4. Wait for the models to finish loading. The app opens its interface in a
   browser window; it is running locally, not on a website that receives photos.

For a portable copy, use the Windows ZIP on the
[release page](https://github.com/collinswakholi/hopbox-infer/releases/latest),
extract the whole folder, and open `HopBox-Infer.exe`. Keep all its files together.

## Your first folder

1. Click **Import folder** and select the folder containing your photos. A parent
   folder can include several trial folders. Tick the folders you want to process.
2. Click a folder name, then a photo in the **Photos** list to inspect it.
3. In **Run settings**, leave **Compute device** on **Auto** to start. Choose
   **CPU** if GPU processing is unavailable. Start with **Batch size: 1 photo**.
4. Choose an **Output folder** using **Browse**. Use a fresh results folder for
   each run to avoid replacing earlier results. Keep it outside your photo folders.
5. Check a few representative photos with **Detections**, **Chart** and
   **Corrected**. Check hop outlines, chart detection and label readability.
6. Click **Run folders**. Follow **Progress** and **Activity**, and wait for the
   run to finish before opening results or clicking **Exit**.

Keep the supplied settings initially: **Image size 2048**, **Confidence 58%**,
**CCM 3×3**, gamma correction and white balance on, and
**QR code, then printed text**. Keep settings and compute choice consistent
within a study; CPU/GPU measurements can have small numerical differences.

**Batch size is not the number of folders to process.** It controls how many
photos are handled together. Batch 1 still processes every photo in the ticked
folders. Try 2, 4 or 8 only after a small run succeeds; larger batches use more
memory and are not always faster.

## Find and check your results

Inside your chosen output folder, each processed photo folder gets:

- **`Features.csv`**: a table with a row for each measured hop, its source photo,
  label information and calculated traits. Open it in Excel or your analysis tool.
- **`Image_Predictions`**: corrected photos, with hop outlines when **Save hop
  outlines** is enabled.
- **`Chart_Detections`**: photos showing the colour-chart detection.

Check missed or merged hops, chart detection and label text before combining
results. A photo without a detected chart or measurable hops may contribute no
rows; check **Activity** and the saved previews. Your original photos are not
modified. A preview alone does not create the complete measurements table.

The [PDF guide](docs/HopBox-Infer-User-Guide.pdf) includes an annotated screen,
step-by-step instructions, batch processing, results and troubleshooting.

## If something goes wrong

- **Slow or out of memory:** select **CPU**, reduce **Batch size** to 1, close
  other applications, and try a small folder in a fresh output location.
- **No photos appear:** check the file formats above and choose the actual photo
  folder. Folder names beginning with `.` or `_` are skipped during scanning.
- **Results look wrong:** inspect **Detections** and **Chart** first. Check photo
  quality and the sample label; avoid changing scientific settings mid-study.
- **App will not open or security software blocks it:** see
  [installation and startup help](SECURITY_AND_STARTUP.md). This build is unsigned;
  do not disable your organisation's protection to run it.

When reporting a problem, include the app version, compute choice and message
shown in **Activity**. Review logs and photos before sharing them publicly.

## For developers and existing users

The app source is in [`app`](app). Build, installer, Rust and test sources are
grouped under [`development`](development); they are not needed to install or
use HopBox Infer. See the [build instructions](development/BUILD.md),
[validation record](docs/VALIDATION_1.0.0.md), and
[custom model instructions](app/DEPLOYMENT.md).
The [GitHub-download VM test report](docs/VM_DOWNLOAD_VALIDATION_1.0.0.md)
records completed GPU batches 1/2/4/8 and CPU batches 1/2.

The previous Docker/PyTorch version is preserved at
[`legacy-docker-2026-10-15`](https://github.com/collinswakholi/hopbox-infer/tree/legacy-docker-2026-10-15).
Existing Docker images are unchanged. Measurements from the older version should
be checked before combining them with this Windows version.

Original HopBox source: **AGPL-3.0-only**, Copyright (C) 2026 Collins Wakholi.
See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party/model notices](THIRD_PARTY_NOTICES.md). The release also includes
license notices and download checksums.
