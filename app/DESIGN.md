# HopBox web app

The browser is the control panel. Large tray photos open in a preview stage on top of the page, not inside a tiny thumbnail.

## Palette

- Mist `#E7EEEA` behind the page
- Tray `#1A2E24` for the header only
- Sheet `#F7FAF8` for the working areas
- Leaf `#2F6B45` for Import, Preview, and Run
- Bract `#7A6233` for photo counts only

The name uses Georgia. Everything else uses Segoe UI so a packaged build does not download a font.

## Layout

Left: folder tree. Each row is a checkbox, the folder name, and the number of photos in that folder and below it. The header shows how many of those photos are selected.

Right: one photo from the folder you clicked, then the batch settings. Color models are CCM, RPCC2, and RPCC3. QR reading can be skipped, used alone, or followed by OCR when the code fails.

Preview opens a separate stage with the chart, each corrected photo, and the hop outlines.
