# Development checkout

`app` contains the current source, assets and compiled Rust helpers. `native`
contains the Rust color worker and Cargo lockfile. `installer` contains C# source
and verified runtime input inventories. `tools` contains reproducible build,
conversion, audit and regression tools. Keep these source folders.

`.build-env` is a local Python 3.12 build/test environment. `venv` is a separate
legacy reference and external `.pt` converter environment; it may contain Torch
and Ultralytics. Neither environment is copied to Git or to another computer.
`release/HopBox-Infer` is the self-contained installed runtime.

`validation` contains local scientific records and fixtures, excluded from Git.
`Data`, `Data_2` and `.archive` preserve photos, original models and migration
references. They are not needed for normal app use and are not release assets.
The numerical reference code needed by portable CI is in `tests/references`.
Full folder, real-model and VM acceptance tests require locally supplied photos,
reference CSVs and a custom-model fixture. See each tool's arguments.

October 15 cleanup removed approximately 8.25 GiB of obsolete packages, caches
and byte-identical generated previews. Every scientific CSV and distinct image
was retained locally. Test tools may regenerate disposable outputs.
