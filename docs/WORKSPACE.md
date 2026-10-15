# Development checkout

Users need only the installer or portable Windows package. The
[user guide](HopBox-Infer-User-Guide.pdf) explains installation and use.

`app` contains the application source, assets and locally compiled helpers.
`development` groups build sources: `tools` for packaging/conversion/validation,
`installer` for C# setup/launcher and input inventories, `native` for the Rust
worker, and `tests/references` for independent numerical checks. See
[build instructions](../development/BUILD.md). These are required to rebuild
the app; they are not end-user installation steps.

`.build-env` is the local Python 3.12 build/test environment. `venv` is a separate
legacy reference and external `.pt` converter; it may contain Torch/Ultralytics.
Neither environment is copied to Git or another computer. `release/HopBox-Infer`
is the self-contained runtime. Runtime seeds under `development/installer` are
also excluded from Git and staged from the pinned build-input release archive.

`validation` contains local scientific records and fixtures, excluded from Git.
`Data`, `Data_2` and `.archive` preserve photos, original models and migration
references. They are not needed for normal app use and are not release assets.
Full folder, real-model and VM checks require locally supplied photos, reference
CSVs and a custom-model fixture; see the acceptance tools' arguments.

Generated test outputs may be regenerated. Scientific CSVs, distinct validation
images and original datasets remain preserved when temporary payloads or
byte-identical preview copies are cleaned up.
