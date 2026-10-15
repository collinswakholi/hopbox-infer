# Development sources

App users can [download the installer](https://github.com/collinswakholi/hopbox-infer/releases/latest)
and follow the [user guide](../docs/HopBox-Infer-User-Guide.pdf). Nothing in this
folder needs to be installed separately.

This folder keeps the source needed to build, maintain and verify HopBox Infer:

- `tools`: packaging, model conversion, audits and acceptance checks.
- `installer`: C# setup/launcher sources and runtime input inventories.
- `native`: Rust colour worker, dependency lockfiles and original notices.
- `tests`: independent numerical reference implementations.

Follow [BUILD.md](BUILD.md) from the repository root. Local runtime seeds,
compiler environments and private test photos remain excluded from Git.
The immutable `v1.0.0` tag retains the original release layout; current `main`
uses this grouped development layout. Published installer bytes are unchanged.
