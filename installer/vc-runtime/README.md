# Microsoft Visual C++ runtime inputs

These Microsoft-signed x64 DLLs are local build inputs, with versions, original
signer names and SHA-256 hashes in `inventory.json`. The builder verifies every
inventoried file before creating a fresh package. Keep the complete set.

The original release ZIP was verified as SHA-256
`26df483db1636e11607764b24e443b3541ef183fb8bd37c81d287e78822c1883`
before recovering the inputs on October 14, 2026. The `vccorlib140.dll` recovery
was additionally checked against the inventory and its Microsoft signature.
See `validation/security-startup-fix/input-recovery.json` and
`runtime-restoration.json` for the local recovery record.

Do not replace these files with arbitrary DLL downloads. To refresh the runtime,
use Microsoft's licensed redistributable/build tools, retain the redistribution
terms, verify signatures and record the new versions and hashes.
