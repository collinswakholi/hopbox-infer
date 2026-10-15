# Preserved upstream notices

The authoritative AGPL-3.0 text in the root LICENSE was obtained from
https://www.gnu.org/licenses/agpl-3.0.txt. Copyright and scope are stated in NOTICE.

- `CRAFT-MIT.txt`: https://github.com/clovaai/CRAFT-pytorch/blob/master/LICENSE.
- `EasyOCR-Apache-2.0.txt`: the preserved license beside the vendored geometry source,
  originating at https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE.
- `WeChat-QRCode.txt`: https://github.com/opencv/opencv_contrib/blob/4.x/modules/wechat_qrcode/LICENSE.
- `pulp-wasm-simd-flag-0.1.1-LICENSE.txt`: root license at
  https://github.com/sarah-quinones/pulp/blob/5eb07fd7b68edf0a5e19f71737d315f72a510295/LICENSE.
- `zune-inflate-0.2.54-LICENSE*.txt`: root notices at
  https://github.com/etemesi254/zune-image/tree/69502ce83fdfecdd0beefd677e2abb3781b29d98.

The last two source revisions are the revisions recorded in those downloaded
crates' `.cargo_vcs_info.json`; their crate archives omitted standalone license
files. The release license archive also preserves notices from all other active
worker dependencies available in the local Cargo registry, the installed Python
runtime and its third-party package metadata.
