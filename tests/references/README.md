# Numerical reference implementations

These are the original NumPy/OpenCV/SciPy/scikit-image implementations preserved
before the Rust ports. They are test-only and are not copied into the app.
They contain no photos or trained weights. The code is copied without changing
its formulas; `test_rust_science.py` loads it for differential checks on synthetic
inputs. Full real-image integration still requires local migration fixtures.

- `features_numpy.py`: SHA-256 `2ab9a3f713543991dc141c361ec1477a8ad1851048df71575a420430e2d240c0`.
- `decode_numpy.py`: SHA-256 `e2eab18e8f9697f98df38dc2639a0ef7fa8a288761c29f0da0f3f1829edea360`.

The reference decoder imports the current NMS through `hopbox.inference`;
independent NumPy NMS parity is checked separately in `test_postprocess.py`.
