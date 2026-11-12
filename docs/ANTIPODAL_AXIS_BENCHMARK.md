# Antipodal direction sampling: validation and timings

2026-11-07. Shared implementation: `corner_detector/antipodal_support.py`; callers are the detector and `corner_lab_open3d_cone.py`.

## Numerical checks

- Counts 42/64/162/642/2562 contain unit vectors, no duplicates and exactly one opposite per direction. Only 21/32/81/321/1281 unoriented axes are evaluated.
- The original 42/64 coordinates and order are bit-identical. Larger icospheres retain smaller ones as prefixes.
- Optimized Cone/Tube results match an independent full-signed reference: min-side score, total-support ties, earliest signed-axis index, signed optimum and both masks. Tests include duplicate/zero samples, cap/tube boundaries and ties.
- Cone and Tube use the same grids and independently select their best axes. Q10/CC are unchanged when switching method. Existing lab functions/dataclasses remain available.
- The standalone GUI was exercised at all five counts with both Cone and Tube overlays. The hidden Filament test process stalled during teardown after all checks and close requests, and was terminated; normal standalone shutdown is not certified by this harness. No live sensor test was performed.

## CPU microbenchmark

Python 3.10.12, NumPy 1.21.5; `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. No concurrent test suite during this run.
2048 deterministic synthetic nonzero offsets: NumPy seed 20261107, Gaussian unit directions, independent uniform radii 1–45 mm. Cone half-angle 35 degrees; Tube radius 4 mm.
Prebuilt direction grids, 3 warmups, median of 21 evaluations. Full-signed reference evaluates every direction and materializes its full masks; optimized code uses +/- representatives and bounded 64-axis blocks. Grid construction and the rest of the detector pipeline are excluded.

| Signed directions | Evaluated axes | Cone full (ms) | Cone optimized (ms) | Tube full (ms) | Tube optimized (ms) |
|---:|---:|---:|---:|---:|---:|
| 42 | 21 | 0.648 | 0.205 | 0.731 | 0.203 |
| 64 | 32 | 0.971 | 0.213 | 1.057 | 0.265 |
| 162 | 81 | 2.321 | 0.402 | 2.762 | 0.513 |
| 642 | 321 | 6.253 | 1.255 | 8.749 | 1.776 |
| 2562 | 1281 | 22.515 | 4.825 | 41.183 | 6.847 |

At 642 directions this fixture is approximately 5x faster than full-signed evaluation. Gains include reduced matrix/mask memory, not only halved dot products. Timings depend on hardware, BLAS and point distribution; they are not end-to-end detector or sensor latency.

642 directions are denser than the old 42 default and can increase both per-query work and the maximum support score. Use 42 explicitly for prior numerical comparisons. The experimental B threshold remains .08 with a 3-point minimum on each side; denser sampling does not calibrate that threshold.
