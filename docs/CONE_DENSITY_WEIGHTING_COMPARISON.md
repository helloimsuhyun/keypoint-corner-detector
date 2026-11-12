# Direction-density weighted Q10: controlled comparison

Date: 2026-11-06.

Fixed query at the origin, one supplied NMS ID (Fusion 0.8), identical raw inputs for OFF/ON. Reuses `tests/test_ced_corner_synthetic.py` surfaces. R=12 mm, CC factor=4, q=0.1, threshold=0.4, Local cap=40°, steps=5°/15°, density angle=10°, weight bounds=0.5–2.0. The unequal-density fixture retains the dense z=0 face and samples the other faces on a 3 mm grid. Gaussian noise uses seed 19, sigma=0.03 mm; the isolated point is (-6,-6,-6) mm.

Updated decision policy: Density ON + Local ON uses the listed Local weighted optimum for primary status/E and blue ACCEPT markers. Density OFF preserves the original Supporting Plane decision, with Local shown for comparison. The raw and weighted Q values are evaluated at the same listed Local axis. OFF selects interpolated raw Q; ON selects discrete weighted Q. These are controlled fixture outcomes, not end-to-end detection precision/recall or sensor validation.

| Surface | Mode | Primary status / E | Local axis (x,y,z) | Local raw Q | Local weighted Q | Half-angle ° | Raw coverage % | Weighted coverage % |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1-face-corner | OFF | ACCEPT / 0.707107 | (0.6608, 0.7480, 0.0616) | 0.738657 | n/a | 42.383 | 90.00 | 90.00 |
| 1-face-corner | ON | ACCEPT / 0.747994 | (0.6608, 0.7480, 0.0616) | 0.738657 | 0.747994 | 41.583 | 89.09 | 90.50 |
| 2-face-corner | OFF | ACCEPT / 0.573015 | (0.6320, 0.6166, 0.4694) | 0.616572 | n/a | 51.934 | 90.48 | 90.48 |
| 2-face-corner | ON | ACCEPT / 0.620745 | (0.6207, 0.5814, 0.5260) | 0.590721 | 0.620745 | 51.629 | 88.57 | 90.17 |
| 3-face-corner | OFF | ACCEPT / 0.626475 | (0.5774, 0.5774, 0.5774) | 0.626475 | n/a | 51.209 | 90.00 | 90.00 |
| 3-face-corner | ON | ACCEPT / 0.631933 | (0.5774, 0.5774, 0.5774) | 0.626475 | 0.631933 | 50.807 | 90.00 | 90.53 |
| 1-face-edge-interior | OFF | REJECT / 0.173648 | (0.2962, 0.9397, -0.1710) | 0.270264 | n/a | 74.320 | 90.00 | 90.00 |
| 1-face-edge-interior | ON | ACCEPT / 0.475170 | (-0.0436, 0.9962, 0.0755) | 0.234225 | 0.475170 | 61.630 | 83.33 | 90.05 |
| 2-face-edge-interior | OFF | REJECT / 0.138675 | (0.0000, 0.7071, 0.7071) | 0.138675 | n/a | 82.029 | 90.28 | 90.28 |
| 2-face-edge-interior | ON | REJECT / 0.150510 | (0.0616, 0.6608, 0.7480) | 0.136870 | 0.150510 | 81.344 | 89.24 | 90.18 |
| plane-interior | OFF | INDETERMINATE / n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| plane-interior | ON | INDETERMINATE / n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| uneven-corner | OFF | ACCEPT / 0.652020 | (0.6896, 0.6896, 0.2210) | 0.680480 | n/a | 47.119 | 89.92 | 89.92 |
| uneven-corner | ON | ACCEPT / 0.706569 | (0.7066, 0.7066, -0.0390) | 0.547429 | 0.706569 | 45.044 | 85.27 | 91.86 |
| isolated-noise | OFF | ACCEPT / 0.626475 | (0.5774, 0.5774, 0.5774) | 0.626475 | n/a | 51.209 | 90.00 | 90.00 |
| isolated-noise | ON | ACCEPT / 0.631933 | (0.5774, 0.5774, 0.5774) | 0.626475 | 0.631933 | 50.807 | 90.00 | 90.53 |
| uneven-noisy-corner | OFF | ACCEPT / 0.638406 | (0.6839, 0.6629, 0.3046) | 0.662304 | n/a | 48.524 | 90.00 | 90.00 |
| uneven-noisy-corner | ON | ACCEPT / 0.697369 | (0.7004, 0.7121, 0.0488) | 0.545228 | 0.697369 | 45.784 | 83.08 | 90.13 |

ON density/weight summaries (min / median / max):

| Surface | Support N | Direction density | Weight |
|---|---:|---|---|
| 1-face-corner | 110 | 14 / 21 / 28 | 0.8257 / 1.0000 / 1.1481 |
| 2-face-corner | 210 | 14 / 22 / 28 | 0.8076 / 1.0000 / 1.1229 |
| 3-face-corner | 300 | 15 / 22 / 28 | 0.8341 / 1.0000 / 1.1229 |
| 1-face-edge-interior | 120 | 2 / 18 / 24 | 0.5000 / 1.0000 / 1.1471 |
| 2-face-edge-interior | 288 | 9 / 14 / 23 | 0.8165 / 1.0000 / 1.2649 |
| uneven-corner | 129 | 1 / 21 / 28 | 0.5000 / 1.0000 / 1.1481 |
| isolated-noise | 300 | 15 / 22 / 28 | 0.8341 / 1.0000 / 1.1229 |
| uneven-noisy-corner | 130 | 0 / 21 / 28 | 0.5000 / 1.0000 / 1.1481 |

Observed limits:

- All three clean corner queries remain ACCEPT in both modes. The two edge queries change from 0/2 ACCEPT with OFF to 1/2 ACCEPT with ON: the single-face edge is a false positive at the fixed 0.4 threshold. This tiny fixture set does not establish a performance rate.
- The symmetric plane remains INDETERMINATE because CED is undefined. Its existing fail-open keep policy is unchanged; INDETERMINATE is not an explicit ACCEPT.
- The unequal-density corner has 19 nonzero samples on the sparse vertical faces (z>0). Local includes 6/19 with OFF and 0/19 with ON. Those samples retain positive weights but their share of total support drops from 14.73% to 8.14%; ON can exclude the entire sparse-face support while achieving 91.86% weighted coverage (85.27% actual point coverage). This is an observed loss of genuine sparse-face evidence.
- Isolated noise is removed by the unchanged CC selection in both modes and does not alter their corresponding clean-corner score. The unequal-density noisy corner remains ACCEPT in both modes, but actual Local point coverage falls from 90.00% to 83.08% with ON.
- Nonzero minimum weights do not guarantee face retention. Equal weights can still change Q due to discrete inverse-CDF selection rather than NumPy interpolation. Density calculation adds O(N²) work with bounded 512×512 pairwise blocks. The current user settings select Density ON and Local ON; use --no-cone-density-weighted for the previous unweighted path.

Reproduction: `python3 -m unittest discover -s tests -p test_cone_density_weighting.py -v`. The table is generated by the same module’s `surface_comparisons()` helper; no detector scoring or NMS is changed.

Initial density-extension validation: all 106 related tests passed, including 11 new tests for weighted arithmetic, self-exclusion/batching, fixed-query support, negative-score optimization/ties, coverage and read-only debug geometry. Five OFF paths matched every original numerical field against a pre-change snapshot, excluding elapsed time. The full suite ran 288 tests and retained the identical pre-existing 3 failures and 15 errors. CLI help, imports and whitespace checks passed. Matplotlib debug previews were rendered headlessly and the weighted preview was visually inspected; actual Open3D GUI and sensor paths were not run.

2026-11-06 decision update: final ON/ON scores now equal Local weighted Q (after the existing nonnegative clipping), including Local-pass/Plane-fail cases. The fixed-fixture ACCEPT/REJECT labels above remain the same at threshold 0.4; their ON primary E values now match Local instead of Plane. Legacy numerical tests explicitly select Density OFF so user-tuned defaults do not change their expected arithmetic.

Decision-update validation: all 107 related tests passed. Full suite ran 289 tests with 3 failures and 15 errors already present before this update. The three other option combinations matched all original numerical fields except elapsed time. A headless Local-decision debug preview was inspected; actual GUI/sensor paths were not run.
