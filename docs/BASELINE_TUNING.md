# Dataset baseline parameter adjustment — 2026-11-09

`run_baselines.py` now defaults to `--preset dataset`. `--preset author`
reproduces the preceding demo parameters. All explicit numeric CLI arguments
override their preset. The original Sommer C++ implementation and Ahmed
classification equations/stage order are unchanged. No output snapping,
candidate merging, GT filtering or new NMS was added.

## Selected values

| Parameter | Author | Dataset |
|---|---:|---:|
| Sommer normal radius, mm | 20 | 10 |
| Sommer minimum pair distance, mm | 100 | 5 |
| Sommer maximum pair distance, mm | 1000 | 100 |
| Sommer minimum votes | 5 | 3 |
| Sommer refinement stride | 10 | 5 |
| Sommer input voxel sampling, mm | 0 (off) | 0 (off) |
| Ahmed initial/edge voxel, mm | 10 | 5 |
| Ahmed analysis-cloud voxel, mm | 5 | 2.5 |
| Ahmed edge neighborhood K | 400 | 100 |
| Ahmed edge threshold multiplier | 4.5 | 2 |
| Ahmed thinning neighborhood K | 150 | 30 |
| Ahmed corner patch K | 20 | 20 |
| Ahmed exported L | 10 | 10 |

The original 10-NN PCA, 5 mm patch-span condition, 50–150 degree strict angle
gates and three-cluster size OR remain unchanged. Ahmed's initial/edge grids
share `voxel_mm`; the analysis grid is half that value, preserving the author
10/10/5 mm relationship. The direct Python library/standalone port defaults
remain the author config. Only the dataset runner selects the dataset preset.
The MATLAB backend retains author numerical settings and rejects Python-only
parameter overrides rather than ignoring them.

## Protocol

- ROI input only, seed 1, no tuning of seeds or individual scene parameters.
- Selection scenes: `0001, 0013, 0022, 0030, 0035` (28 in-ROI GTs).
- Separate confirmation scenes: `0004, 0016, 0026, 0033, 0039` (28 GTs).
- Twelve predefined Sommer configurations and twelve Ahmed configurations.
  Ahmed evaluated L=2,6,10 for candidate configs; the author comparison used L=10.
- Selection criterion: mean per-scene F1 at 15 mm. Assignment maximizes the
  number of one-to-one matches within tolerance, then minimizes matched
  distance. Duplicate predictions cannot match the same GT twice. Sensitivity
  is also recorded at 10 and 20 mm.
- Sommer explored normal radii 10/15/20/30 mm, pair minima 2/5/10/100 mm,
  maxima 60/80/100/120/150/200/1000 mm, votes 3/5, strides 2/5/10 and
  representative input sampling 0/4/5/6/8/10 mm, in the recorded combinations.
  Sampling did not win selection; the default remains off.
- Ahmed explored voxel 3/5/10 mm, edge K 50/100/200/400, threshold 2/3/4.5,
  thinning K 10/30/50/150 and corner K 10/20/30, in recorded combinations.
- Scene ROI boxes with unusually large extents (`0009`, `0024`, `0041`) were
  outside this small selection. Original ROI/GT files were not edited.

## Observed results

These numbers are matches to existing manual labels, not a claim of complete
physical-corner ground truth. Precision below treats every unmatched prediction
as a false positive. Unlabelled physical corners may therefore be penalized.

| Split / method | Preset | Matched / GT | Predictions | Label precision | Label recall | Mean scene F1 |
|---|---|---:|---:|---:|---:|---:|
| Selection / Sommer | author | 0 / 28 | 1 | 0% | 0% | 0% |
| Selection / Sommer | dataset | 1 / 28 | 1 | 100% | 3.57% | 8.00% |
| Confirmation / Sommer | author | 0 / 28 | 0 | 0% | 0% | 0% |
| Confirmation / Sommer | dataset | 0 / 28 | 0 | 0% | 0% | 0% |
| Selection / Ahmed | author | 1 / 28 | 146 | 0.68% | 3.57% | 2.22% |
| Selection / Ahmed | dataset | 6 / 28 | 383 | 1.57% | 21.43% | 5.55% |
| Confirmation / Ahmed | author | 0 / 28 | 283 | 0% | 0% | 0% |
| Confirmation / Ahmed | dataset | 8 / 28 | 631 | 1.27% | 28.57% | 3.93% |

At 10/15/20 mm, Ahmed confirmation matches are respectively 7/8/8 for the
dataset setting and 0/0/0 for the author setting. Sommer confirmation matches
remain 0/0/0. The confirmation set was not used to reselect parameters.

Ahmed improves label recall here, but its precision remains very low. Sommer's
selection-scene gain does not generalize in this check. It remains an
orthogonal-three-plane detector, and no claim is made that visible two-edge
or two-plane corners can be recovered by these parameter changes.
On confirmation scenes 0004/0016/0026/0039 no final graph triangle existed;
0033 had three triangles but no accepted refined corner. Increasing only a
refinement parameter cannot address all those failures.

## Reproduction artifacts and limits

Artifacts live in `corner_dataset_tool/baseline_results/tuning_20261109/`:
`protocol.json`, `results.json`, `tune_summary.json`, `check_summary.json`,
`selected.json`, the executable experiment snapshot `reproduce.py`, input/GT
snapshots, per-config predictions, logs and reusable normal inputs. Matching
self-checks cover one-to-one assignment, a boundary match, an out-of-range
prediction and empty predictions. All stored metrics were recomputed from the
saved predictions against the input snapshots before writing this report.

The confirmation scenes can share capture objects with selection scenes; this
is not an independent-object test or a paper benchmark. Full-scene false
positive rates were not evaluated because labels may omit the background.
The parameter search is small and does not establish global optima. MATLAB
port parity and live camera paths were not tested. The dataset is unchanged.

## Scene 0002 ROI tuning — 2026-11-10 (in-sample)

The user requested one Sommer three-plane corner and four Ahmed corners near
this scene's four annotations. This is deliberately **scene-specific tuning**;
scene_0002 is the selection data, not a held-out result. Existing `dataset`
and `author` presets remain unchanged. Explicit `--preset scene2` selects:

| Parameter | Sommer | Ahmed Python |
|---|---|---|
| Input | saved scene_0002 AABB ROI | same ROI |
| Normal support / pair limits | 10 mm / 5–100 mm | — |
| Votes / refinement stride / input voxel | 3 / 5 / disabled | — |
| Initial and edge voxel / analysis voxel | — | 5 / 2.5 mm |
| Edge K / displacement multiplier / thinning K | — | 100 / 2 / 5 |
| Corner patch K / PCA K | — | 27 / 5 |
| Coordinate-span threshold | — | 8 mm |
| Strict pair-angle interval | — | 53.3 < theta < 116 degrees |
| Exported L / seed | — | 1 / 1 |

PCA K controls both edge direction and cloud normal estimation. Coordinate span
and angle tests retain the author's equations, clustering and strict boundaries;
only constants become configurable. All ten L passes run in their original
order with one RNG stream. There is no prediction-count cap, GT snapping,
postprocessing or GT use in the computation. The unmodified Sommer C++ sources
and the Python translation of the Ahmed author source remain the algorithms.
The original MATLAB backend keeps its fixed constants and rejects `scene2`.

| Output | Predictions | Matched GT | Error to GT, mm |
|---|---:|---|---|
| Sommer | 1 | #1 | 8.802877 |
| Ahmed | 4 | #1 / #2 / #3 / #4 | 5.495276 / 3.882883 / 7.277608 / 3.109654 |

Distances use unsnapped predicted coordinates and one-to-one Euclidean matching
within 15 mm. Sommer's other three GT labels are intentionally not claimed as
detections. Ahmed's four predictions also all match within 10 mm.

The prior dataset Sommer settings already satisfy this scene's requested
count, so no further Sommer numerical change was needed. The initial Ahmed
sweep stored 432 configurations × ten L levels in `ahmed.json`. Small edge
thinning neighborhoods and PCA K=5 recovered all four vertices, initially with
many extra predictions. Subsequent sweeps varied corner K, PCA K, coordinate
span and angle limits. For each sampled configuration/level, the angle search
checked candidate-set boundaries to minimize extra detections while retaining
one candidate within 15 mm of each GT. This explicitly uses GT during parameter
selection. It selected corner K=27, span=8 mm and L=1; angle constants were then
rounded to 53.3/116 and verified by complete reruns. A 0.1-degree grid confirmed
four matches for lower bounds 53.1–53.5 and upper bounds 113.5–117.9 in the tested
rectangle. A lower bound of 54 loses GT #4: this result is sensitive and must
not be presented as general benchmark performance. No other-scene or full-scene
validation of this new preset was performed.

Re-run with side-by-side GT / Sommer / Ahmed:

```bash
corner_dataset_tool/.venv/bin/python corner_dataset_tool/run_baselines.py \
  corner_dataset_tool/dataset/scene_0002 --mode roi \
  --method orthogonal ahmed --preset scene2 --visualize
```

Reopen the verified saved run without recomputing:

```bash
corner_dataset_tool/.venv/bin/python corner_dataset_tool/view_baselines.py \
  corner_dataset_tool/baseline_results/scene2_tuned_20261110 \
  --method orthogonal ahmed
```

Results, resolved commands/options, all L masks, source hash and GT distances
are under `corner_dataset_tool/baseline_results/scene2_tuned_20261110/`.
The independent `.venv` rerun is `scene2_tuned_20261110_venv/`; predictions agree
with system Python within 1e-9 mm. Earlier search outputs are in
`scene2_tuning_20261109/`; the angle/neighbor sweep scripts and logs are copied
into its `search_20261110/` subdirectory. Some exploratory sweeps were stopped
once an exact four-match configuration was found; no exhaustive optimum is
claimed. Native Open3D comparison window opening/rendering/closing was run on
the saved results. No visualization algorithm changed.

Validation: 39 focused baseline/port/preset/visualization tests pass; full
`python3 -m unittest discover -s tests -v` runs 326 tests with the same 20 failures
and 19 errors as the 323-test pre-change run (no added failing cases). Detector
CLI help, baseline CLI help and imports pass. MATLAB parity and live camera
paths were not tested. Raw dataset/annotations were not modified.
