# Architecture and data contracts

The canonical numerical description and revision ledger live in
[`PIPELINE_METHODS.md`](PIPELINE_METHODS.md). This file records active module
boundaries and indexing contracts.

## Dependency direction

```text
pointcloud_inputs -> PointCloudFrame / QuerySelection
runner -> preprocessing + query_debug/dense_heatmap + visualization.query_ui
runner / query_ui -> ball_query -> extremity
extremity -> extremity_voting (optional quantile soft votes)
preprocessing -> organized RGB-D / voxel scene-index adapter
query_debug -> validator_pipeline + visualization.manager
dense_heatmap -> edge_suppression + candidate_extraction + validator_pipeline
validator_pipeline -> cornervins_plane_extraction + plane_edge_graph_pipeline + refiner
validator_pipeline (validator experiment) -> validator_fine + guided_ransac
plane_edge_graph_pipeline -> corner_2d + boundary_support + line_refinement_2d
visualization.manager -> visualization.plots + visualization.view3d
```

Numerical modules do not import visualization or GUI modules. All detector and
validator screens are opened through `visualization.manager`.

## Local vertex hypotheses (ball / extremity256)

`extremity256` calls the existing dense CED/bilateral/PCA/threshold/NMS composition
with identical detector settings, then supplies those exact NMS indices to
`extremity.refine_ced_candidates`. The existing `detector` mode and its validator
composition are unchanged. This new stage is not a replacement candidate
generator. It creates no FPS queries, moves no input query and aggregates no
votes across queries. Each query can produce zero, one or many hypotheses.

`ball` reuses the frozen RGB-D click UI. In both new modes a click inspects
the nearest **existing CED query**, never creating an additional query or moving
the existing one. The frame's independent local results are computed once and
cached for inspection. Both modes reuse the existing data
loaders and default 10 mm voxel preprocessing. Voting operates on original raw
XYZ, not voxel centroids.

The local module owns its ROI, optional Euclidean object components, 3-D
direction sweeps and hypothesis peak suppression. It calls no plane clustering,
projection, Hough, or plane-edge refinement. Raw KD-tree indices are cached once
per frame/batch; object components are computed once per query's maximum padded
ROI and reused across scales. Every component receives its own sweep, including
components that do not contain the query. Connectivity is not instance
segmentation: touching objects can merge and sparse surfaces can fragment.

All tunable defaults live in the `ball_*` block of `settings.SettingsDefaults`.
`settings.ExtremityConfig` resolves/validates those settings and is re-exported
by `extremity` to preserve existing imports. Both the CLI and default library
calls share these values; numerical `extremity.py` has no separate parameter
defaults. Ball `*_x` lengths multiply `voxel_3d_mm` when settings are resolved.
At the default 10 mm voxel, radii `(3.0,)`, padding `1.0`, component radius
`0.45`, and peak NMS `0.5` reproduce the original 30/10/4.5/5 mm lengths.
`ExtremityConfig` and all downstream computations still use resolved mm fields;
explicit library `ExtremityConfig(radii_mm=...)` calls retain their mm contract.

The output explicitly distinguishes index spaces:

- `query_index`, candidate `point_index`, `candidate_indices`, and deduplicated
  `validator_seed_indices` index the voxel scene.
- `raw_point_index`, `candidate_raw_indices`, and `roi_raw_indices` index the
  original input cloud. A hypothesis position is an actual raw point.
- `candidate_local_indices` index the stored `roi_points_mm` and its vote arrays.
- Scene IDs are nearest-voxel adapters; `scene_mapping_distance_mm` records the
  discrepancy. Different raw hypotheses may map to the same voxel sample.

The existing `run_validator_candidates` can consume `validator_seed_indices`
with the original voxel scene and organized context. Hypotheses remain explicitly
unverified until that separate stage runs; no extra validation pipeline exists.

All live and exported ball views pass through `visualization.manager`.
At frame startup, an additional non-blocking overview shows all original CED
NMS queries in red and every stored Ball hypothesis in magenta over the original
voxel-scene Fusion heatmap, using the same valid mask, linear Q99 color range
and colors as the detector's NMS panel. `ball_query` retains the existing score
result for this display; no scores are recomputed. Query markers retain voxel
scene coordinates and Ball markers retain actual raw point coordinates.
Only identical raw point IDs share a display marker; no cross-query NMS or
coordinate averaging is applied. The overview stays open while clicks replace
the three local views, and is closed on recapture/quit. `--no-3d-view` skips it.
`view3d` opens three live ball windows in order: local object components,
the full raw-vote heatmap before filtering/NMS, and final NMS hypotheses on a
neutral gray ROI. All retain the original raw scene, unchanged red query and
core/padded balls; only the third adds candidate markers. Raw display colors
use neutral gray for zero and linear colors from zero to the raw maximum,
without changing stored votes or normalized scores. Titles/terminal report
the numeric range and zero fraction, including fractional soft vote mass.
`manager` owns the three-window lifecycle in
`extremity_views`; closing one leaves the other two active. `query_ui` keeps
the click/recapture/quit flow, without the former C/H switch or automatic ball
summary popup. `plots` retains stored score/offset figures for headless use;
synthetic exports keep their normalized-score diagnostics and GT markers.

Selecting a Ball query also opens a Tk-embedded, rotatable 3-D raw-ROI point
picker and unit-direction sphere. Clicking an observed ROI point, including
rejected or padding points, colors directions by their actual contribution
to that point's stored voting total. The default heatmap uses a linear
zero-to-point-maximum color range with a numeric colorbar; zero is gray.
The existing binary Q99 directions remain selectable from the display menu.
`extremity.inspect_point_directions` reads the stored ROI/components/config,
then computes only the selected point's directional diagnostics on demand.
It preserves the configured Fibonacci directions, precision and padded
same-component support; `extremity_voting.point_direction_diagnostics`
reconstructs the active hard/dense/partition/sparse directional mass using
the same per-direction normalization and sparse support cutoff, followed by
core gating. Directional mass sums to the selected scale's stored vote total
up to floating-point accumulation. Q99 membership is retained separately,
with inclusive ties. Switching display modes reuses diagnostic arrays and
introduces no FP filter or score change.
`view3d` owns the 3-D figure and screen-to-original-ROI-index mapping;
`manager` composes diagnostics/display; `query_ui` owns Tk canvas events,
scale selection and cleanup on query change, recapture or quit.

`extremity_voting.py` owns optional quantile soft-weight calculation. It returns
one direction-normalized vote map per component/scale and stage timers;
`extremity.py` retains ROI construction, component composition, core gating,
scene/raw ID mapping and hypothesis NMS. Hard voting is unchanged/default.
The dense/partition/sparse backends add no planes, projections to 2-D, smoothing,
coordinate averages, or query generation. `soft_voting_benchmark.py` reuses
the existing CED functions and synthetic geometry to compare modes on identical
inputs; it routes all research plots through `visualization.manager`.

## Coarse-to-fine guided validator (`--mode validator`)

`validator_query.prepare_validator_queries` keeps the existing Coarse detector
composition and numerical settings: CED → bilateral CED → weighted PCA/Fusion →
radius-local-maximum NMS. Its cached Fusion heatmap remains the overview.
No query-local work runs before a Coarse query is selected.

`validator_pipeline.run_validator_pipeline` dispatches `ValidatorDetectorConfig`
to `validator_fine.find_fine_queries`, followed by
`guided_ransac.extract_guided_planes`. The existing fine/detector composition
still receives `ValidatorConfig` and is unchanged. The former anchored extractor
and anchored-only configuration/CLI fields are removed.

`validator_fine.py` owns Fine-voxel evaluation/support indexing and composition,
calling the unchanged dense CED/bilateral/PCA/Fusion scorer on a complete support
halo. Its radius is the Fine search radius plus the larger PCA/bilateral radius
plus the Fine CED radius. Only scores inside the search ball enter threshold/NMS;
outer support ensures the same neighborhoods as a full Fine-voxel-cloud evaluation.
Fine CED, minimum neighbors, PCA radius/saturation, bilateral radius/spatial
sigma/range sigma, Fusion threshold mode/value and NMS method/radius have their
own `validator_fine_*` settings. Fine uses none of the Coarse scoring/threshold
settings. `fine_nms_method` chooses Greedy (default) or the existing local-maximum
extractor; neither changes the Coarse NMS implementation.
Fine threshold modes are fixed, global-mean and peak-relative (default).
Peak-relative uses max(the configured absolute threshold, a configured ratio
times the maximum finite valid Fusion score inside the Fine evaluation ball).
The ratio defaults to 0.5; the absolute floor remains 0.10. Eligibility stays
strictly above the resolved threshold; empty or zero-score support adds no
candidates. This changes only Fine eligibility, not scores or NMS geometry.

The nearest Fine NMS voxel point is highlighted by default. In the interactive
3-D flow, Coarse selection stops after Fine scoring/NMS, and a Fine click starts
RANSAC. The three panels show Coarse heatmap/NMS, Fine heatmap/NMS, and plane
fitting in order. No candidates means an explicit unchanged-Coarse fallback
that still runs RANSAC. Selecting another Fine point reuses Fine scores and
reruns only RANSAC. Headless and default library calls keep automatic nearest-Fine
selection and the full pipeline. `validator_query`
caches stage voxel clouds/KD-trees per frame and the current Coarse query's Fine result.

`guided_ransac.py` owns the independent Fine-centred RANSAC-voxel ROI and Sequential
RANSAC. Only the first of three distinct samples has Gaussian distance-biased
probability; the other two are uniform. Planes are not constrained through the
query. Rank by inlier count, then RMS, and hard-remove accepted support. Minimum
support is fixed using the original ROI count for all extraction rounds. There
are no weights in the support score, normal-angle gates, refits, reassignment or
plane/edge/corner refinement. Full formulas/defaults are in PIPELINE_METHODS.

`settings.py` exposes independent `validator_fine_voxel_mm` and
`validator_ransac_voxel_mm` (current defaults 5 and 10 mm). Fine `*_x` lengths multiply the
Fine voxel; RANSAC `*_x` lengths multiply the RANSAC voxel. Defaults preserve
45/17.5/15 mm Fine search/CED/NMS and 60/5/10 mm RANSAC radius/threshold/sigma.
Fine PCA radius, bilateral radius and spatial sigma default to 22.5/15/7.5 mm.
Each has its
own Fine-voxel multiplier. The normalized-CED range sigma, PCA ratio saturation
and Fusion threshold/peak-relative ratio are dimensionless; minimum neighbors is an integer.
Counts, ratios and seed do not scale. Configs passed to numerical functions store
resolved mm fields. The CLI length flags now end in `-x`.

`prepare_validator_voxel_context` builds each stage cloud from the full input
using the existing voxel downsampler. Equal voxel sizes share a cloud and tree;
a size equal to the Coarse voxel reuses the supplied scene. No Fine/RANSAC score,
fit or support count uses the pre-voxel raw cloud. Source samples are used only
to construct new voxel clouds. `validator_query` caches this context per frame.

The public `query_index` remains a Coarse scene ID. Fine selection uses
`fine_query_index`/`candidate_fine_indices` in the Fine voxel cloud; RANSAC support
uses `roi_ransac_indices`/`inlier_ransac_indices` in the RANSAC voxel cloud. Plane
masks index the stored RANSAC ROI. A nearest-Coarse adapter provides
`fine_query_scene_index`, Fine `*_scene_indices`, and the public
`roi_point_indices`/`inlier_point_indices` in the original supplied scene space.
Adapters can be many-to-one; positions and counts stay native to each stage.
There is no implicit interchange of these IDs or change to other mode contracts.

The existing manager-owned Open3D worker/lifecycle now has three labeled native
panels: (1) cached Coarse Fusion heatmap and orange NMS markers, (2) the stored
Fine evaluation-point Fusion heatmap and yellow NMS markers, and (3) gray RANSAC-voxel
ROI/outliers and RGB planes/inliers/normals after a Fine click. Fine colors use
the same valid mask, linear valid-score Q99 range and shared color function as
Coarse. The nearest/selected Fine query is red in panel 2; panel 3 also retains
the red Fine sphere, orange Coarse marker and RANSAC wire sphere.

`run_validator_pipeline(stop_after_fine=True)` returns `planes_computed=False`
when Fine candidates exist. `validator_query` uses this stop only for interactive
Coarse selection; a Fine click, default library calls and headless calls run the
full pipeline. A new Coarse selection clears previous fits; Fine reselection
preserves Coarse/Fine heatmap geometry and their cameras, replacing only the red
Fine marker and plane result. A computed zero-plane result remains distinct from
an uncomputed fit. Fine picking occurs only in panel 2 and returns (Coarse scene
ID, Fine voxel ID), so stale clicks can be ignored. Disk geometry remains the
plane/ROI-sphere intersection; material/visibility and all numeric settings are
unchanged. R/Q/Esc work in all three native panels.

Run `python3 -m corner_detector --mode validator --source stl --input
/path/to/model.stl` on one shell line. Click an orange Coarse query in panel 1,
then a yellow Fine NMS query in panel 2 to fit planes in panel 3. Scroll to zoom
and drag to rotate. `--no-3d-view` retains automatic nearest-Fine selection for
RGB/depth Coarse clicks.

## Primary fine/detector validator contract

Fine mode opens the frozen, aligned RGB/depth Tk window directly. A click
in that window selects or reselects the query; there is no preliminary OpenCV
query picker whose cancellation can terminate the validator before Tk starts.

There is one fine/detector plane-extraction path. Every fine clicked query and every detector NMS
seed calls `validator_pipeline.run_validator_pipeline`. The function always:

1. limits the organized RGB-D pixels to the exact inclusive Euclidean query ball;
2. runs CornerVINS-style organized AHC segmentation on those pixels;
3. performs per-region deterministic RANSAC, full-resolution inlier
   reclassification, and least-squares refitting;
4. repeatedly merges compatible patches and builds final convex hulls;
5. optionally continues through padded 2-D boundary/Hough, 3-D finite-edge
   reconstruction, plane adjacency, and graph-local refinement.

The former FPS-reference/SAH plane-clustering backend and its backend selector
do not exist in the active package. Plane extraction is automatic even when
the plane-edge graph is disabled; the standard coarse-AHC, refined-AHC,
RANSAC-plane, and final plane/convex-hull views are prepared for every
successful query.

## Index and organization contract

- `PointCloudFrame.points_mm` is the original camera-frame cloud in millimetres.
- The detector and validator operate on a 3-D-voxelized `(M,3)` scene array.
- `query_index`, plane support IDs, projection IDs, boundary support IDs, edge
  support IDs, and corner provenance all index that same voxel scene array.
- A valid RGB-D pixel retains its raw camera-space point and is mapped to its
  nearest voxel-scene index only for downstream indexing.
- AHC adjacency comes from the original image grid. There is no KNN or
  unorganized-cloud fallback.
- CornerVINS segmentation and plane fitting use only the exact query core.
  The later boundary projection halo never changes that fitted support. The
  active default is a 50 mm halo, so a 100 mm query has observed projection
  context out to 150 mm.
- The mutable organized context caches extraction results by query index,
  CornerVINS configuration, and extraction radius.

## Detector-to-validator composition

By default the whole-cloud detector stops creating hypotheses after dense score
thresholding and deterministic 3-D NMS. Every retained scene index enters the
same validator pipeline. Batch output preserves each query result for
diagnostics and aggregates final refined finite edges and corners; it does not
maintain a second plane extractor.

`--mode detector --only` stops execution after that same CED/PCA/fusion/NMS
composition. `DetectorConfig.only` suppresses validator calls, and configuration
resolves `validator=None`. No plane, edge or corner stages run. Visualization
still goes through `manager.show_dense_heatmaps`, selecting only the existing
NMS Fusion heatmap panel in `view3d.show_dense_score_heatmaps`. Its linear Q99
colors, invalid-point gray, original scene-indexed markers and M toggle are
unchanged. Other score panels and the registered RGB auxiliary window are
omitted. `--no-3d-view` also suppresses that one window. `--only` is rejected
for other modes; omitting it preserves the standard detector composition.

Example: `python3 -m corner_detector --mode detector --only --source stl --input
/path/to/model.stl` (one shell line).

## Visualization contract

The four standard plane views all preserve the complete scene:

1. coarse AHC regions after Feng Algorithm 3;
2. refined AHC regions after Feng Algorithm 4;
3. RANSAC-accepted, least-squares-refitted plane memberships and normals;
4. final merged planes and their convex hulls.

Outside-ROI points are dark, ROI points are light gray, the exact 3-D ROI is a
yellow wire sphere, and the query is red. Plane or region colors overwrite the
gray ROI color. AHC block size is in image pixels, not millimetres; its metric
footprint varies with depth.

## Ownership

- `cornervins_plane_extraction.py`: organized AHC, RANSAC/refit, patch merge,
  convex hull, and downstream schema adapter.
- `validator_pipeline.py`: primary stage composition and batch candidate entry.
- `plane_projection.py`: small shared 3-D visualization geometry helpers.
- `boundary_support.py`: continuous Gaussian boundary-proximity weights.
- `corner_2d.py`: continuous 2-D boundary evidence.
- `plane_edge_graph_pipeline.py`: Hough, finite lines, 3-D restoration, merging,
  and plane adjacency.
- `line_refinement_2d.py`: continuous per-seed $(rho, theta)$ optimization,
  support-profile finite segments, score filtering, and final 2-D segment NMS.
- `refiner.py`: graph-local robust plane/edge refinement and corners.
- `visualization/manager.py`: all visualization orchestration.
