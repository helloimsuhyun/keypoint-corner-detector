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
validator_pipeline (validator experiment) -> query-local raw crop/voxel + guided_ransac
validator_pipeline (guided geometry) -> validator_plane_support + validator_edge_graph + validator_corner_graph
runner -> total -> dense_heatmap + validator_pipeline.run_validator_candidates
plane_edge_graph_pipeline -> corner_2d + boundary_support + line_refinement_2d
visualization.manager -> visualization.plots + visualization.view3d
```

Numerical modules do not import visualization or GUI modules. All detector and
validator screens are opened through `visualization.manager`.

## Local vertex hypotheses (ball / extremity256)

`extremity256` calls the existing dense CED/bilateral/PCA/threshold/NMS composition
with identical detector settings, then supplies those exact NMS indices to
`extremity.refine_ced_candidates`. Detector scores/NMS are shared; CLI `detector` now stops at NMS, while explicit library validation remains available. This new stage is not a replacement candidate
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

## Whole-scene total mode (`--mode total`)

`total.run_total_mode` composes the unchanged dense detector/NMS and the existing
guided batch entry point. Every NMS scene ID calls `run_validator_pipeline` with
the same guided settings and an unchanged seed coordinate. Only ACCEPT results
contribute their existing selected Corner and `corner_edges`; there is at most
one selected Corner per seed. No scene-wide refit, cross-query merge, additional
NMS or altered geometric threshold is introduced. Local Edge/Vertex IDs are
qualified by their Coarse `query_index` in aggregate records.

Reuse one validated raw/Coarse KD-tree context across detector scoring, NMS and
all queries. Core and halo are still cropped/voxelized independently for every
query; no global voxel substitution or RANSAC sampling change is allowed.
`include_debug=False` skips only the core-only Before/After comparison, marks it
uncomputed, and makes guided batch records compact instead of retaining each
query's support clouds and diagnostic graphs. Keep final coordinates, active
intervals, contributor provenance, status, errors and timings. The last local
voxel cloud alone stays in the shared cache. Identity-equal scene arrays avoid
a redundant full equality comparison; shape/finite/index checks remain.
Total preprocessing skips the organized pixel-to-scene mapping, which this
guided path does not use. Dense candidate logs are suppressed; summary/progress
and stage timings remain. These changes remove work without modifying the
selected-query validator's default debug behavior.

Manager owns exactly two result windows: (1) Fusion heatmap using the existing
full-scene valid mask, linear Q99 colour range and detector NMS markers; (2) the
original gray raw scene with accepted final Corners in red, Boundary contributors
in green and shared intersection contributors in cyan. Actually used extension
is yellow dashed and residual links are thin gray. Render contributors only
over retained active intervals. At a 5 mm validator voxel, Edge bars have radius
0.75 mm and Corner markers radius 4 mm (twice the original total-view radius).
Batch meshes by colour instead of one
draw call per Edge/Corner. View creation is in `view3d`; `query_ui` owns R for
recapture and Q/Esc for closing both views. There is no Tk query picker, RGB
auxiliary window, or validator debug window in total mode. `--no-3d-view` keeps
computation and console output. `--no-validator-geometry` is rejected for total.
Mode entry points remain; total shares the guided Corner refinement/ranking
policy described below with the interactive validator.

## Coarse-query guided validator (`--mode validator`)

`validator_query.prepare_validator_queries` reuses the unchanged detector
CED/bilateral/PCA/Fusion/threshold/NMS composition. Only an existing Coarse NMS
scene ID can be selected. `run_validator_pipeline` dispatches
`ValidatorDetectorConfig` directly to raw query-radius cropping, local voxelization,
`guided_ransac.extract_guided_planes`, and geometry composition. The independent
`ValidatorConfig` / `--mode fine` organized AHC/Hough/refiner path is unchanged.

There is no guided Fine scoring/NMS/Top-K, query relocation, fallback or second
click. All `validator_fine_*` settings and `fine_*` result/cache/picker fields are
removed. The original Coarse coordinate centers every subsequent stage.
`validator_ransac_radius_x * validator_ransac_voxel_mm` defines the inclusive raw
core crop; apply the existing downsampler only after this crop. Matching Coarse
and local voxel sizes never causes Coarse-cloud reuse.

For geometry, crop the disjoint raw shell out to
core radius + absolute `roi_padding_mm` (default 10 mm) and voxelize it
independently. The combined native cloud stores core first,
halo second. Only core voxels enter RANSAC; geometry reads both. Separate grids
can have nearby samples at their interface and never mix outside raw points
into core centroids. This intentionally changes sampling from the former
full-scene stage voxelization, preserving the RANSAC and geometry formulas.

`prepare_validator_voxel_context` caches raw/Coarse trees per frame and only the
last query-local voxel cloud. Cache keys include query ID/coordinate, voxel,
core and support radii. `query_index` remains a Coarse scene ID, native
`*_ransac_indices` address result `ransac_points_mm`, and `ransac_scene_indices`
supplies nearest-Coarse adapters. Counts/masks use native points; adapters may be
many-to-one. `roi_raw_point_indices` records the original raw core crop.
Local IDs cannot be reused across queries. The query coordinate need not be a
local voxel centroid. All lengths are mm; validator `*_x` lengths multiply the
RANSAC voxel, while absolute mm settings, counts, ratios and angles do not.

The main view shows (1) unchanged detector Fusion colors/NMS, (2) stored gray
query-local voxel input with radius/voxel/raw→voxel counts, and (3) fitted planes.
Validator performs numerical work for only the selected existing Coarse NMS
query. There is no automatic scene-wide query traversal. Coarse
selection calls the canonical query pipeline; panel 2 is inspection only.
Window 5 shows pair intersections. Window 4 shows padded Hull, ROI clipping,
TLS endpoint provenance and core-only Before / padded After comparison,
replacing the former individual-plane Boundary window. Window 6 is last and contains
four panels for the same selected query: original per-plane/shared edges,
overlap masking, actual extension/vertex-LS diagnostics, and the final selected
Corner with only its contributing Edges on the original raw cloud.
There is no window 7. All three auxiliary windows show only that same query.
All visualization stays manager-owned and reads stored numerical results.

## Guided geometry validation after RANSAC

`--mode validator` now continues from the unchanged fitted planes through
`validator_plane_support.reconstruct_plane_support`,
`validator_edge_graph.build_validator_edge_graph`, and
`validator_corner_graph.reconstruct_validator_corners`. The shared pipeline owns
composition; the existing fine-mode AHC/Hough/refiner path is unchanged.

Support is recollected independently for every fitted model out to
core radius + padding. The sole padding setting is absolute
`validator_roi_padding_mm=10.0`, yielding 85 mm support for the 75 mm core.
The former support-radius floor, voxel padding and contact-margin controls are
removed. The configured strict residual gate is
reused, including points removed by sequential RANSAC. Memberships can overlap.
Disconnected coplanar components are never hulled together: a component with
original core support nearest the Coarse query is selected. Multiple competing
components near the query, missing anchored support, or degenerate hulls remain
UNKNOWN. RANSAC normals, offsets and original inlier arrays are retained.

Each patch owns tangent coordinates, convex hull, selected and recollected
native RANSAC IDs, connectivity diagnostics and area. Finite plane-pair edges
use a general least-squares intersection clipped against both hulls. Boundary
candidates come from consecutive hull segments, with conservative collinear
coalescing. Edge point-distance/count validation is removed; finite candidates
clip each original padded Hull segment against the exact plane/core-circle
before TLS and apply the existing finite-edge minimum length. Both-exterior
segments crossing the core remain; no ROI-circumference edge is created. Original
interior Hull Nodes are ACTIVE; crop-created Nodes remain INACTIVE through TLS
endpoint inheritance and cannot be Corner evidence or extrapolation origins.
Inactive-ended Lines still provide interior geometry. Groups require a shared
Node ID and cannot bridge separated chains. Original endpoint metadata, fitted
coordinates, source Hull Edge IDs and maximum TLS residual remain separate.
Fitted extrema are re-clipped when needed without promoting inactive states.
Final vertices also require core membership. `boundary_debug` stores every stage.
Long unobserved intervals are no longer rejected by edge-support evidence. Duplicate edges need same type/plane IDs,
compatible direction, line distance and positive interval overlap; no missing
interval is filled. Plane-pair adjacency is stored explicitly. After deduplication,
intersection-aligned, nearby overlapping boundary ranges are union-masked in their
original TLS parameter space. `all_edges` retains original endpoints/directions,
masks and inactive records; `edges` contains only active sources. Cut endpoints
are not vertex observations and cannot be extrapolation origins.

Active-edge closest-point candidates use an intersection anchor when present,
with complete 5 mm anchor-span grouping. A single intersection constrains the
vertex LS solution to its finite line; multiple intersections use 3-D multi-line
LS. Boundary–boundary midpoints initialize candidates, while all incident
Boundary Lines participate in the final 3-D LS fit and every group-merge refit.
Same-plane pairs still
require a common ACTIVE endpoint that survives masking on both Lines. Nonadjacent
pairs and absorbed internal Nodes cannot create same-plane Corners. Every refined
position is rechecked against one line-distance threshold, active finite extents,
one absolute-mm endpoint tolerance and the unchanged Coarse query corner gate.
`corner_min_edge_angle_deg` alone gates candidate angles; `vertex_group_span_mm`
alone controls candidate grouping. Duplicate parallel-angle and voxel-scaled
corner endpoint controls were removed, without changing the current effective
30-degree / 5 mm line / 5 mm extension / 5 mm group / 10 mm query gates. Groups merge only
with shared edges and complete member-position compatibility. No point-support
checks are reintroduced. Original planes, TLS lines and hulls stay unchanged.

Only accepted vertices inside the unchanged core and query gate enter final
selection. Prefer shared-intersection contributors, then lower perpendicular
Line RMS, lower total endpoint extension, query distance and stable vertex ID.
Boundary-only vertices are a fallback when no shared-intersection candidate
passes the gates. Per-vertex `intersection_edge_count`, `line_fit_rms_mm` and
`total_extension_mm` expose these criteria. Ranking-only RMS/extension values
are rounded to 1e-8 mm to ignore machine roundoff. This is separate from the
existing finite-segment distance acceptance checks. Defaults are 5 mm Line
distance, 5 mm endpoint tolerance, 5 mm group span and 10 mm query gate, narrowed
from the experimental 30/30/30/50 mm settings. No point-support validation or
new detector/plane threshold is introduced.

Results expose `validation_status` (ACCEPT/REJECT/INCONCLUSIVE/NOT_COMPUTED),
`verified`, `corner_position_mm`, `corner_distance_mm`, `selected_vertex_id`,
`corner_edge_ids`, `corner_edges`, `plane_patches`, graph edges/vertices and
stage timers. `verified=True` only for ACCEPT. `--no-validator-geometry` produces NOT_COMPUTED; a fitted zero-plane result is
INCONCLUSIVE. `planes_computed` and the plane fields remain.
Patch `support_indices` and `support_ransac_indices` are native RANSAC IDs;
`support_point_indices` supplies the nearest-Coarse scene adapter. Edge support
arrays/memberships are empty with `point_support_checked=False`; the retained
quality value 1.0 is neutral. Edge/vertex acceptance uses geometry only and
never substitutes plane membership for measured edge support. Native plane
coordinates/counts are never replaced by adapter IDs.

The manager-owned geometry views are split by stored edge type:

- Window 5: all plane support patches/hulls and only `plane_intersection`
  edges in cyan, following the finite plane-pair overlap construction. It shows
  no per-plane boundary edges, corner vertices or final-selection highlights.
  Stored plane-pair line origins/directions also drive yellow infinite-line
  guides (displayed locally) and translucent per-plane tolerance strips, even
  when no finite edge survives. These guides are not detected edges.
- Window 4: A/B/C plus paired D-before/D-after panels, moved from the former
  window 7. Blue dashed core/orange dashed padding, green original Hull/Nodes,
  blue clipped segments/active endpoints and orange inactive X marks expose
  provenance. C shows TLS Line IDs/errors and absorbed Nodes. The pipeline
  stores a core-only support/Hull/graph comparison with unchanged models/core
  voxels, never a second RANSAC; its extra work is timed as `roi_debug_before`
  and never affects the selected Corner. D's two cameras synchronize. S exports
  A/B/C and paired D PNGs under `outputs/validator_roi/query_<id>_<timestamp>/`.
- Window 6: four process panels in a 2-by-2 layout for one Coarse query. Panel 6-1 shows original
  unmasked boundaries in plane RGB colours, shared intersections in cyan and
  muted hull rims. Panel 6-2 shows retained intervals in gray/cyan and removed
  boundary intervals as magenta dashes, including fully deactivated sources.
  A blue translucent cylinder over each finite shared edge shows the stored
  overlap line-distance radius; its caption also reports angle/minimum-overlap gates.
  Panel 6-3 shows active geometry, yellow dashes only for extension actually
  needed by each stored refined vertex (not the full endpoint allowance),
  white pair candidates and candidate-to-LS shifts, final green accepted/red
  gated-out vertex markers, and the selected corner's green incident edges.
  Thin gray residual links remain separate from extension along each edge axis.
  Wire-halo markers keep short extensions visible. Panels 6-1/2/3 omit ROI
  spheres and background support clouds. Panel 6-4 contains only the selected
  Corner/incident active Edges and actual extension over the original unvoxelized
  gray point cloud. Boundary contributors and the Corner are green; contributing
  shared intersection Edges are cyan. Captions list each type's IDs/counts and
  explicitly identify a Boundary-only selected Corner when there is no shared
  contributor. The display does not add unrelated shared Edges or change Corner
  selection. Manager passes raw XYZ to the display-only worker once;
  query changes replace just the final overlays. All four cameras start with
  local query framing, and no other query is validated or accumulated.

Window 5 uses muted gray Hull rims and no ROI wire spheres. Window 4 deliberately
shows both ROI spheres for clipping diagnostics. Panels 6-3/6-4 highlight final
contributors as described above.
All stages read the same stored results; no score, support or graph is recomputed.
The original third panel retains only core points, fitted-plane
disks (alpha 0.60), normal arrows, core sphere and query marker; no hulls,
recollected halo points, edges or corner markers are added there. Fitted disks
remain illustrative and separate from the actual hulls in windows 4/5/6.
All windows share the existing display-only worker and manager lifecycle.
Each auxiliary window closes independently; main-window close, recapture and
quit close all of them. Every auxiliary panel retains its camera when redisplaying
the same Coarse query. Only the Coarse overview picks queries.
All defaults and optional CLI flags live in the validator section of settings.
Detailed equations, default rationale and limitations are in PIPELINE_METHODS.

## Primary fine/detector validator contract

Fine mode opens the frozen, aligned RGB/depth Tk window directly. A click
in that window selects or reselects the query; there is no preliminary OpenCV
query picker whose cancellation can terminate the validator before Tk starts.

There is one shared plane-extraction path. Every fine clicked query and every explicitly validated detector NMS seed calls `validator_pipeline.run_validator_pipeline`. CLI detector mode does not automatically validate NMS seeds. The function always:

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

The CLI whole-cloud detector stops after dense score thresholding and deterministic 3-D NMS. Both CLI configuration and default `DetectorConfig` use `validator=None`; candidates remain unverified. Dense scores, thresholds, NMS and scene IDs are unchanged. Library callers can explicitly supply `ValidatorConfig` to invoke the same validator pipeline. That optional batch output preserves each query result and aggregates final refined finite edges and corners; it does not maintain a second plane extractor.

`--mode detector --only` additionally limits display to the NMS panel after that same CED/PCA/fusion/NMS composition. `DetectorConfig.only` suppresses validator calls, and configuration
resolves `validator=None`. No plane, edge or corner stages run. Visualization
still goes through `manager.show_dense_heatmaps`, selecting only the existing
NMS Fusion heatmap panel in `view3d.show_dense_score_heatmaps`. Its linear Q99
colors, invalid-point gray, original scene-indexed markers and M toggle are
unchanged. Other score panels and the registered RGB auxiliary window are
omitted. `--no-3d-view` also suppresses that one window. `--only` is rejected
for other modes; omitting it retains all dense diagnostic panels, without automatic validation.

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
- `validator_plane_support.py`: guided-mode support recollection, connected patches and hulls.
- `validator_edge_graph.py`: guided-mode finite intersections, observed boundary edges and adjacency.
- `validator_corner_graph.py`: supported finite-edge vertices, centre gate and validation status.
- `plane_projection.py`: small shared 3-D visualization geometry helpers.
- `boundary_support.py`: continuous Gaussian boundary-proximity weights.
- `corner_2d.py`: continuous 2-D boundary evidence.
- `plane_edge_graph_pipeline.py`: Hough, finite lines, 3-D restoration, merging,
  and plane adjacency.
- `line_refinement_2d.py`: continuous per-seed $(rho, theta)$ optimization,
  support-profile finite segments, score filtering, and final 2-D segment NMS.
- `refiner.py`: graph-local robust plane/edge refinement and corners.
- `visualization/manager.py`: all visualization orchestration.
