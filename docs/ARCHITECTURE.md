# Architecture and data contracts

The canonical numerical description and revision ledger live in
[`PIPELINE_METHODS.md`](PIPELINE_METHODS.md). This file records active module
boundaries and indexing contracts.

## Dependency direction

```text
pointcloud_inputs -> PointCloudFrame / QuerySelection
runner -> preprocessing + query_debug/dense_heatmap + visualization.query_ui
runner (dataset) -> dataset_input -> corner_dataset_tool.dataset_io + dense_heatmap
runner / query_ui -> ball_query -> extremity
extremity -> extremity_voting (optional quantile soft votes)
preprocessing -> organized RGB-D / voxel scene-index adapter
query_debug -> validator_pipeline + visualization.manager
dense_heatmap -> edge_suppression + candidate_extraction + optional q10_validator + validator_pipeline
q10_validator / refiner -> connectivity (unchanged Euclidean radius-graph CC)
q10_validator -> refiner + spherical_antipodal (unit-sphere RANSAC/per-plane angular support)
validator_pipeline -> cornervins_plane_extraction + plane_edge_graph_pipeline + refiner
validator_pipeline (validator experiment) -> query-local raw crop/voxel + guided_ransac
validator_pipeline (guided geometry) -> validator_plane_support + validator_edge_graph + validator_corner_graph
runner -> total -> dense_heatmap + validator_pipeline.run_validator_candidates
plane_edge_graph_pipeline -> corner_2d + boundary_support + line_refinement_2d
visualization.manager -> visualization.plots + visualization.view3d
```

Numerical modules do not import visualization or GUI modules. All detector and
validator screens are opened through `visualization.manager`.

## Detector metric radii

Detector `centroid_radius_mm` is the physical CED support radius, independent
of the scene voxel. Its user-tuned initial value lives in `settings.py`.
PCA follows CED through `pca_radius_factor` (default 1.5), so a 40 mm CED
gives 60 mm PCA support. CLI exposes
`--centroid-radius-mm` and `--pca-radius-factor`; the former independent
`--pca-radius-mm` option is removed. Resolved `DetectorConfig` still carries
metric `centroid_radius_mm` / `pca_radius_mm` fields for numerical consumers.

`pca_ced_cutoff` / `--pca-ced-cutoff` is a finite dimensionless [0,1] cutoff
on PCA SUPPORT samples' original CED. Default 0.0 preserves positive-weight
support: require CED-valid, CED>0 and CED>=cutoff inside the inclusive PCA ball.
Equality is included, but zero weights remain excluded. This never thresholds
the query's own CED or changes the original/bilateral CED fields. Geometric
`pca_neighbor_count` is unchanged; weighted support counts, covariance, ratios,
validity and downstream Fusion/NMS can change with a positive cutoff. Existing
minimum-support/eigenvalue gates remain; no new effective-sample-size gate is
added. Results/logs record the resolved cutoff. Shared Coarse and comparison
callers propagate the same setting in all bilateral modes.

Bilateral space uses explicit CED multipliers: `bilateral_radius_factor=1.0`
and `bilateral_sigma_spatial_factor=0.5`. Resolve both to mm in the detector
resolver; remove the former optional mm settings and their `None` branches.
The CLI likewise uses `--bilateral-radius-factor` and
`--bilateral-sigma-spatial-factor`. `bilateral_sigma_ced=0.05` remains a
positive dimensionless tolerance for the selected normalized CED/Fusion
score differences, not a spatial scale; the historical field/CLI name remains. Thus changing CED from 80 to 40 mm changes default
PCA/bilateral/spatial sigma from 120/80/40 to 60/40/20 mm. Numerical consumers
and direct library configs keep their existing resolved-mm interfaces.
NMS also follows CED through the user-tuned `nms_radius_factor` in `settings.py`:
`nms_radius_mm = centroid_radius_mm * nms_radius_factor`. For example, a 40 mm
CED and factor 0.6 give NMS 24 mm. CLI uses
`--nms-radius-factor`; the absolute-mm option and `None` fallback are removed.
The numerical Local Maximum NMS still consumes a resolved positive mm radius.
`min_neighbors` stays a count. Changing voxel size
changes sampling, not these configured radii; scores can still change when the
sampled cloud changes. Scoring equations and default numerical behaviour remain.
`SettingsDefaults.bilateral_mode="ced"` preserves the existing CED filter.
`"off"` skips filtering. `"fusion"` first computes original CED times the
configured PCA ratio and then filters the completed Fusion field once, using
only fusion-valid queries/support. The same radius/spatial/range sigmas apply;
range differences are measured in the selected field. Settings, resolved config,
`--bilateral-mode {ced,fusion,off}`, shared scoring and comparison callers all
carry this mode; the former boolean is removed. Original CED/PCA, validity,
coordinates and scene IDs stay unchanged, while Fusion/NMS can change by mode.
`fusion_score_raw` stores the product before optional Fusion-stage filtering;
`fusion_score` is always the final score used by thresholding/NMS. In fusion/off
modes `bilateral_ced_score` copies original CED; its historical
`bilateral_valid_mask` keeps CED validity in every mode. Neighbor counts/weight
sums describe the selected filter's support and are zero for off. CED filtering
keeps its original bounded shared lookup; Fusion filtering needs a later pass
once every product score exists. The shared lookup/timer attribution remains.
All shared Coarse detector consumers use the same resolver. The separate fine
config retains `centroid_radius_x * voxel_3d_mm` and its fine-only flag. This
change adds no multi-scale loop or new candidate merging.

## Selectable detector Fusion and post-NMS lambda1 gate (2026-11-11)

`SettingsDefaults.fusion_mode` / `--fusion-mode` selects `pca_ratio`
(existing CED × raw/tanh-saturated weighted-PCA ratio) or `local_minimum`
(current user-selected setting).
The latter ports the standalone lab's Min-Max local minimum equation using
Normal-PCA lambda1, the largest eigenvalue of the unweighted centered covariance
over geometric neighbors in the inclusive PCA-radius ball. Compute this field
at every scene voxel with enough geometric neighbors, including voxels whose
own CED is invalid; they can still provide eigenvalue support to another query.
For finite lambda1 samples within the same PCA radius R,
L=(local max-lambda1)/(local max-local min+tau*R²), S=selected CED*L.
`local_minimum_tau` defaults to the lab's 0.005. No Median, ratio factor,
saturation or absolute variance gate enters the new score. It requires valid
query CED and Normal-PCA L; weighted-PCA validity remains a separate diagnostic.
Invalid dense L/S entries use zero plus validity masks. Weighted PCA, original
CED cutoffs, scene IDs and its diagnostics retain their existing definitions.

Both Fusion modes honor `bilateral_mode`: ced filters CED before the selected
product, off uses original CED, fusion filters the selected completed product.
`fusion_score_threshold` remains the pca_ratio setting;
`local_minimum_fusion_threshold=0.2` supplies the other mode's configured fixed
threshold. The resolver places the active value in DetectorConfig. Existing
fixed/global-mean threshold selection and strict score>threshold remain.

`post_nms_lambda1_ratio_max` / `--post-nms-lambda1-ratio-max` is OFF with `None`;
the current user-selected setting is 0.20. When configured in [0,1], apply an inclusive upper bound to the
cached CED-weighted largest eigenvalue divided by R_PCA², only at original
NMS survivors. Reject invalid weighted PCA/nonfinite values. This is distinct
from Normal-PCA lambda1 used by L. Never rerun NMS or replace a rejected peak
with a suppressed neighbor. `nms_candidate_indices/mask` retain raw scene NMS
IDs; `candidate_indices/mask/candidates` contain gate survivors in the original
order. `post_nms_lambda1_gate` records aligned values, pass/invalid masks and
rejected scene IDs. Only retained candidates enter Q10 and later seed handoff.

Dense detector, validator/total/new-validator, ball callers and the standalone
comparison adapter share these settings and core functions. The numerical
modules never import the lab GUI. New mode computes Normal PCA using the shared
bounded neighbor pass, then a bounded local min/max pass; legacy mode avoids
this additional work. Managed six-panel diagnostics replace the ratio-used
panel with cached L for the new mode and show actual Fusion/NMS output without
recomputation. Logs distinguish raw NMS, post-gate retained counts and the
configured mode/tau. Selecting pca_ratio with the post-NMS gate OFF preserves
the original scores and candidate decisions.

## Default detector Q10 relocation and final antipodal Lite

As of 2026-11-11, final bidirectionality uses the standalone lab's cleaned
per-plane MaxGap/span test. `q10_validator._final_evidence` collects nonzero
full-R analysis directions; `spherical_antipodal.segment_origin_spherical_planes`
fits up to three origin-constrained great circles. The existing detector
RANSAC settings, including its user-tuned tolerance, remain unchanged.
`evaluate_plane_max_gap` links circular neighboring azimuths within the configured gap,
keeps the largest group, and trims sparse bins only from its ends. Default
10-degree bins require max(2,ceil(.5*median(nonempty bin counts))) points;
interior bins remain. Compute cleaned span = 360 - maximum circular gap.
Every nonempty cleaned arc is classified by its span, without minimum retained
point-count or retained-fraction gates. Any such plane with span >= the
configured edge-span threshold rejects (equality rejects); otherwise an empty
arc or no fitted planes yields UNKNOWN, and all nonempty passing arcs yield
CORNER. The former q10_gap_min_points / q10_gap_min_retained_fraction settings
and CLI flags are removed. RANSAC minimum support and tail density settings
remain independent and unchanged. The standalone lab retains its own gates.
The PCL-style max-gap>90 flag is diagnostic and uses the pre-trim group.

`settings.py` centralizes `q10_gap_*` settings/CLI and the existing Q10 and
`q10_sphere_*` controls. The N+/N-/B score, axis search, beta caps, side minimum
and their configuration/CLI are removed, including the old threshold alias.
Fixed result metadata `antipodal_mode="sphere_max_gap"` identifies the method;
there is no method selector. Angular grouping is independent of physical mm
plane extraction and optional Q10 spatial CC; no final spatial CC is added.

The detector runtime skips organized RGB-D-to-voxel mapping, which only plane
consumers need. Fine, validator and ball modes retain their organized context.
Dense scoring batches CED queries in groups of 64. After all original CED scores
exist, each valid query shares one max(bilateral radius, PCA radius) lookup and
squared-distance array between CED-stage bilateral filtering and PCA, again in
bounded 64-query batches. Fusion-stage filtering runs only after all product
scores exist, using the same tree in a separate scalar-query pass. Scalar KD-tree neighbor traversal order is preserved with
`return_sorted=False`; strict CED and inclusive bilateral/PCA boundaries remain.
A smaller-radius boundary within 64 float64 epsilons of radius squared uses
the original scalar-radius lookup. No whole-scene neighbor graph is cached.
Shared lookup time is included in the bilateral timer; PCA time is its kernel.

Detector, validator, total and new-validator share the configured Q10 default;
`--no-q10-validator` disables it. Scale/PCA Lite and the separate multiscale
heatmap/Weighted Local Cone experiments remain removed. The `lite_validator`
result alias and inspector names refer to Q10 only. Dense scoring/NMS is retained.

`q10_validator_enabled` remains the whole post-NMS validator master switch.
`q10_refine_enabled=True` / `--[no-]q10-refine` independently controls Q10
relocation AND its `q10_min_q10` gate. The ON path retains existing numerics; the current initial switch follows settings.py. With refine OFF, skip nearest-analysis snapping, Q10 calculation,
search, Q10 CC and the min-Q10 gate; run existing RANSAC/MaxGap at the exact
original NMS XYZ using the same analysis-voxel radius-R support. MaxGap alone
sets ACCEPT/REJECT/INDETERMINATE. Missing analysis samples in the NMS search
ball no longer block this OFF path. Global voxelization and all angular
settings remain unchanged.

OFF records carry `refine_enabled=False`, `q10_value/q10_reject=None`, no refined
analysis ID, and a one-position trace at original XYZ with no analysis/support
IDs, NaN Q10/axis/CC values, zero tested/support counts, actual final ROI count,
`converged=False` and `stop_reason="refine_disabled"`. Final angular evidence
is cached by exact coordinate tuple rather than a missing analysis ID. Effective
Q10 CC is false and Q10 cache/early-reject counts stay zero. Diagnostics show
refine/min gate OFF and omit the Q10 threshold/arrow, while keeping cached
plane colors and MaxGap evidence. Master OFF still skips the entire validator.

With refinement ON, `refiner.Q10PositionRefiner` owns discrete relocation on a globally downsampled
analysis cloud using `q10_voxel_mm`. Search once inside the fixed original-NMS
ball, with the resolved detector NMS radius; never move the search center. At
an uncached candidate, optional spherical query CC uses non-self median NN over
the full radius-R analysis neighborhood and epsilon=median NN*`q10_cc_factor`.
Its graph uses original voxel XYZ in mm and cannot leave R. When Q10 CC is OFF,
use full-R samples and skip NN/graph evaluation; trace NN/epsilon are NaN.
The switch and direction mode retain the current values in settings.py.

Compute ordinary NumPy cosine Q10 on the selected support and direction mode.
No radial/density weighting, CED-axis optimization or antipodal test enters Q10.
Retain at least max(10,ceil(.8*initial support count)) samples and require strict
Q10 improvement >1e-9 over the initial snapped analysis point. Equal scores
prefer points nearer the original NMS center, then analysis ID. More than 3500
fixed-ball candidates is INDETERMINATE rather than silently subsampled. The
trace contains the initial point and at most one replacement; every position
stays within the original NMS ball (+1e-8 mm). With no analysis point in that
ball, preserve the original position as INDETERMINATE and skip final evidence.

Q10/support/axis and spherical CC information, including invalid responses,
are cached by analysis ID within one frame/cloud/R/CC-factor/switch/direction
instance. `support_analysis_indices` gives selected Q10 support per trace step;
`roi_counts` gives full-R counts, while `neighbor_counts` gives selected counts
including the query. `cc_median_nn_mm` and `cc_radii_mm` describe optional Q10 CC.
A support-count gate can defer a quantile without caching it as invalid; a later
seed with a smaller requirement can reuse support and compute Q10 normally.

At convergence, read the cached `trace.q10_values[-1]` before final evidence.
A finite Q10 below `q10_min_q10` immediately REJECTs with only
`q10_below_threshold`; skip final ROI collection, spherical RANSAC and per-plane
MaxGap evaluation. Equality passes. An undefined/non-finite Q10 skips these same
stages and stays INDETERMINATE. Only a finite passing Q10 reaches final angular
evidence, cached once per distinct final analysis ID. Q10 is never recomputed.
A known antipodal failure then rejects; ACCEPT requires both checks to pass.
`antipodal_evaluated` means the final evidence stage was requested or reused
(it can still find insufficient support). It is false on early rejects and
undefined/nonconverged Q10; `antipodal_reject` and `antipodal` stay None there.
`q10_early_reject_count` counts seeds rejected by that first gate, while
`final_evaluation_count` counts distinct passing origins evaluated. An early
reject no longer stores an unevaluated second rejection reason or plane masks.
Q10 relocation, support, cosine threshold arithmetic and all ID spaces remain.
The cleaned-span gate is independent of the Q10 lower bound.

Stored `spherical_planes`/`plane_antipodal` dictionaries contain all models,
overlapping memberships, projected directions/angles and per-plane gap tests.
These established result keys remain, but their old axis/count/B fields are
replaced by spans, gap, retained masks/counts, reliability and tail diagnostics.
The `antipodal` summary selects a rejecting plane first, then a reliable plane,
widest span, greatest kept count and smallest plane ID. Its retained/discarded
masks align to every nonzero `vector_analysis_indices` entry and are false
outside the selected plane. All analysis
IDs address `context.analysis_points_mm`. `geometry_analysis_indices` is the
union of fitted memberships. Retained/component IDs cover the full ROI and
masks are all true; `cc_evaluated=False`, NN/epsilon=None assert no final graph.
Original scene NMS IDs, coordinates and Fusion remain immutable. Store
`refined_position_mm` separately and preserve the existing ACCEPT-only refined
seed handoff to the canonical plane validator. No duplicate merging is added.

All screens remain manager-owned and read stored evidence only. The inspector
retains six Q10/path/support/history panels and per-plane polar histograms/cards.
Orange/green/purple identify fitted planes. Cached raw azimuth arrays populate
36 ten-degree bins; arcs use the corresponding plane color and dim crosses
mark discarded angles. Cards read stored span/gap, kept/raw counts, total/tail exclusions,
reliability, plane RMS and newly explained support. Unreliable planes display
UNKNOWN rather than PASS. An early Q10 rejection displays RANSAC SKIPPED.

`view3d.q10_spherical_plane_primitives` reconstructs display-only circles,
projections and kept arcs from cached angles/endpoints in the model's angular
basis. Projections use radii 1.11+.025*plane_id. The native mm view keeps one
analysis cloud and .8R cleaned arcs around the final refined point. As of
2026-11-11, voxel colors read ALL cached plane memberships: P1 orange, P2 green,
P3 purple; multiple memberships are white. Points discarded by every member
plane use 30% brightness; retention in any plane keeps the point bright. Blue
means unscored and gray means no fitted plane. Per-plane membership/retention
matrices preserve overlap evidence without duplicating or moving points.
Projected inliers and cleaned arcs use the same plane palette; excluded
projections/crosses are dimmed in their own plane color. The Q10 arrow stays
cyan. Native/Tk titles and legends use these current semantics. Physical panels keep equal mm
scales. Step navigation changes Q10 history while final angular evidence stays
fixed. Spawned-view lifecycle and manager ownership are unchanged. Display
thinning never changes numerical supports, scores or stored arrays.

Original NMS markers are red, converged positions orange and final statuses
ACCEPT green / REJECT red / INDETERMINATE amber. All statuses remain clickable
at their refined coordinates. The spawned native-view lifecycle is unchanged.
With registered RealSense RGB, the final Lite view overlays only ACCEPT green
markers on the unchanged original RGB cloud. `--only` retains this final RGB
view. Other statuses remain inspectable in NMS/convergence views.

## Saved dataset detector input (2026-11-10)

`--source dataset --mode detector --input PATH` accepts a single scene containing
`points.npz`, a dataset root, or the `corner_dataset_tool` directory containing
`dataset/`. `dataset_input.run_dataset_detector` uses the existing `SceneData`
loader and numeric scene ordering, then calls unchanged `voxel_downsample_3d`
and `run_dense_heatmap_mode`. No camera, fake depth/intrinsics or organized
pixel mapping is required. Dataset input currently supports detector mode only.

`--dataset-region roi` is the default and requires saved `annotations.json`;
select its inclusive AABB using `SceneData.roi_indices`. `scene` selects every
raw row and does not require labels. Empty selections, missing input/ROI or
insufficient voxel support fail explicitly. XYZ remains mm and RGB follows the
same selected original row IDs. Original raw indices, ROI-local input rows,
scene voxel indices and Q10 analysis indices remain distinct; no voxel centroid
is claimed to be an observed/GT point. The algorithm never uses GT corners.

The selected raw XYZ supplies both scene voxelization and Q10's independent
analysis voxelization. Selected RGB/XYZ also supplies the manager-owned color
result view, with the same Lite marker/inspector behavior as live input. Closing
views advances to the next scene; `Ctrl+C` stops a root traversal.
`--no-3d-view` runs all computation/logs headlessly. Results are not accumulated
across scenes or exported. All input capture/annotation/ROI files remain
read-only. Existing synthetic depth noise requires organized frames and is
rejected for this saved-XYZ source; saved depth/RGB images are not reconstructed.
STL, Kinect and live RealSense dispatch/configuration remain unchanged.

## Shared lab demo input (2026-11-11)

`pointcloud_inputs.demo.make_demo_cloud` owns the standalone lab generator
with exact default-geometry parity. The lab imports this function, preserving its public name and saved
demo behavior. `--source demo --mode detector` generates the same 12,601 raw
XYZ/RGB rows in memory without opening the lab or importing its GUI. Runner
uses existing `voxel_downsample_3d` and `run_dense_heatmap_mode`, passing original
XYZ for Q10 and aligned XYZ/RGB for manager-owned diagnostics. No files are read
or written, and the known corner index does not enter detection.

Only the input changes: current detector settings/CLI retain control of scene
voxelization, scoring, NMS and Q10/MaxGap. The fixed geometry seed is 7; the
depth-noise seed does not reseed it. Demo supports detector mode including
`--only` and `--no-3d-view`, and rejects an input path or organized depth noise.
It provides no image grid for the organized fine validator. Scene voxel IDs,
raw rows and Q10 analysis IDs retain their existing distinct index spaces.

The shared generator additionally accepts normal-noise sigma (default 0.25 mm),
smooth-bump amplitude (0 mm), wavelength (20 mm) and seed (7). No-argument
output remains byte-identical to the original demo; detector CLI still uses
these defaults. The lab's `Input -> Demo surface` controls generate only when
`Generate / apply Demo` or the common Demo button is pressed. Slider changes
update labels without changing loaded data. The original three 4,200-point
blocks, RGB/order and fixed known corner row are retained. Each plane receives
independent normal Gaussian noise plus A*sin(2*pi*u/w)*sin(2*pi*v/w) bumps;
same-seed sampling/noise draws are unchanged by intensity/wavelength changes.

Default settings retain captures/demo_corner and its existing labels. Any
nondefault parameter creates a new timestamped demo_corner_* folder, with
parameters in metadata.json and fresh demo labels, so existing variants or
default refinements cannot become attached to different geometry. Generation
validates before writing. Demo reload reuses existing cache/worker invalidation
and does not modify camera capture/depth handling, scoring or validators.

## Standalone lab dataset selection (2026-11-10)

`corner_lab_open3d_cone.py --scene N` resolves
`corner_dataset_tool/dataset/scene_NNNN/points.npz`, anchored beside the script.
`--dataset` overrides the root and accepts the tool directory with `dataset/`
as well. Scene numbers identify folders, never ordinal positions. Resolution
and input conflicts fail before GUI initialization. Existing `--camera`,
`--demo` and `--open` behavior is retained. Refinement job tokens/worker flags,
result caches and spherical-plane settings are initialized before panel
callbacks or startup scene loading can use them. Numerical defaults match the
visible controls; loading and selecting a scene cannot use missing state.

`load_cloud` supports the lab's points_mm/colors_rgb cloud.npz and the dataset's
xyz_mm/rgb points.npz. Both retain full raw order/coordinates/RGB without ROI
cropping, depth reprojection or unit conversion. Existing lab corners.json
remains authoritative; for dataset points only, its absence permits validated
GT import through `SceneData.load`, mapping raw_point_index to point_index and
integer IDs to stable gt_NNN labels. Missing annotations permits unlabeled
inspection. Loading writes nothing; lab edits/refinements still save only its
separate corners.json, never the dataset annotations or raw capture. Label
validation completes before replacing the current scene. Q10/voxel/RANSAC/
angular-coverage calculations and existing scene/analysis index spaces remain
unchanged by this input adapter. This standalone viewer is separate from the
canonical detector/validator and its manager-owned UI.

### Standalone lab workspaces (2026-11-11)

`--dataset-region roi` or the persistent `Dataset: saved ROI only` checkbox
selects the saved inclusive AABB from `SceneData.roi_indices`; default `scene`
preserves full-scene behavior. ROI selection requires annotations and rejects
empty/malformed saved ROI before replacing the loaded scene. Source XYZ/RGB
stay in full original raw order. A separate selected-raw-ID array limits the
display, picking tree, shared analysis voxelization and whole-scene comparison
input. Picking maps tree rows back to original raw IDs. Only labels anchored
inside the ROI appear in the active list; excluded labels are retained and
merged on explicit lab saves. Dataset annotations/raw captures are unchanged.
Demo/captured legacy clouds continue to use their complete input.

The lab opens in `Heatmap/NMS`, with a full-width cloud and shared scope,
CED/PCA radii and full-scene voxel/NMS controls. One criterion selects both
its cached heatmap and its own extrema: Fusion MAX, bilateral CED MAX,
CED-weighted PCA ratio MAX, CED-weighted lambda1 MIN, normal PCA lambda1 MIN,
or CED × local lambda1 minimum MAX. The last criterion uses existing Normal-PCA
lambda1 (largest covariance eigenvalue) and bilateral CED. Local min/max use
finite eigenvalues within the PCA radius R in the evaluated field, with
L=(local max-lambda1)/(local max-local min+contrast_tau*R²), S=CED*L.
`Min-Max contrast tau` is separately configurable, initially 0.005,
and joins the comparison cache key; change it then recompute. Original five
scalar fields/extrema are retained unchanged. The new field has MAX NMS and
its own lower score gate; zero responses are omitted from its peaks. Its
heatmap uses a fixed [0,1] range to retain visible suppression of low contrast.
`Local lambda1 minimum | HEATMAP` additionally displays cached L(p) itself,
before multiplication by CED. It is a heatmap-only view: NMS controls are
disabled, no new extrema/threshold/matches are introduced, and selecting it
does not recompute L or apply unrelated NMS gates.
`Auto Contrast` is an optional display-only checkbox for both L and CED*L,
initially OFF. ON uses existing finite-sample P2--P98 color limits; OFF retains
[0,1]. All-invalid fields stay grey with a [0,1] fallback. The legend identifies
the active scale. Toggling recolors the cached result without changing scores,
extrema, threshold gates, matches, workers or the comparison cache key. Existing
criteria retain their original color behavior and disable this checkbox.
Two additional criteria expose the Median variant: `CED x Median lambda1
minimum | MAX` and `Median lambda1 minimum | HEATMAP`. They use the same
Normal-PCA lambda1, bilateral CED and radius R as Min-Max, replacing the local
maximum with the finite-neighbor median and clamping L to >=0. Median contrast
tau is independently configurable, initially 0.005, matching the Min-Max default.
Both tau values enter worker snapshots/cache keys. Median CED*L has
independent positive-response MAX NMS and a lower score gate; pure Median L
is display-only. All four L/CED*L views support fixed [0,1] / Auto Contrast.
Original fields, NMS, pairings, ROI/indices, Q10 and ratio diagnostics remain
unchanged; this adds no ratio factor/gate to either localization score.
Heatmap and NMS visibility are independent; switching criteria or hiding
markers does not recompute scores or alter cached extrema/counts. The existing
four paired-method comparisons remain in a collapsed optional group.
The NMS thresholds group accepts seven independent, finite nonnegative limits;
blank is OFF. MAX fields retain scores >= limit; MIN lambda1 fields retain
scores <= limit in mm². Apply filters a copy of cached extrema and rematches
the surviving sets, without rerunning scores/NMS or changing the base cache.
Heatmap colors remain those of the full selected input. Counts report kept /
unthresholded extrema for the selected field, plus filtered primary/secondary
and pair counts in comparison mode.
Full-scene heatmaps hide the underlying RGB cloud to keep voxel colors clear;
turning the heatmap off or clearing it restores RGB. Local views retain the
surrounding cloud for depth picking. The shared CED radius remains editable
even when the optional PCA refinement mode is unweighted.

`Corner` reveals the existing unit sphere and histogram alongside the cloud,
with grouped Q10 refinement, spherical planes, MaxGap and inspection overlays.
A successful Ctrl+click enters this workspace using the unchanged raw-ID
label/inspection behavior. `PCA` retains the existing eigen-field/refinement
tools; `Input` contains camera preview and capture. Demo/open controls remain
available above every tab. Tab changes affect layout and inspection visibility
only; original score/NMS arithmetic, Q10/RANSAC/MaxGap decisions, worker cancellation,
stored labels and scene/analysis indexing contracts are unchanged.
Register optional unit-sphere/histogram widgets after all permanent controls.
Open3D derives top-level ImGui window IDs from visible child order (see
[Window::DrawOnce](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/visualization/gui/Window.cpp#L787)); hiding an
earlier child otherwise changes the tab panel's ID and resets its selection.
Native tab callbacks also avoid reassigning the already selected tab index.
Programmatic navigation only assigns when the requested tab differs, including
during an active comparison worker.

### Selectable Q10 direction (2026-11-09)

`q10_direction_mode` / `Q10ValidatorConfig.direction_mode` / CLI
`--q10-direction-mode {unit_mean,unit_median,centroid}` selects the axis used
both during relocation and for the cached final Q10 quality gate. The default
`unit_mean` exactly preserves the existing unit-direction mean. On the same
Q10 support S (query-connected R sphere when its CC is ON, full R sphere when OFF),
let v_i=x_i-p and u_i=v_i/||v_i|| for ||v_i||>1e-7 mm. Choose a as:

- `unit_mean`: arithmetic mean of the u_i;
- `unit_median`: coordinatewise median of the u_i (not a geometric median);
- `centroid`: arithmetic XYZ centroid of S minus p, retaining original distances.

Normalize m=a/||a||, then use the unchanged ordinary NumPy lower-10% cosine
quantile Q=quantile_0.1(clip(u_i dot m,-1,1)). All quantile observations have equal
weight. The centroid mode changes only the axis; it introduces no quantile weights
and does not reuse the differently sampled dense CED centroid. Fewer than three
nonzero directions or ||a||<1e-8 makes Q undefined with existing INDETERMINATE
handling. Each refiner instance/cache owns one immutable direction mode. Keep
the `mean_axes` trace field for callers; it now stores the selected normalized
Q10 axes. Records and result metadata identify `direction_mode` explicitly.

The physical 3-D debug cloud shows a solid cyan arrow from each selected trace
position p to p+R*m, with shaft/head radii .018R/.05R and lengths .78R/.22R.
It follows cached step selection; its display length is R, not the magnitude
of the mean/median/centroid vector. Physical-mm trajectory/final-support plots
show the same arrow in teal. Unit-sphere/histogram labels name the selected
mode. These diagnostics only read stored axes and positions; they never
recompute directions, supports or acceptance. The antipodal axis remains an
independent search and keeps orange/violet support colors.

## Detector seed handoff to validator and total (2026-11-09)

Detector, interactive validator, new-validator and total share the same `q10_validator_enabled`
default and `--[no-]q10-validator` switch and all Q10 options. Fine/ball/extremity
keep their existing paths and do not run Q10. Dense CED/PCA/Fusion/NMS values and
the original candidate extraction are unchanged. `prepare_validator_seeds` in
`validator_pipeline.py` composes NMS provenance with Q10 results once:

- Validator master OFF: all NMS scene IDs, with their original scene XYZ.
- Validator master ON: `accepted_indices` (status ACCEPT only), with each record's stored
  `refined_position_mm`. REJECT and INDETERMINATE are excluded. No accepted
  seed means no validator queries; there is no fallback to original NMS points.
  Q10's own records/kept_indices still retain INDETERMINATE for diagnostics;
  this stricter rule applies only to downstream validator seeds.

`scores.validator_seeds` stores aligned candidate_indices, positions_mm,
original_positions_mm and source (`detector_nms` without the validator,
`q10_refined` with refinement ON, or `maxgap_nms` with refinement OFF). Refinement
OFF still passes only angular-ACCEPT IDs and preserves their exact NMS centers. It does not
overwrite the scene, NMS arrays, Fusion scores or Q10 records. Distinct original
NMS IDs remain distinct even if they converge to the same coordinate.

The canonical `run_validator_pipeline` still receives a scene `query_index`;
its optional `query_position_mm` supplies an independent finite XYZ center.
`run_validator_candidates` accepts aligned `query_positions_mm`. Without these
arguments both retain the original index-based behavior. With them, raw core/
halo crop, voxelization, guided RANSAC, plane support and edge/corner geometry
use that center. The organized AHC adapter similarly centers its pixel mask,
ROI and projection support there. No synthetic point is appended and no scene
point is moved. Support IDs remain in the original scene space, with raw,
query-local RANSAC and Q10-analysis IDs still distinct.

Result `query_position_mm` is the center actually used; `coarse_query_position_mm`
is the original `points_mm[query_index]`. Voxel and organized extraction caches
include both original ID and actual center, so ON/OFF or a changed refinement
cannot reuse a crop from a different center. Q10 equations, support gates,
refinement bounds and validator equations/radii are otherwise unchanged.

Interactive preparation receives the raw cloud required by Q10. Kept IDs and
positions are frozen together and passed to the spawned view. Marker placement,
3-D and RGB picking, ROI markers and the numerical query all use these same
cached positions while returning original scene IDs. Display does not run Q10
or mutate scene data. Total uses the same seed bundle and canonical batch
pipeline without debug comparisons. Library detector-to-validator composition
also forwards Q10 positions when its explicit validator is enabled.

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

`total.run_total_mode` composes unchanged dense scoring/NMS with
the shared optional Q10 seed policy and `validator_pipeline.run_validator_candidates`. Every kept seed runs the canonical
query-local pipeline and contributes every unverified Corner, without selection,
structural validation or cross-query merging. `final_corners` and `final_edges`
qualify local Corner/Edge/Endpoint IDs by `query_index`. Keep each contributor's
active intervals and original geometry. Compact query records retain Corner
lists and timers; optional core-only debug comparisons are skipped.

Reuse the existing raw/Coarse KD-trees, independent core/halo voxel sampling,
seeded RANSAC and scene/native-ID contracts. Manager owns two result windows:
unchanged detector Fusion/NMS and original gray raw XYZ with all Corner
hypotheses red, Boundary contributors green, intersections cyan and Endpoint
links yellow. Existing batching, sizes and R/Q/Esc lifecycle remain. Total
requires geometry enabled and opens no intermediate query windows.

## Coarse-query guided validator (`--mode validator`)

`validator_query.prepare_validator_queries` reuses the unchanged detector
CED/bilateral/PCA/Fusion/threshold/NMS composition followed by configured Q10.
Only a kept original Coarse NMS scene ID can be selected; its separate start
position is the Q10 refined center when enabled. `run_validator_pipeline` dispatches
`ValidatorDetectorConfig` directly to raw query-radius cropping, local voxelization,
`guided_ransac.extract_guided_planes`, and geometry composition. The independent
`ValidatorConfig` / `--mode fine` organized AHC/Hough/refiner path is unchanged.

There is no guided Fine scoring/NMS/Top-K, validator-side relocation, fallback or second
click. All `validator_fine_*` settings and `fine_*` result/cache/picker fields are
removed. The selected detector seed center is used at every subsequent stage.
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
Local IDs cannot be reused across queries. The actual seed center (Q10 refined when enabled) need not be a
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
overlap masking, endpoint Corner integration, and every merged Corner
candidate with its contributing Edges on the original raw cloud.
There is no window 7. All three auxiliary windows show only that same query.
All visualization stays manager-owned and reads stored numerical results.

## Independent projection-only validator (2026-11-09)

`--mode new-validator` adds a separate interactive path. It shares detector
scoring/NMS and the configured Q10 policy (ACCEPT-only refined seeds when ON;
original NMS seeds when OFF). It calls the same canonical
`run_validator_pipeline`, query-local raw crop/independent core-halo voxelization,
`extract_guided_planes` and `reconstruct_plane_support` as `validator`. There is
no copied RANSAC, support recollection, connectivity or projection algorithm.
The original `validator` and `total` stage order remains unchanged.

`NewValidatorConfig` contains the independent plane/projection controls and common 2-D Q10 lower bound and is resolved
from independent `new_validator_*` defaults/CLI fields in `settings.py`:

| Setting suffix (`new_validator_…`) | Initial value | Units / meaning |
|---|---:|---|
| ransac_voxel_mm | 5 | local voxel mm |
| ransac_radius_x | 15 | core radius / local voxel (75 mm initially) |
| max_planes | 3 | maximum sequential planes |
| ransac_iterations | 256 | iterations per plane |
| inlier_threshold_x | 1 | strict residual threshold / local voxel |
| guided_sigma_x | 1.5 | existing guided first-point sigma / local voxel |
| min_inliers | 35 | absolute inlier minimum |
| min_inlier_ratio | .08 | unchanged ROI-relative minimum |
| random_seed | 42 | existing per-query RNG seed |
| roi_padding_mm | 10 | additional support shell, independent of voxel |
| plane_connect_radius_x | 4 | existing 3-D support CC radius / local voxel |
| min_q10 | .10 | shared lower bound for every plane cosine Q10 |

These values were copied from the active validator at creation, then made
independent. Existing `--validator-*` controls do not affect the new path;
`--new-validator-*` controls do not affect the old path. New mode always runs
projection, without an exposed geometry/hull/edge/corner switch. Shared plane
input validation retains the original validator's arithmetic and errors.

The support path retains strict plane residual membership over the padded ROI,
the same query-near connected component selection with core overlap, plane
normal/offset normalization, tangent basis and projected query origin. For
selected native support IDs i, cache ordinary mm coordinates
`UV_i=[(x_i-origin) dot u, (x_i-origin) dot v]` in `projection_uv` before any
rank check/hull. The existing minimum of three connected samples still applies;
sparse support has an empty `(0,2)` projection and its existing failure reason.
In new mode only, compute the centroid-direction Q10 gate described below, then stop: no SVD/rank rejection, ConvexHull, finite edges,
overlap/TLS or Corner generation runs. Thus a collinear projected set can be
shown, without asserting that it forms a hull. Projection patch validity means
anchored connected support, not hull/edge/Corner validity. Old mode continues
its original rank/hull checks with unchanged numerical results.

Results explicitly report `projection_only=True`, `projection_computed=True`,
`geometry_computed=False`. `validation_status` is now ACCEPT/REJECT/INDETERMINATE for the all-plane 2-D Q10 gate, while `verified=False` still disclaims structural Corner verification. Plane patch
UV/support/native/scene IDs remain available even in non-debug batch output.
Hull arrays stay empty and areas zero; Edge/Corner lists are not computed.
Raw IDs, query-local RANSAC IDs, original scene IDs and separate query centers
retain the existing contracts. Display never reconstructs or modifies UVs.

The managed UI retains seed, query-voxel and fitted-plane views. Its only extra
window shows per-plane cached UV scatter plots (U/V in mm, projected query at
zero), rendered by `visualization.plots` and displayed through the manager-owned
Open3D image widget. Hull/edge/Corner windows are not created. The existing
spawned-view isolation, cached seed coordinates and close/recapture flow remain.

### Per-plane centroid cosine Q10 gate (2026-11-09)

The projected query is UV origin (0,0). For each fitted plane, use the same
cached selected-support `projection_uv` coordinates. Compute the arithmetic
centroid c=mean(UV), then normalize d=c/||c||. This uses original UV distances,
not the mean of unit directions. Exclude projected zero offsets (length <=1e-7
mm) only from cosine observations. For the remaining v_i, compute
cos_i=clip((v_i/||v_i||) dot d,-1,1), then ordinary unweighted NumPy Q10 at .1
with its existing linear interpolation. UV basis rotation/reflection changes
coordinate components but not these cosines. No external axis, radial weighting,
histogram or axis search is introduced.

Use one independent setting `new_validator_min_q10` / `NewValidatorConfig.min_q10`
/ `--new-validator-min-q10`, initially .10 and finite in [-1,1]. Equality passes.
With one, two or three fitted planes, every plane must have a defined Q10 >=
that same threshold. One known failure makes the result REJECT. Otherwise any
undefined score, or zero fitted planes, is INDETERMINATE and never ACCEPT.
Fewer than three nonzero UV directions or centroid norm <1e-8 mm leaves its
score undefined; no fallback direction is invented. This Q10 gate adds no
additional connectivity/rank/hull condition beyond the existing UV support
selection. Existing plane-component uncertainty remains separate metadata.

`plane_q10` stores each plane_id, centroid_uv_mm, normalized direction_uv,
nonzero_mask, cosines, nullable q10/passed and reason. Results also store the
common q10_threshold, passed_plane_count, boolean accept, validation_status and
failure_reason. The `projection_q10` stage timer and total elapsed include this
calculation. Existing RANSAC, CC, UVs, IDs, detector/Q10 seed policy and old
validator equations remain unchanged; Hull/Edge/Corner are still not computed.
The gate classifies projected direction distributions, not physical Corners.

The projection plot reads stored evidence only: a teal arrow from query to
centroid, per-plane Q10/threshold/PASS-REJECT-UNKNOWN and the combined result.
It never recalculates the axis or quantile while rendering.


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
original core support nearest the Coarse query is selected. Multiple nearby
components do not invalidate that selection, and the Corner gate radius is not
used for plane support. Missing anchored support or degenerate hulls remain
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
endpoint inheritance and retain inactive provenance after fitting.
Inactive-ended Lines still provide interior geometry. Groups require a shared
Node ID and cannot bridge separated chains. Original endpoint metadata, fitted
coordinates, source Hull Edge IDs and RMS/maximum TLS residuals remain separate.
Fitted extrema are re-clipped when needed without promoting inactive states.
`boundary_debug` stores every stage; the Corner stage adds no core gate.
Long unobserved intervals are no longer rejected by edge-support evidence.
Each valid finite candidate becomes a graph Edge without same-type deduplication,
ordered by descending support quality, descending length and source ID. Each
`source_edge_ids` list contains only its own source ID. Distinct Hull chains
remain separate even when they share endpoint Node IDs. Plane-pair adjacency
is stored explicitly. Intersection-aligned, nearby overlapping boundary ranges are union-masked in their
original TLS parameter space. `all_edges` retains original endpoints/directions,
masks and inactive records; `edges` contains only active sources. Mask cut endpoints are not materialized as new Corner endpoints.

`validator_corner_graph.generate_corners(edges, min_angle_deg=20.,
corner_radius_mm=5.)` independently generates Corner candidates from finite
Edges. Filter by unoriented direction angle, solve infinite-Line closest points,
check skew residual against existing Edge fitting tolerances and test proximity
to both Edges' nearest endpoints with one shared radius. Endpoint IDs are
Edge-local sides 0=start/1=end. Integrate only groups sharing an Edge at the same
Endpoint, with no conflicting Endpoint assignment and full pairwise position
diameter bounded by that same radius. Preserve the smallest-endpoint-gap member
position; union Edge/Endpoint provenance without averaging or refinement.

The Corner stage has exactly two controls. CLI/config fields are
`validator_min_angle_deg` / `min_angle_deg` and `validator_corner_radius_mm` /
`corner_radius_mm`. Current user-edited CLI values are 45 degrees and 15 mm;
standalone reference defaults remain 20 degrees and 5 mm. Existing Edge fitting
bounds are metadata, not a new Corner setting. Plane/Hull/TLS/mask computations
and native scene-index arrays remain unchanged.

There is no same-plane shared-Node requirement, Node-activity/mask-endpoint
recheck, endpoint extrapolation allowance, query/core gate, anchor or multi-Line
LS, selected Corner or structural acceptance. Original Edge clipping/Node
provenance and overlap masks are still stored. Only valid Edges enter the
generator, which uses their supplied original finite endpoints. Thus crop-created
or masked original endpoints may produce candidates. No observed physical edge
or Corner is certified.

Results store a `corners` list: position, contributing Edge IDs, Endpoint map,
representative pair intersection residual and endpoint gap. Status is CANDIDATES
or NO_CANDIDATES, with `verified=False`; disabled geometry is NOT_COMPUTED.
Remove singular Corner/selected-vertex and old vertex/refinement result fields.
Pipeline composition remains in `run_validator_pipeline`, including the optional
core-only Before comparison. Interactive and total consumers use all Corners.
All manager-owned Corner views show unverified candidates and Endpoint links;
they do not recompute numerical geometry or move input Edges.

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

Dataset baseline comparisons also enter through `visualization.manager`.
`baselines.config` owns dataset/author presets. The dataset runner defaults to
the dataset preset, resolves explicit CLI overrides once, and records the
resolved values. Ahmed library calls retain author defaults. Optional Sommer
input voxel sampling selects real observed representatives, retaining a separate
sample-to-raw map; normal support IDs are mapped through it. Shared comparison
backgrounds still use the original saved ROI/full-scene input. Numerical
stages do not read GT; the offline preset selection is documented separately in
`BASELINE_TUNING.md` with a distinct confirmation-scene check and its limits.
`corner_dataset_tool.baselines.ahmed` owns the explicit Python port of the
author's MATLAB edge/corner stages; `--method ahmed` selects it by default.
`--ahmed-backend matlab` retains the original author-code adapter. Port results
are labelled `python_port`, including a source hash and unverified MATLAB parity;
these paths do not import or alter the detector's numerical stages.
`corner_dataset_tool.view_baselines` loads the saved raw-index selection, GT
and predictions, without reading live annotations or recomputing detectors.
`manager.show_baseline_comparison` gives each scene its own spawned GUI process
owned by `query_ui`; `view3d` builds one row of GT/method panels and synchronizes
their cameras. Every panel uses the identical saved ROI/full-scene input in mm.
GT and prediction positions and all numeric/index arrays remain unchanged.

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
