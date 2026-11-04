# Active detector and validator methods

This is the canonical ledger for the active numerical pipeline. All coordinates
and metric thresholds are in millimetres unless stated otherwise.

## 1. Stage order

Interactive fine queries and explicitly validated detector NMS seeds share one entry point:

```text
voxel scene + organized RGB-D context
  -> exact query-radius organized mask
  -> CornerVINS-style AHC regions
  -> per-region RANSAC + full-resolution least-squares refit
  -> repeated compatible-patch merge + convex hull
  -> optional padded 2-D boundary + weighted Hough
  -> 3-D finite edges + plane adjacency
  -> optional graph-local robust refinement + corners
```

`run_validator_pipeline(points_mm, query_index, cfg, estimator=None,
organized_context=...)` is the canonical numerical entry. CornerVINS plane
extraction is unconditional for `ValidatorConfig` calls. CLI detector mode stops at NMS and does not construct this config. There is no backend selector and no FPS/SAH
reference-plane clustering fallback.

`--mode detector` stops after the unchanged dense CED/bilateral/PCA/fusion-threshold/3-D NMS composition. Both the CLI detector and default `DetectorConfig` use `validator=None`; no plane, edge or corner computation runs. Scores, thresholding, NMS and scene IDs are unchanged. Library callers can still explicitly supply `ValidatorConfig` to invoke the shared validator.

`--mode detector --only` additionally limits visualization to the NMS heatmap. It resolves
`DetectorConfig.only=True`, `validator=None` and omits all validator, plane,
edge and corner computation. Returned dense scores and NMS provenance are the
same as the normal detector before validation; they contain no
`validator_results` or post-validation candidate statuses. Rendering selects
only the existing NMS Corner Candidates panel, preserving its Fusion valid
mask, linear Q99 range, red scene-indexed markers and M marker toggle.
No geometry/CED/PCA/Fusion-only or auxiliary registered RGB windows open.
`--no-3d-view` suppresses the remaining window too. The option is valid only
with `--mode detector`; omitting it retains the existing dense diagnostic panels without post-NMS validation.

### 1.1 Coarse-query guided sequential validator

`--mode validator` reuses the detector's unchanged voxel-scene CED → bilateral
CED → weighted PCA/Fusion → threshold → radius-local-maximum NMS. Selecting an
existing Coarse NMS sample runs the entire following composition immediately:

```text
Coarse NMS query q_c (unchanged scene coordinate)
  -> inclusive radius crop from original raw XYZ
  -> query-local voxel downsampling
  -> first-point-guided sequential RANSAC (core voxels only)
  -> support recollection / hulls / finite edges / corner decision
```

There is no validator Fine score, second threshold/NMS, Top-K, query movement,
Fine picker, or fallback stage. `validator_fine.py`, `validator_fine_*` defaults
and CLI options, and resolved `fine_*` fields were removed. The separate
`--mode fine` organized AHC/Hough/refiner pipeline is unchanged.

**Crop before voxel.** For raw input $X$ and the selected scene coordinate $q_c$,
let $X_c=\{x\in X:\|x-q_c\|\le R\}$. Apply the existing Open3D centroid voxel
downsampler to $X_c$, using `validator_ransac_voxel_mm`, to obtain $P_c$.
The downsampler's local grid origin follows that cropped input. Do not reuse
Coarse voxels even when sizes match, and do not crop an already voxelized
full-scene cloud. Therefore this intentionally changes sampling/centroids and
potential plane results relative to the former global stage clouds.
The query stays $q_c$ even when it is not a local voxel centroid.

When geometry is enabled, also crop the disjoint raw shell
$X_h=\{x:R<\|x-q_c\|\le R_s\}$, $R_s=R+P$.
Voxelize this shell separately and concatenate $P=[P_c;P_h]$. This prevents
outside observations from moving core centroids or entering RANSAC; only $P_c$
is passed to the unchanged RANSAC extractor. Geometry support reads $P$.
The two independent voxel grids can produce nearby samples along the interface;
this preserves raw core/halo separation rather than a single global lattice.
With geometry disabled, no support shell is prepared. Here $P$ is the resolved
`roi_padding_mm` (absolute default 10 mm, finite/nonnegative). There is no
independent support-radius floor or outer-contact margin. Default core/support
radii are 75/85 mm; padding never changes the core voxel cloud.

`prepare_validator_voxel_context` caches validated raw/Coarse arrays and trees
per frame. It stores only the most recent query's voxel cloud, keyed by query
ID/coordinate, voxel size, core radius and effective support radius. Repeating
that query reuses voxelization and reruns fitting; another query replaces the
local cache. A context from a different Coarse scene is rejected. Source clouds
and Coarse scores are never modified.

All validator `*_x` lengths multiply the RANSAC voxel; absolute `*_mm` fields,
counts, angles, ratios and seed do not scale. Current workspace settings are:

| Setting | Value | Resolved meaning |
|---|---:|---|
| `validator_ransac_voxel_mm` | 5.0 | local voxel in mm |
| `validator_ransac_radius_x` | 15.0 | 75 mm raw core radius |
| `validator_inlier_threshold_x` | 1.0 | strict 5 mm plane gate |
| `validator_guided_sigma_x` | 1.5 | 7.5 mm sampling sigma |
| `validator_max_planes` | 3 | maximum accepted planes |
| `validator_ransac_iterations` | 256 | hypotheses per round |
| `validator_min_inliers` | 35 | absolute support floor |
| `validator_min_inlier_ratio` | 0.08 | fraction of original voxel ROI |
| `validator_random_seed` | 42 | query-local RNG seed |

**Guided sequential RANSAC.** For the inclusive RANSAC-voxel ball
$I=\{i:\|p_i-q_c\|\le R\}$, compute the fixed acceptance count
$N_{min}=\max(N_{configured},\lceil\rho |I|\rceil)$. It never shrinks as planes
are removed. In each round let $A$ be the remaining samples and set
$g_i=\exp[-\|p_i-q_c\|^2/(2\sigma_s^2)]$. The first sample is drawn with
$P(a=i)=g_i/\sum_{j\in A}g_j$, using log-weight subtraction for stable
normalization and uniform fallback when no valid normalization survives.
Probabilities are renormalized on remaining points every round. Draw the other
two points uniformly without replacement from $A\setminus\{a\}$.

With $v_b=p_b-p_a$, $v_c=p_c-p_a$, reject any pair separation at or below
$10^{-9}$ mm or cross-product norm at or below
$10^{-12}\max(1,\|v_b\|\|v_c\|)$. Otherwise set
$n=(v_b\times v_c)/\|v_b\times v_c\|$, $d=-n^Tp_a$. The unchanged Coarse query defines
only the ROI and first-point sampling probabilities; it is not constrained to
lie on a plane.

Evaluate $r_i=|n^Tp_i+d|$ only on remaining ROI samples; strict inliers have
$r_i<t$. Select by greatest **integer inlier count**, then smallest inlier RMS;
exact ties retain the first hypothesis. There is no distance-weighted score,
soft support, MSAC, plane-angle filter, query-plane penalty, refit or
reassignment. Accept only support at least $N_{min}$, save native RANSAC voxel IDs and
remove those inliers from both sampling and evaluation in subsequent rounds.
Stop at zero to three accepted planes, insufficient remaining support or no
valid hypothesis. A query-local seeded RNG makes results reproducible for the
same input order/configuration.

**Entry point, IDs and results.** `run_validator_pipeline(points_mm, query_index,
cfg, raw_points_mm=...)` remains the canonical composition. Cached calls supply
`voxel_context`. `ValidatorDetectorConfig` selects this guided path;
`ValidatorConfig` retains the independent organized pipeline.
`query_index` indexes the supplied Coarse scene. `query_position_mm` and
`coarse_query_position_mm` both equal its unchanged coordinate.
`roi_raw_point_indices` addresses the original raw input before local voxelization.
`ransac_points_mm` stores $[P_c;P_h]$; native `roi_ransac_indices`, plane
`inlier_ransac_indices`, and geometry support IDs address this query-local array.
RANSAC masks address `roi_points_mm`, and accepted plane masks remain disjoint.
`core_voxel_count` and `halo_voxel_count` record the concatenation boundary.
Vertex `distance_to_query_mm` replaces the former Fine-distance name;
all distance gates now use the unchanged Coarse query.

`ransac_scene_indices` is the nearest-Coarse adapter for this local cloud.
It supplies scene-domain `roi_point_indices`, plane `inlier_point_indices`, and
geometry `support_point_indices` without replacing native coordinates, masks or
support counts. Mapping can be many-to-one. These adapters do not identify raw
contributors to a voxel. Local IDs from different queries are not interchangeable.
`planes_computed=True` includes empty-ROI/zero-plane runs; there is no Fine-wait
state. Report query, raw/voxel counts, voxel/core radius, plane diagnostics,
corner coordinates and incident finite edges, and voxel/RANSAC/geometry timers.

**Visualization.** The main native window keeps three panels:

1. Coarse Fusion heatmap and orange NMS queries, using the detector's valid mask,
   linear full-frame Q99 range and color function. Only this panel picks queries.
2. Stored core voxel input in gray, the orange unchanged query, and its radius
   sphere. Caption reports raw count → voxel count, voxel size and query radius;
   no second heatmap or candidate selection is calculated.
3. Fitted-plane inliers/disks/normals, core points and the same orange query.
   A two-line caption shows the stored per-plane RANSAC `inlier_count` with
   P0/red, P1/green and P2/blue labels, and the stored effective `min_required`.
   Counts are not recollected support counts; rendering does not change acceptance.
   The disk for $(n,d)$ has center $c=q_c-(n^Tq_c+d)n$ and radius
   $\sqrt{\max(0,R^2-|n^Tq_c+d|^2)}$. RGB extraction-order planes retain
   alpha 0.60, rims/diameters and normal arrows. Disks illustrate models,
   not observed boundaries.

Separate windows 4/5/6 display padded Hull/clipping/TLS diagnostics, pair
intersections and final integrated corner geometry as described below. All rendering reads stored
results through the manager, without fitting or mutating memberships. The
existing display-only worker, scene upload, R/Q/Esc and bounded shutdown remain.
`--no-3d-view` keeps RGB/depth Coarse selection and console reporting.
Finite RANSAC may miss weak planes, hard removal assigns shared support to earlier
planes, and local voxelization changes support counts. Fixed `min_inliers` may
require adjustment for sparse observations. No device accuracy claim is made.

### 1.2 Guided Plane–Edge–Corner validation

The default guided pipeline now continues after RANSAC:

```text
unchanged Coarse NMS -> raw query crop -> local voxel input
  -> unchanged guided sequential RANSAC (core only)
  -> independent support recollection in core + halo
  -> connected per-plane tangent projection + convex hull
  -> hull/strip finite plane intersections
     + padded Hull segment clipping -> ACTIVE/INACTIVE Nodes -> TLS boundary lines
  -> intersection/boundary overlap masks + active finite-edge graph
  -> anchored candidate grouping + constrained / multi-line least squares
  -> Coarse-query-centred vertex gate -> ACCEPT / REJECT / INCONCLUSIVE
```

Composition stays in `run_validator_pipeline`; the new numerical owners are
`validator_plane_support.py`, `validator_edge_graph.py` and
`validator_corner_graph.py`. There is no new Hough path, orthogonal-box constraint,
or connection to the independent fine-mode AHC/refiner algorithms.

**Support and indexing.** With the stored unit plane $(n_j,d_j)$ and selected
Coarse query $q$, recollect native RANSAC-cloud points as
$I_j=\{i:\|p_i-q\|\le R_s,\ |n_j^Tp_i+d_j|<t\}$, where
$R_s=R_{core}+P$ and $t$ is the existing RANSAC gate.
Every plane reads the original stage cloud; support may overlap across models,
and earlier hard removal never excludes a point from recollection. The original
models, disjoint core masks, native inlier IDs, counts and RMS remain unchanged.
Pipeline adds stable extraction-order `plane_id` fields to its plane records. Padding changes recollected support/Hulls but never
adds samples to RANSAC or changes its fitted models/core memberships.

Build a Euclidean radius graph on each $I_j$. Prefer components that overlap the
original plane's core inliers, then choose the nearest to the query (ties: core
overlap count, component size, label). Never hull disconnected components
together. Missing anchored support or multiple anchored components within the
corner gate mark connectivity UNKNOWN. At least three noncollinear points are
still required to form a 2-D convex hull; this is a geometric requirement.
All component sizes, original-core overlaps, selected component and all
recollected IDs remain diagnostic output. Connectivity cannot guarantee object
instances or resolve noise bridges.

Project the selected connected support using an orthonormal basis $(u,v)$ and
$o=q-(n_j^Tq+d_j)n_j$. Store $(p_i-o)^Tu,(p_i-o)^Tv$, hull vertices/equations,
lifted 3-D vertices and 2-D area. A relative SVD rank check and Qhull failure
handling retain degenerate/insufficient patches as UNKNOWN. Planes are not refit.
**Padded Hull -> ROI clipping -> Node-aware TLS.** Use support radius
`R_s = R_core + roi_padding_mm`. Intersect each original padded Hull segment
with the original core before TLS; never re-hull clipped points or join them
along the ROI circumference. In plane coordinates, project the query to the
stored plane origin and use the exact section radius `sqrt(R_core^2 - d^2)`,
where d is the query-to-plane distance. The existing analytic segment/sphere
helper works in these 2-D coordinates. A nonpositive section radius, zero-length
segment, tangency or empty interval yields no segment. Both-exterior segments
that cross the core retain their interior interval.

Retained original Hull Nodes keep stable plane-local IDs, source ORIGINAL and
`corner_enabled=True`. New clipping Nodes have distinct per-source-edge IDs,
source CLIPPED and `corner_enabled=False`. Retain inactive Nodes and their
Edges; inactivity forbids endpoint Corner evidence, not interior line use.
Each clipped Edge retains its Plane ID and original Hull Edge ID. TLS groups
merge only through a shared Node ID, including true cyclic adjacency; separated
open chains cannot reconnect. Merged Lines retain ordered support Node IDs,
all source Hull Edge IDs, their original traversal start/end Node identities,
source and enabled states independently of fitted coordinates. Store the fitted
maximum TLS residual. The SVD formula, complete-direction checks, maximum-error
threshold and greedy first-eligible merge policy are unchanged. Input clipping
can change fit values and merge outcomes.

Reclip fitted TLS extrema to the core if needed before minimum-length checks.
Any extra fit-created crop endpoint is CLIPPED/inactive with its parent Node ID;
projection never promotes a clipped Node to active. Intersection segments retain
their existing hull/strip computation and core clipping. Preserve source endpoints,
`roi_clipped_start/end`, pre-core support overlap and surviving core interval
diagnostics. Final vertices require the core and unchanged query gate. No point
support check is restored. `boundary_debug` stores original Hull Nodes/Edges,
clipped Nodes/Edges and final Lines; the UI only reads these snapshots.
Finite padding does not prove physical boundary existence or resolve interior
occlusion/hull bridges; a segment with inactive endpoints can still support a
genuine interior intersection.

Patch `support_indices` aliases native `support_ransac_indices`;
`support_point_indices` is its explicit nearest-Coarse adapter. Recollected IDs
and support ROI IDs retain both native and public scene domains. Adapter
duplicates never change plane support counts. Edge point-support slots are
retained but empty because point-to-edge memberships are no longer evaluated.

**Finite edges.** Exclude plane pairs whose unoriented normal angle is below
the configured threshold. For each remaining pair, solve
$[n_i^T;n_j^T]p_0=-[d_i;d_j]$ by least squares; its unit direction is
$v=(n_i\times n_j)/\|n_i\times n_j\|$. Translate $p_0$ along this line toward
the query for local parameterization. Do not fix an arbitrary coordinate axis.
For each tangent hull $H_i$, express the line as $p_i+t v_i$, with unit 2-D
direction $v_i$. Clip the unchanged polygon by the two half-planes
$-\delta\le\det(v_i,x-p_i)\le\delta$. Original vertices inside the strip and
boundary intersections define a convex polygon; project its vertices onto the
line and take the minimum/maximum line parameter. Convert projected parameters
back to the original unit 3-D direction when its tangent projection is not
exactly unit length. The two hull intervals contribute only their common
positive-length interval, subject to the unchanged minimum edge length.
Empty or point-like clipped intervals are invalid; strip boundary residuals
within $10^{-8}$ mm are treated as zero. The hull is never buffered.
`validator_intersection_strip_half_width_mm` is an independent absolute-mm
half-width (default 5.0, finite and nonnegative), exposed as
`--validator-intersection-strip-half-width-mm`. Zero uses the original exact
hull-half-space/line clipping without changing its numerical tolerances.
Values 0/2.5/5/7.5 mm are configurable; no comparison experiment was performed.

`plane_pair_diagnostics` retains its existing fields and adds
`strip_half_width_mm`, per-plane `hull_line_min_distance_mm`,
`exact_interval_valid`, `tolerance_interval_valid`, `interval_exact_mm` and
`interval_tolerance_mm` (arrays ordered by `plane_ids`), plus
`common_interval_mm`, `edge_length_mm`, `support_passed`,
`support_rejection_reason` and `recovered_by_tolerance`. Intervals parameterize
the query-nearest line origin in mm; absent intervals are None.
`recovered_by_tolerance` means a positive-width common interval reaches the
minimum length while the exact common interval does not. Point-support checks
have been removed: `point_support_checked=False`, `support_passed=None`, and
`support_rejection_reason=None`. These diagnostics are also attached to generated
intersection candidates and retained representatives. Strip intervals remain
geometric candidates and do not prove the existence of observed physical edges.

Boundary candidates start as core-clipped successive padded Hull segments,
including the original closing segment when it survives. Skip zero-length segments;
merge only Node-adjacent groups, including
the last/first group. Retain each group's original ordered vertices and segment
directions. For each proposed merge, deduplicate vertex coordinates for fitting,
compute their mean $\mu$, and obtain the unit TLS direction $v$ from the leading
right singular vector of the centered 2-D coordinates. Align its sign with hull
traversal. Require the existing complete-direction agreement of original segments,
plus agreement of both current group TLS directions and the proposed TLS direction,
using `corner_min_edge_angle_deg` without absolute-dot reversal equivalence.
Require $\max_i\|(I-vv^T)(p_i-\mu)\|\le\tau_{merge}$; RMS alone never accepts
a merge. The independent absolute-mm `validator_boundary_merge_max_error_mm`
(default 2.5, finite/nonnegative; CLI `--validator-boundary-merge-max-error-mm`)
replaces the former RANSAC inlier tolerance for this boundary merge check only.
Refit from all group vertices after every merge, retaining cyclic order and the
cyclic adjacency across surviving groups. After convergence, refit each group and use
$\mu+\min_i((p_i-\mu)^Tv)v$ and $\mu+\max_i((p_i-\mu)^Tv)v$ as finite endpoints.
Lift these 2-D endpoints with the stored plane origin and tangent basis, then
pass fitted coordinates together with inherited Node provenance to `_edge`. Degenerate TLS fits
produce no candidate. Plane-pair clipping and the existing minimum-length check
remain independent of boundary TLS fitting. Hull-based boundaries
remain hypotheses and do not guarantee physical object boundaries.

**Point-support removal.** `validator_edge_support_radius_x` and
`validator_edge_min_support`, their resolved config fields and CLI options are
removed together with point-to-segment distance filtering, minimum support
counts, gap/span/RMS diagnostics and point-support quality calculation.
A finite candidate is geometrically accepted only after Hull segment clipping,
core-sphere clipping and minimum-length check. The former boundary-midpoint-only
ROI check is replaced by this provenance/clip guard; the old point-to-edge radius
is not reused as a crop margin. Plane support recollection,
component selection, hull validity, TLS, strip width and plane-angle gates remain.
Edge `support_ransac_indices`, `support_indices`, mapped `support_point_indices`
and `support_by_plane` remain empty; `support_count=0` is an unmeasured
compatibility slot, not a measured absence of support. `support_quality=1.0` is
neutral for all edges, so deduplication falls back to length/source ID and it
provides no point-evidence ranking. Edge/result `point_support_checked=False`
and terminal `point support=disabled` make the new semantics explicit.

Finite-edge deduplication uses complete-link agreement of type/plane IDs,
unoriented direction, symmetric line offset and strictly positive interval
overlap. Keep the highest-ranked representative and all `source_edge_ids`; do not
extend it or fill a gap. Different-direction edges are never merged. The
original candidate records preserve other members' support/provenance.

**Intersection–boundary masking.** After same-type finite-edge deduplication,
compare each valid intersection $I$ with each boundary $B$ that owns one of its
planes. Require $\arccos(|u^Tv|)\le\theta_o$, both original boundary endpoints
within $d_o$ of the infinite intersection line, and projected overlap
$\max(0,\min(L_I,b_1)-\max(0,b_0))\ge\ell_o$ with sorted boundary projections.
Defaults are 5 degrees, 5 mm and 2.5 mm. Map each qualifying overlap back to the
original boundary's own parameter, accounting for direction sign, and union all
mask intervals from all intersections. Preserve original TLS start/end/direction
and source IDs. Store `masked_intervals_mm`, complementary
`active_intervals_mm` and `masked_by_intersection_ids`; a fully masked boundary
is INACTIVE and absent from active `edges`, while `all_edges` retains it.
`boundary_overlap_diagnostics` records masks. Intersections stay active.
A nonparallel crossing boundary is not masked. Active pieces are parameter
intervals of the original length-validated edge, not newly fitted segments;
mask cut endpoints are never vertex evidence or extrapolation origins.

**Vertex candidates and least squares.** Inspect active edge pairs. For two
Boundaries on the same plane, require a shared original ACTIVE endpoint Node
that survives overlap masking on both Lines. Absorbed intermediate Nodes and
nonadjacent Boundaries are not Corner candidates. Retain the existing midpoint
as the initial pair candidate after that topology gate. Cross-plane Boundary pairs and
intersection/boundary or multiple-intersection pairs retain their existing
closest-point/LS path; inactive endpoint provenance still forbids endpoint
evidence and extrapolation without disabling valid interior line support. Exclude
angles below the single `corner_min_edge_angle_deg` threshold. Solve closest
points using the full-rank matrix $[u,-v]$ and a stable least-squares solve.
Their gap must meet the single `vertex_line_distance_mm` threshold. Parameters
must be active and within the original finite extents plus
`vertex_endpoint_tolerance_mm`. Mask boundaries/interiors are excluded.
The same line-distance and endpoint thresholds apply when validating refined
vertices against each incident finite segment. There are no additional min/max
caps from legacy corner endpoint or parallel-angle settings.
When an intersection is present, its closest point $I(t)$ is the candidate,
not the midpoint of the two closest points. Boundary–boundary pairs start at
the midpoint, with the same active-interval and existing geometric gates;
their final grouped position is refined using all unique incident Lines.

Group candidates on each anchor by sorted $t$, requiring the entire group
span to be no more than `vertex_group_span_mm` (default 5 mm); proximity chains
cannot merge distinct locations. For one intersection and incident boundaries,
use unit weights and $Q_j=I_3-v_jv_j^T$:
$t^*=\sum_j u^TQ_j(b_j-a)/\sum_j u^TQ_ju$, $p^*=a+t^*u$.
Reject a denominator at or below $10^{-12}\max(1,N)$; clamp $t^*$ to the
intersection's actual finite range and recheck residuals and active extents.
This also resolves a single boundary's anchor point. For Boundary-only groups
and groups with at least two intersection Edges, solve using every unique
incident Line with equal weight:
$(\sum Q_j)(p-r)=\sum Q_j(b_j-r)$ in local reference coordinates, rejecting
rank deficiency by relative singular-value threshold $10^{-12}$.
Boundary-only groups no longer retain the first pair's midpoint when further
Lines join. Group merges refit all incident Lines with the same rank and
post-fit active-extent/residual checks; the single-intersection constrained
solution is unchanged.

Combine groups only with shared incident edge IDs, complete compatibility of
all member positions within the group span, successful refinement using all
unique incident lines, and valid residual/active-extent checks after refinement.
The refined point must also stay within the group span of every member.
Different intersection locations remain distinct. `vertex_candidates`,
`vertex_rejections`, per-vertex `refinement_method`, original member positions
and final `ls_residuals_mm` retain provenance. No RANSAC plane, hull or TLS line
is moved. The legacy vertex `supported` flag means geometric reachability only;
point-support validation remains disabled as previously requested.

Keep geometrically reachable vertices inside the core and with $\|v-q\|\le r_{corner}$.
For every refined vertex, store `intersection_edge_count`, perpendicular
infinite-Line fitting RMS `line_fit_rms_mm`, and `total_extension_mm` summed
over unique incident Edges. With `t = (p - start) dot direction`, an Edge's
extension is `max(0, -t, t - length)`. This ranking RMS is distinct from the
finite-segment distances in `ls_residuals_mm`, which still enforce the existing
acceptance gate and include endpoint overrun. Among accepted vertices, rank
lexicographically by: shared intersection present first, lower Line RMS,
lower total extension, lower query distance, then stable vertex ID. Round only
the ranking RMS and extension to eight decimal mm places so sub-1e-8 mm
floating-point noise does not override extension/distance tie breaking; this
adds no physical acceptance threshold. An intersection-free candidate remains
eligible when no shared-intersection candidate passes all existing gates.
Neutral legacy support quality and Fusion are not ranking terms. Defaults are
restored from experimental 30/30/30/50 mm to 5/5/5/10 mm for Line distance,
endpoint tolerance, full group span and query gate respectively. CLI overrides
remain available. Distance is measured from the unchanged Coarse query.
`corner_position_mm` is this reconstructed
position, not a replacement query sample. `corner_edge_ids` and `corner_edges` directly
provide its incident finite segments, endpoints, plane IDs and support records.
`verified=True` occurs only with ACCEPT and at least two valid geometric directions.
No planes, unreliable patches or insufficient local edge geometry yield
INCONCLUSIVE; sufficient patches without a geometric gated junction yield
REJECT. Disabled geometry yields NOT_COMPUTED while
preserving `planes_computed` independently.

**Initial settings.** Fields below live in `SettingsDefaults`; resolved fields
omit the `validator_` prefix in `GuidedValidatorConfig`. CLI uses hyphens.

| Setting | Default | Rationale / units |
|---|---:|---|
| `validator_geometry_enabled` | True | `--validator-geometry` / `--no-validator-geometry` |
| `validator_edge_overlap_angle_deg` | 25.0 | maximum unoriented intersection/boundary angle |
| `validator_edge_overlap_distance_mm` | 15.0 | both boundary endpoints to intersection line |
| `validator_edge_overlap_min_length_mm` | 2.5 | minimum finite overlap along the intersection |
| `validator_vertex_line_distance_mm` | 5.0 | sole closest-point gap / refined finite-segment distance gate; absolute mm |
| `validator_vertex_endpoint_tolerance_mm` | 5.0 | sole original endpoint extrapolation allowance; absolute mm; cannot extend a mask cut |
| `validator_vertex_group_span_mm` | 5.0 | full anchor-parameter span and complete spatial grouping gate |
| `validator_roi_padding_mm` | 10.0 | sole padding setting; absolute mm, support radius = core + padding; 0 supports core-only comparison |
| `validator_plane_connect_radius_x` | 4.0 | 20 mm at 5 mm RANSAC voxel; unchanged support component radius |
| `validator_plane_parallel_angle_deg` | 20 | small-angle intersection exclusion, not an orthogonality gate |
| `validator_intersection_strip_half_width_mm` | 15.0 | independent strip half-width in mm; 0 retains exact hull/line clipping |
| `validator_boundary_merge_max_error_mm` | 2.5 | independent maximum 2-D TLS vertex residual for adjacent boundary merges; absolute mm |
| `validator_edge_min_length_x` | 1.0 | 5 mm minimum finite edge length |
| `validator_corner_min_edge_angle_deg` | 30 | sole vertex minimum angle; also used by boundary coalescing and same-type edge direction matching |
| `validator_corner_gate_radius_mm` | 10 | absolute Coarse-query-centred mm gate |

The duplicate `validator_vertex_parallel_angle_deg` and
`validator_corner_endpoint_tolerance_x` defaults, resolved fields and CLI options
are removed. Current effective angle/distance/extension/group/gate values remain
30 degrees / 5 mm / 5 mm / 5 mm / 10 mm. Callers formerly using the removed
options must select the corresponding surviving thresholds explicitly. Changing
RANSAC voxel size no longer silently caps the absolute-mm vertex distances via
an additional voxel-multiplied endpoint setting.

All `*_x` geometry lengths multiply the RANSAC voxel; counts/angles do not scale.
These are initial experimental values, not calibrated accuracy claims. Existing
Coarse score/NMS settings and RANSAC fitting rules are preserved. The direct
query-local crop/voxel stage and its settings are described in section 1.1.
Positive lengths, boolean enable and (0,90]-degree
angle validation follow the existing config/CLI pattern.

**Display and timing.** The original third panel retains only core observations,
fitted-plane disks/normals and original query/core-ball markers, using alpha 0.60
and the original core-radius camera framing. Manager-owned auxiliary native
windows separate (4) padded Hull/ROI clipping/TLS and Before/After diagnostics,
(5) plane-pair intersections, and (6) the final four-panel edge/mask/vertex/raw
process. The former independent per-plane Boundary view is removed, its place
taken by the former window 7; no window 7 remains. Window 5 filters by stored
edge type, uses gray Hull rims for patch bounds and cyan finite shared Edges,
and omits ROI spheres, query markers and Corner overlays. Window 4 deliberately
shows core/padded ROI guides and Node provenance. Both read stored results;
the detailed panel semantics are described below.
Window 5 additionally draws the stored `intersection_line_origin_mm` and
`intersection_line_direction` as a yellow infinite-line guide over the display
span $[-R_s,R_s]$ around its query-nearest origin. On each owning plane, a
translucent RGB rectangle spans the same line range and the stored strip
half-width $\pm\delta$, with thin coloured strip boundaries and alpha 0.28.
Zero width draws only the line. These guides also appear when hull overlap or
minimum length fails, but not for pairs excluded as parallel.
A display-only normal offset of $0.001\max(1,R_s)$ mm avoids coplanar hull/strip
flicker; stored coordinates, clipping, support and acceptance are unchanged.
The caption reports the strip width and distinguishes guides from cyan accepted
finite edges. Main panel 3 retains its existing content.
Window 6 displays original edges, removed intervals, extension/LS estimation and
the selected Corner/incident Edges on raw XYZ in four panels as detailed below.
Only panel 6-4 adds a background cloud; none of its panels adds ROI spheres.
The disk remains an illustrative model/sphere intersection in panel 3; actual
computed hulls remain in windows 4/5, with muted rims in panel 6-1.
All windows live in the existing display-only worker. Auxiliary windows close
independently; main-window close, recapture and quit clean up all windows.
R/Q/Esc work in every view, including all process panels. Each auxiliary camera
is retained when redisplaying the same Coarse query. Rendering reads
stored results without model fitting, support reassignment or result mutation.
Only Coarse picks queries; its heatmap and cached scores are unchanged.
`stage_elapsed_s` reports local voxel preparation, RANSAC, support, edges,
vertices and inclusive geometry duration; `elapsed_s` is the current pipeline
call total. Geometry is an inclusive subtotal, not an additional disjoint stage.
No performance claim or per-stage benchmark was measured in this task.

**Reference and limits.** Inspected the official CornerVINS
[`RunBorderDetection` / `ComputeIntersectionLine`](https://github.com/zydddd/CornerVINS/blob/main/src/plane_features/plane_detector.cpp)
for plane-pair connectivity and finite hull-overlap intervals. The implementation
here uses a general least-squares line point, overlapping core/halo memberships,
observed boundary edges and two-edge corner acceptance; it does not copy the
three-orthogonal-plane/SO(3) box refinement or the axis-fixed line solution.
Tolerance-strip clipping reconstructs the geometric principle in Zhang et al.,
CornerVINS, Section V-B / Algorithm 1 / Fig. 4(a–c),
[IEEE TRO 2025](https://doi.org/10.1109/TRO.2025.3567532), using the local
`ref_paper/` PDF. The public `ComputeEndPoints` call supplies a tolerance, but its
definition is not reproduced here; this is not claimed to be an identical port.
The existing fine-mode finite-segment matcher/refiner was reviewed but is not
called with native RANSAC IDs or replaced by this path.
Convex hulls cannot reconstruct concavity; occlusion boundaries, missing boundary
samples, noise bridges and centroid discretization remain limitations. A centre
gate reduces ROI-cut influence but does not prove physical boundary existence.
Without edge/vertex point-support validation, hull bridges and tolerance-recovered
segments may be accepted without nearby observations; ACCEPT now certifies the
remaining geometric gates, not measured point support.
**Selected-query execution and process display.** Validator runs only after
selecting one existing Coarse NMS query. Automatic all-query traversal and its
aggregate scene window are removed. Window 4 contains only the selected
query's ROI debugging. The canonical per-query pipeline, native/scene
index adapters, graph masks and LS equations remain unchanged. There is no
cross-query result accumulation or new NMS.

Window 6 has four panels in a 2-by-2 layout for that one stored query result:

1. Original boundary/shared edges before masking, using `all_edges` including
   fully masked sources. Boundary RGB follows plane IDs; shared edges are cyan.
   Gray hull rims provide local plane context without background point clouds.
2. Active retained intervals in gray/cyan and removed boundary intervals as
   magenta dashes. Original TLS endpoints and numerical masks are never mutated.
   A blue translucent cylinder (alpha 0.16) with radius
   `edge_overlap_distance_mm` spans each finite shared edge, with blue end rings.
   This illustrates the infinite-line perpendicular-distance gate over the finite
   interval used for overlap, not a requirement that boundary endpoints lie
   between cylinder caps. Plane membership, direction angle and minimum finite
   overlap remain separate conditions. `edge_overlap_settings` snapshots the
   actual distance/angle/minimum-length settings for the viewer; the caption
   reports them without substituting the hull-intersection tolerance-strip width.
3. Active geometry and the final vertex calculation. Yellow dashes show only
   extrapolation actually needed by each stored refined vertex's incident Edges,
   not the full configured endpoint allowance. For unit edge direction u and
   stored vertex p, compute the display parameter `t = (p - start) dot u`.
   Draw `[t, 0]` when t is negative, or `[length, t]` when t exceeds the length;
   interior projections have no extension. Thin gray links distinguish a
   nonzero refined-vertex/line residual from extension along the edge axis.
   Short extensions use proportionally short dashes. Small filled vertex markers
   with wire halos avoid hiding these intervals. No new intersections, numerical
   endpoint changes or validation gates are introduced. White dots show stored pair candidates; gray links
   show movement to each stored refined LS estimate. Accepted vertices are
   green, gated-out vertices red, and the selected corner has a larger marker
   and thicker green incident edges. These are diagnostics, not new intersections
   or support evidence. Captions report candidate/vertex counts and final status.
4. Only the final selected Corner and its contributing Edges over the original
   unvoxelized input cloud in neutral gray. Draw contributing Boundary intervals
   and the selected marker/halo in green, and contributing shared intersection
   intervals in cyan, with the selected Corner's actually
   used extrapolation in yellow dashes and any projection residual in thin gray.
   Omit other vertices, candidates, Hulls and noncontributing Edges. Upload the
   raw cloud once through the manager/display-worker startup path; later query
   updates replace only overlay geometry. Initial camera framing stays local to
   the selected query although the background contains the complete raw scene.
   This display never validates additional NMS queries. Missing optional raw data
   is labelled unavailable and is never replaced by Coarse voxels under a raw
   label. A query without a selected Corner shows only the background and status.
   `corner_edges` remains the sole contributor list. Report each type's count
   and IDs; explicitly label a Boundary-only Corner when shared contributors
   are absent. Do not add unrelated shared Edges or change numerical selection
   to force an intersection into this display.

The four panels share identical initial framing for direct comparison and keep
independent cameras when redisplaying the same query. All read stored results
through the existing manager/worker. R/Q/Esc and auxiliary-window cleanup remain.
Main panel 3 and window 5 retain their roles. Window 4 replaces the old
per-plane Boundary display with the former window 7's A: padded Hull, B: ROI
clipping, C: TLS/endpoint inheritance with Line IDs and maximum fitting errors,
and D: core-only Before versus padded/clipped After. A uses gray plane support,
green Hull/Nodes, blue dashed core and orange dashed padded spheres. B uses blue
retained Edges/active Nodes, orange inactive X marks, gray dashed original Hull
and light-red dashed removed portions. C uses thin gray input, thick blue final
Lines, blue active endpoints, orange inactive X marks and gray absorbed Nodes.
D shares initial framing and synchronizes camera interaction between its two
panels. Captions report clipped active/inactive Node counts, retained Edges,
final Lines/endpoint states and post-mask Corner candidate counts. Only C draws
3-D Line/error labels. Store the handles returned by SceneWidget.add_3d_label
in each display panel and remove them individually with remove_3d_label on
query updates, without requiring a SceneWidget.clear_3d_labels method. These
display caches never mutate numerical results. The pipeline computes the core-only support/Hull/graph
comparison once from identical stored models and core voxels; there is no second
RANSAC, alternate query or influence on final selected Corner. This additional
Hull/TLS/graph work is timed separately as `roi_debug_before` and contributes to
total geometry time. The renderer never recomputes numerical geometry.
S captures window 4's A/B/C PNGs and a paired `D_before_after.png` under
`outputs/validator_roi/query_<id>_<timestamp>/`, including counts, legends and
C's Line/error listing. No capture has been generated or actual GUI/device run
performed during implementation; tests were explicitly excluded by the user.
At the user's request, instruction-section-15 synthetic geometry experiments
were not executed, and the existing unittest suite was not run during this task.
A numerical one-plane/two-boundary-edge regression fixture was added but left
unexecuted; the prior plane-only viewer fixture explicitly disables geometry.
One-/two-/three-plane acceptance and measured edge/corner
accuracy remain unverified; syntax/import/CLI checks do not establish these.

### 1.3 Whole-scene total composition

`--mode total` explicitly enables the whole-scene path without changing CLI
`detector` (NMS only) or interactive `validator` (one selected query). Reuse the
same detector settings, CED/bilateral/PCA/Fusion/threshold/3-D NMS equations,
then invoke the canonical guided validator independently for every NMS scene ID.
For each ACCEPT result retain exactly its existing selected Corner and incident
`corner_edges`, including original Boundary/shared Edge types and active finite
intervals. Return aggregate `final_corners` and `final_edges`, per-query compact
status/provenance/timing records and detector scores. Query-local Edge and Corner
IDs are qualified by `query_index`. Do not average, snap or merge different
queries' results, add a final NMS, or force shared Edges into Boundary-only
Corners. Nearby queries may therefore produce redundant geometry.

Total uses `ValidatorDetectorConfig(mode="total")` and the same resolved guided
parameters. No new detection hyperparameters are added. Geometry must be enabled.
All raw/Coarse lengths remain mm, and the crop-before-local-voxel, separate
core/halo, source index and seeded RANSAC contracts are unchanged.

Preparation builds raw/Coarse KD-trees once. The Coarse tree is shared by dense
scoring/NMS and nearest-scene adapters; the raw tree is reused across all seed
ROIs. Total preprocessing omits the unused organized pixel/scene map. The shared
pipeline's optional `include_debug=False` skips core-only comparison support,
Hull, TLS, graph and Corner work, setting `roi_debug_before=None` and
`debug_comparison_computed=False`; its timer is zero because it did not run.
Default interactive calls retain that comparison. Guided batch mode with debug
disabled stores compact records rather than accumulating raw/local point clouds
and intermediate graphs. Finite/shape/index validation remains; when arrays are
the identical cached Coarse object, the redundant full equality comparison is
omitted. Dense logs omit individual seed dumps in total mode. No numerical
RANSAC, residual, connectivity, TLS, strip, mask, angle, LS or gate computation
is approximated. Progress and prepare/detector/validator times are reported.

The manager opens exactly two result windows together. The first is the detector
Fusion heatmap with the unchanged full-scene Q99 normalization, invalid-gray
mask and red NMS markers. The second shows original raw XYZ in gray (0.38),
final Corners in red, Boundary contributors in green, shared contributors in cyan, actual
extrapolation in yellow dashes and residual links in thin gray. Show only the
existing selected Corner per accepted query, without intermediate geometry.
Render Edge bars only over active intervals, using the existing segment-cylinder
helper. Edge radius is max(0.5, 0.15 times validator voxel) mm and Corner radius
max(2, 0.8 times validator voxel) mm: 0.75/4 mm at a 5 mm voxel. Combine meshes
by colour before adding them to the scene to bound draw-call count. Both views
remain interactive; R recaptures, Q/Esc closes both, and closing one leaves the
other active. There are no intermediate validator/Tk/RGB windows. Headless
mode computes the same results; existing source acquisition is unchanged.

Only Python syntax was checked. No test suite, synthetic data, device/GUI run,
benchmark or speedup measurement was performed; the optimization claims concern
removed/reused work, not a measured runtime multiplier.

## 2. Scene and organized-image indexing

For fine/detector, the working scene is the 3-D-voxelized array $P=\{p_i\}_{i=1}^M$. Every query,
plane support, boundary support, edge support, and provenance record uses this
scene-index domain. The organized depth image retains raw camera-space samples
$X(u,v)$. Each valid pixel is mapped to its nearest voxel-scene sample only when
the final membership is adapted for downstream stages; AHC and fitting use the
raw organized samples.

For query $q=p_k$ and radius $R_q$, the extraction mask is exactly

$$
\mathcal I_q=\{(u,v):X(u,v)\text{ valid and }\|X(u,v)-q\|_2\le R_q\}.
$$

The inequality is inclusive. Full-scene points and the later projection halo
do not participate in segmentation, RANSAC, refitting, merging, or hulls. Pixel
adjacency always comes from the depth-image grid; no KNN organization fallback
exists.

## 3. Organized AHC segmentation

The clean-room implementation follows the published CornerVINS/Feng AHC
algorithm structure. It starts from square image blocks of side $b$ pixels.
A block is rejected if it contains a missing pixel, touches a depth
discontinuity, or has excessive plane-fit MSE. The default depth-dependent
threshold is

$$
T_{\mathrm{MSE}}(z,\epsilon)=(1.6\times10^{-6}z^2+\epsilon)^2,
$$

with $\epsilon_{\mathrm{init}}=5$ mm and $\epsilon_{\mathrm{merge}}=8$ mm.
Adjacent regions are contracted in minimum merged-MSE order. Refinement applies
block erosion, adjacency-constrained pixel growing with squared point-to-plane
residual at most $9\,\mathrm{MSE}$, and a final adjacency-constrained merge.

The block size is an image-space quantity. For a fronto-parallel surface at
depth $z$ and focal lengths $(f_x,f_y)$ in pixels, its approximate metric
footprint is

$$
L_x\simeq bz/f_x,\qquad L_y\simeq bz/f_y.
$$

Therefore a 3-pixel block does not correspond to one fixed millimetre size.

## 4. Region filtering, RANSAC, refit, merge, and hull

The paper-sized region threshold is 20,000 pixels. The active `roi_test` policy
uses 100 pixels for query-local Kinect data and records this adaptation in
diagnostics.

Each retained AHC region is voxel-subsampled at 50 mm only to create
deterministic RANSAC hypotheses. Every raw full-resolution region sample is
then reclassified using the 15 mm point-to-plane gate. A region is retained
when at least 90% of its samples are inliers, after which its plane is refit by
least squares on those inliers.

Two patches are merge-compatible when their unoriented normal angle is strictly
below 5 degrees and their symmetric separation is strictly below 20 mm. The
separation is the maximum of the two centroid-to-other-plane residuals. After
every merge the combined samples pass through the same RANSAC/refit procedure.
Final inliers are projected into a plane basis, a 2-D convex hull is computed,
and the hull is lifted back to 3-D.

The mutable organized context caches extraction by `(query_index,
CornerVINSPlaneConfig, query_radius_mm)`.

## 5. Padded per-plane projection and boundary

Plane extraction remains query-core-only. The downstream boundary stage may
inspect a larger observed context. Its outer radius is
$R_{\mathrm{proj}}=R_q+h_{\mathrm{proj}}$, where the active default halo is
$h_{\mathrm{proj}}=50$ mm (100 mm query core, 150 mm maximum projection
radius). In the automatic CornerVINS path, observed halo points first pass the
same 15 mm point-to-plane residual gate used by the downstream plane refit;
they do not alter AHC membership or the fitted CornerVINS plane. For paths with
valid normals, a sample gets

$$
w_i^n=k_n\!\left(\frac{\arccos(|n_i^\top n|)}{\sigma_{n,\mathrm{proj}}}\right),
\qquad
w_i^r=\frac{1}{1+(((p_i-c)^\top n)/\sigma_{r,\mathrm{proj}})^2},
$$

$$
w_i^{\mathrm{proj}}=w_i^n w_i^r.
$$

Only samples passing the configured projection weight gate enter the tangent
projection. The plane's fitted query-core members remain the immutable plane
support; halo samples only supply neighborhoods and boundary context.

`corner_2d.estimate_boundary_uv` owns continuous boundary evidence. It uses
the padded tangent samples for neighborhoods and density while classifying the
core evaluation mask.

## 6. Weighted Hough and 3-D finite edges

Each boundary voter contributes its continuous weight to every sampled Hough
angle. A cell is eligible only when both conditions hold:

$$
N_{\mathrm{votes}}\ge N_{\min},\qquad
S_{\mathrm{weighted}}/N_{\mathrm{votes}}\ge \bar w_{\min}.
$$

The configured weighted-score gate may be fixed, peak-relative, global-mean,
or off. Peak suppression is independent in angular and rho coordinates and
handles the periodic equivalence $(\rho,\theta)\equiv(-\rho,\theta+\pi)$.
Selected cells are only seeds for the following continuous stage; the stored
accumulator and its NMS are not modified by refinement.

### 6.1 Continuous 2-D line refinement

`line_refinement_2d.py` refines each Hough seed independently. The only
optimization variables are the infinite-line offset and angle. Relative to the
fixed projected ROI centre $o$,

$$
d_i=n(\theta)^\top(p_i-o)-\rho,\qquad
s_i=t(\theta)^\top(p_i-o),
$$

where $n=(\cos\theta,\sin\theta)$ and
$t=(-\sin\theta,\cos\theta)$. Both choices of the inside sign are optimized;
the lower-loss result is retained. For each sampled along-line coordinate,

$$
B(s)=\sum_i w_i G_{\sigma_d}(d_i)G_{\sigma_s}(s-s_i)
       \operatorname{sigmoid}(d_i/\delta),
$$

$$
O(s)=\sum_i G_h(d_i)G_{\sigma_s}(s-s_i)
       \operatorname{sigmoid}(-d_i/\delta),
$$

$$
q(s)=(1-e^{-B(s)/c_B})e^{-O(s)/c_O},\qquad
\mathcal L=-\int_{-R}^{R}q(s)(G_{\sigma_s}*q)(s)\,ds.
$$

All kernels and sigmoids are continuous in $(\rho,\theta)$. Point spacing,
seed residuals, and the fixed ROI extent determine
$\sigma_d,\sigma_s,h,\delta,c_B,c_O,R$ once per projected plane, so every seed
on that plane shares one score scale. L-BFGS-B is attempted first; a failed,
non-finite, or non-improving result is compared with bounded Powell. Initial
and final parameters and losses, convergence status, inside sign, iteration
count, optimizer, and the full two-parameter path are retained.

Endpoints are not optimization variables. After refinement, Otsu thresholding
of the smoothed support profile selects the longest valid continuous interval.
Candidates failing observed boundary-support count, minimum length, or positive
support score are rejected. No cross-candidate `mean_quality` threshold is
applied before final finite-segment NMS. That NMS requires compatible direction angle,
symmetric perpendicular line distance, and projected interval overlap; hence
separated segments on the same infinite line do not suppress each other.

Every accepted finite 2-D segment preserves its original scene IDs. Those
observed 3-D points are fitted once with PCA/TLS; no synthetic back-projection
or random 3-D line sampling is used. The recovered direction must agree with
the lifted refined 2-D line within the configured tolerance.

Finite edges merge by complete-link compatibility over direction angle,
symmetric line offset, and observed interval gap. Plane adjacency comes from
the plane IDs carried by merged edges.

## 7. Graph-local refinement

`refiner.py` is the only numerical post-processing layer after the immutable
Hough/PCA result. It works per connected plane-edge component. Planes minimize
robust point-to-plane residuals; edges minimize confidence-weighted refined
plane-incidence and stored 2-D boundary-line residuals. Huber is the default
loss. Intersections of refined finite edges are clustered into graph-local
corners. Initial and refined parameters and objective diagnostics are both
retained.

## 8. Dense detector front end

Detector mode computes strict-radius centroid-deviation (CED), bilateral CED,
and CED-weighted PCA. With positive original CED weights $w_j$,

$$
\bar p_i^w=\frac{\sum_jw_jp_j}{\sum_jw_j},\qquad
C_i^w=\frac{\sum_jw_j(p_j-\bar p_i^w)(p_j-\bar p_i^w)^\top}{\sum_jw_j}.
$$

For $\lambda_1\ge\lambda_2\ge\lambda_3\ge0$,

$$
r_i=\operatorname{clip}(\lambda_2/\lambda_1,0,1),\qquad
\widetilde r_i=\tau\tanh(r_i/\tau),\qquad
F_i=\operatorname{clip}(s_i^B\widetilde r_i,0,1).
$$

Omitting $\tau$ uses $\widetilde r_i=r_i$. Fusion thresholding is fixed or the
global mean over finite valid scores. Eligibility is strict, $F_i>T$.
Inclusive-radius 3-D NMS retains a point only when no eligible neighbor has a
larger score; equal scores favor the smaller scene index. CLI detector output stops at these unverified candidates. Explicit library validation passes NMS survivors to the same CornerVINS validator pipeline.

## 9. Default parameter mapping

| Setting | Meaning | Default |
|---|---|---:|
| `voxel_3d_mm` | scene voxel | 10 mm |
| `voxel_2d_mm` | 2-D scale base | 10 mm |
| `query_radius_x * voxel_2d_mm` | exact plane ROI $R_q$ | 100 mm |
| `cornervins_ahc_block_size` | AHC block side | 3 pixels |
| `cornervins_ahc_mse_threshold_mm2` | fixed MSE override | `None` (depth model) |
| `cornervins_region_mode` | region policy | `roi_test` |
| `cornervins_min_region_points_roi` | local minimum region | 100 pixels |
| `cornervins_downsample_leaf_mm` | RANSAC hypothesis voxel | 50 mm |
| `cornervins_ransac_threshold_mm` | full-resolution inlier gate | 15 mm |
| `cornervins_ransac_min_inlier_ratio` | accepted inlier ratio | 0.90 |
| `cornervins_merge_angle_deg` | strict merge angle | 5 degrees |
| `cornervins_merge_distance_mm` | strict symmetric separation | 20 mm |
| `projection_halo_mm` | extra context beyond query core | 50 mm |
| `boundary_radius_x * voxel_2d_mm` | boundary radius | 90 mm |
| `boundary_epsilon_x * voxel_2d_mm` | boundary epsilon | 25 mm |

`settings.py` groups defaults, resolver/config definitions, and CLI help by
consumer: shared input/preprocessing, detector, validator, fine, and ball.
The validator section owns `validator_fine_*` and guided RANSAC settings;
its Fine stage is distinct from `--mode fine`. The fine section retains the
existing `ValidatorConfig` and resolver names for API compatibility and follows
query geometry, CornerVINS extraction, padded projection/boundary, weighted
Hough, continuous 2-D line refinement, 3-D line restoration/merging, and
graph-local refinement. Shared CED support settings (`min_neighbors` and
`centroid_radius_x`) are listed under detector and remain consumed by fine;
validator and ball reuse them for their Coarse detector queries. Only declaration
and help-group order changed; field names, defaults, validation, CLI options and
unit conversions are preserved. The former multiscale-Harris,
SAH-graph, and independent Edge3D diagnostic settings have been removed.
Detector settings end at dense NMS and default to `validator=None`. An explicitly supplied shared `ValidatorConfig` validates every retained candidate. The superseded detector-only K-subspaces
plane-verifier implementation and its private diagnostics have been removed.

## 10. Visualization semantics

Fine mode (`--mode fine`) enters the frozen, aligned RGB/depth Tk window directly. A
left-click selects the query and another click recomputes it in the same
window; no separate preliminary query picker is required.

Every successful query prepares four non-blocking 3-D views, whether or not
the edge graph is enabled:

1. full scene plus query-core coarse AHC labels after Feng Algorithm 3;
2. full scene plus refined AHC labels after Feng Algorithm 4;
3. full scene plus pre-merge RANSAC-accepted planes and normals;
4. full scene plus final merged planes and convex hull boundaries.

All four show dark outside-ROI context, light-gray ROI samples, a yellow exact
query-radius wire sphere, and a red query marker. Visualization reads stored
results and never recomputes or mutates numerical output.

## 11. Limitations

1. The primary plane extractor requires organized depth, intrinsics, and a
   pixel-to-point map.
2. Missing pixels invalidate initial AHC blocks; there is no unorganized fallback.
3. The active `roi_test` minimum-region threshold and 15 mm RANSAC gate are
   Kinect/query-local adaptations, not claimed CornerVINS paper constants.
4. Hough voting is not orientation-guided.
5. Refined three-plane structures are optimized pairwise rather than as one
   global latent line.
6. Detector results from separate NMS queries are not scene-globally deduplicated.

## 12. Implementation traceability

| Stage | Active implementation |
|---|---|
| Dense CED/bilateral/PCA | `corner_detector/edge_suppression.py` |
| Fusion threshold and NMS | `corner_detector/candidate_extraction.py` |
| Detector composition | `corner_detector/dense_heatmap.py` |
| Query-local extremity/vertex peak NMS | `corner_detector/extremity.py` |
| CED/click extremity adapters | `corner_detector/ball_query.py` |
| Shared validator entry | `corner_detector/validator_pipeline.py` |
| Organized mapping | `corner_detector/preprocessing.py` |
| AHC/RANSAC/merge/hulls | `corner_detector/cornervins_plane_extraction.py` |
| Projection and boundary | `corner_detector/corner_2d.py`, `corner_detector/boundary_support.py` |
| Hough/3-D edges/merge | `corner_detector/plane_edge_graph_pipeline.py` |
| Continuous 2-D line refinement/segment NMS | `corner_detector/line_refinement_2d.py` |
| Robust refinement/corners | `corner_detector/refiner.py` |
| Settings and units | `corner_detector/settings.py` |
| Visualization ownership | `corner_detector/visualization/manager.py` |

## 13. Query-local extremity vertex hypotheses

This optional path preserves every existing CED/bilateral/PCA/threshold/NMS
equation, default and candidate location. `--mode extremity256` runs the original
detector candidate composition, takes its NMS scene indices verbatim, then
evaluates each query independently. It does not run the old detector's
post-NMS validator automatically. `--mode detector` also stops at NMS; it does not run the local extremity stage.
`--mode ball` is the clicked inspection UI for the same CED-fed local module.
Neither path introduces FPS, query relocation, plane clustering, projection or
Hough. The module generates unverified vertex hypotheses, not confirmed corners.

Let $q=P_k$ be an unchanged voxel-scene query, $v_3$ the 3-D voxel length in mm,
and $X$ the original raw XYZ. Core radii and padding are resolved as
$r=a_r v_3$ and $\delta=a_\delta v_3$. Research reference values $a_r=3$ and $a_\delta=1$
reproduce $r=30$ mm and $\delta=10$ mm at the default $v_3=10$ mm.
The search support is

$$
S_r(q)=\{i:\|X_i-q\|_2\le r+\delta\},\qquad
C_r(q)=\{i:\|X_i-q\|_2\le r\}.
$$

All ball defaults are declared together in the `ball_*` block of
`settings.SettingsDefaults`, alongside settings such as `query_radius_x`.
The CLI, `settings.ExtremityConfig`, default library calls and synthetic base
configuration all derive their defaults from this block. The original
`extremity.ExtremityConfig` import remains an alias for library compatibility.
The four length options are `--ball-radii-x`, `--ball-padding-x`,
`--ball-cluster-radius-x`, and `--ball-nms-radius-x`, all relative to
`voxel_3d_mm`, not `voxel_2d_mm`. Their `SettingsDefaults` fields have the same
`ball_*_x` names. The old CLI `--ball-*-mm` length options are removed rather
than reinterpreted as multipliers. Resolved `ExtremityConfig` fields and all
algorithm/API coordinates remain mm, including explicit library overrides.
Voxel preprocessing stays at the existing 10 mm default in all modes;
only the query/validator index adapter uses those centroids, while votes use raw
observations. A clicked raw pixel chooses the nearest existing CED NMS query;
both modes compute the CED queries and their independent local maps once per
frame. Clicking only changes the inspected result. Manual GT-near queries are
used exclusively by the synthetic experiment, never introduced into the live
CED pipeline.

Optional object components are connected components of the undirected raw-point
radius graph, with inclusive edge distance
`ball_cluster_radius_x * voxel_3d_mm` (research reference 0.45, or 4.5 mm at a 10 mm voxel).
They are computed on the
maximum padded query ROI once, then reused for all scales. They do not fit or
separate planes, and each component receives a separate sweep, even if it does
not contain $q$. This prevents a disconnected larger component from defeating
a smaller one in the argmax. It does not guarantee correct instances: contact,
noise bridges and sparse sampling cause merges/fragments. No minimum-size
filter silently removes small components; singleton fragments can be strong
false positives. The OFF ablation uses one component.

For $J=256$, midpoint Fibonacci directions use

$$
z_j=1-2(j+\tfrac12)/J,\quad \theta_j=j\pi(3-\sqrt5),\quad
d_j=(\sqrt{1-z_j^2}\cos\theta_j,\sqrt{1-z_j^2}\sin\theta_j,z_j).
$$

For each component $g$ and each scale, one NumPy matrix product evaluates every
inner product. There is no physical cloud rotation:

$$
i^*_{r,g,j}=\arg\max_{i\in S_r(q)\cap g} d_j^\top(X_i-q),\quad
v_{r,i}=\sum_{g,j}\mathbf1[i=i^*_{r,g,j}]\mathbf1[i\in C_r(q)].
$$

The `hard` voting backend preserves the following original equations. Raw scene IDs are sorted before argmax, so exact projection ties choose the
smaller raw input index. An outer-padding winner discards that direction; the
runner-up inside the core does not receive a vote. One map is stored for each
query, with no cross-query vote aggregation. Raw vote counts are always retained.
For one scale, normalized score is $v_{r,i}/J$. For the optional $(12,30)$ mm
multi-scale path,

$$
v_i=\sum_r v_{r,i},\quad e_i=\sum_r\mathbf1[i\in C_r(q)],\quad
s_i=\begin{cases}v_i/(Je_i),&e_i>0,\\0,&\text{otherwise}.\end{cases}
$$

This avoids an automatic score boost from the number of eligible scales, but a
peak seen only at the small scale can be diluted by nonselection at a larger
scale. It is not a maximum-across-scales rule. `scale_support_count` records
how many scales actually voted for a point.

Points with positive votes and $s_i\ge0.04$ enter inclusive 3-D peak NMS with
radius `ball_nms_radius_x * voxel_3d_mm` (research reference 0.5, or 5 mm at a 10 mm voxel).
The `ball_nms_method="local_maximum"` rule rejects a point if any eligible same-component
neighbor has a higher score, or equal score and a smaller raw index. This is
the pairwise plateau rule: a chain of tied neighbors may leave only its earliest
peak even if the chain exceeds the radius. Optional `ball_nms_method="greedy"` NMS processes
descending score/smaller index and suppresses only from retained peaks. Neither
method averages coordinates, imposes Top-1, merges different components or
suppresses across separate query maps.

Each hypothesis stores its raw observed position, vote count, normalized score,
local object component, scale support, unchanged query ID/position, vector
offset and distance from that query. `point_index`/`candidate_indices` are
nearest-voxel indices for the existing validator, while `raw_point_index` and
`candidate_raw_indices` explicitly address original raw XYZ. The mapping distance
is retained; coarse voxelization may map multiple hypotheses to one scene ID.
`validator_seed_indices` deduplicates only those scene IDs. To validate them,
call the canonical `run_validator_candidates` with the original voxel scene,
validator config and organized context. Voting itself asserts no geometric
verification and cannot recover an unobserved vertex.

Diagnostics read stored maps. Every inspected ball query opens three separate
side-by-side Open3D windows in stage order:

1. Local object-component memberships in categorical colors.
2. The full pre-NMS `raw_votes` map, including zero and sub-threshold votes.
   Zero votes are neutral gray; positive raw vote mass is linearly colored from
   zero through its maximum, using color level $v_i/\max_k v_k$. The title/terminal report the
   range and zero fraction. Constant/empty maps are handled without division
   by zero. There is no integer cast of fractional soft votes, threshold or
   NMS display mask. This display scaling leaves stored scores, peaks and
   thresholds unchanged; small votes may share similar colors when peaks dominate.
3. The stored final NMS hypotheses, marked magenta against a neutral gray ROI.

Before any click, a separate frame-wide overview uses the detector's original
voxel-scene Fusion heatmap as its background, with the identical fusion-valid
mask, linear zero-to-valid-Q99 range and Turbo colors used by its NMS panel.
It reads the existing `ced_score_result`, without score recomputation or
interpolation onto raw points. All unchanged CED/PCA/NMS voxel queries appear
in red, and all cached per-query Ball NMS hypotheses retain their observed raw
coordinates in magenta. Both overview and click inspection consume the same
cached results and `cfg.extremity`; no per-view overrides are introduced.
Live Ball query markers have a 2.4 mm display radius and hypothesis markers
a 3 mm display radius in both overview and local views; these are independent
of the unchanged detection/NMS radii and never move the stored coordinates.
Upstream `nms_radius_mm` and local `ball_nms_radius_x * voxel_3d_mm` remain
independent parameters for their respective stages. Repeated identical
raw IDs share a display marker only; no spatial merging, cross-query NMS,
vote aggregation or extra validation is performed. The full per-query records
remain unchanged. This overview is serviced independently, survives query
reselection, and closes on recapture/quit; `--no-3d-view` suppresses it.

All three preserve raw scene XYZ, dark outside-ROI context, the red unchanged
query, and yellow core/gray padded wire balls. Candidate markers appear only
in window 3, so they never obscure the raw vote map. Windows are polled and
closed independently; reselection replaces all three without recomputing votes.
The old C/H color switch and automatic ball 2-D summary window are removed.
No numerical detector, vote threshold, normalized score or NMS setting changes.
Synthetic PNG/HTML exports retain their separate normalized-score diagnostics
and green GT markers; the optional 2-D score/offset figure remains available
for headless inspection, but does not open automatically in ball mode.

### Per-point direction-distribution inspector

Selecting an existing CED query additionally opens a Tk-embedded figure with
two rotatable 3-D panels. The left panel displays every stored raw ROI point
in query-relative mm coordinates, with its unmodified raw-vote heatmap,
magenta hypothesis markers, red query and core/padded wire balls. Wire and ray
collections are initialized with valid 3-D segments before being added to the
axes, avoiding empty-collection autoscale errors in Matplotlib. A point click
selects the nearest projected raw sample within eight screen pixels; exact
screen-distance ties prefer nearer projected depth, then smaller raw ID.
Camera rotation/zoom changes only this visual picking projection, never
detector coordinates or numerical support. Dragging more than four pixels
does not count as point selection. This picking projection is not a detector
plane/2-D projection stage.

For selected raw ROI index $i$ and scale $r$, the read-only diagnostic uses
the same cached component membership and padded support $S_{r,c_i}(q)$ as
voting, including padding points. It preserves the configured Fibonacci
directions and projection dtype and evaluates

$$
t_j=Q_{0.99}^{\rm linear}\{d_j^\top(X_k-q):k\in S_{r,c_i}(q)\},\qquad
b_{ji}=\mathbf1[d_j^\top(X_i-q)\ge t_j].
$$

The right panel places all configured directions (normally 256) on a unit
sphere in camera XYZ axes. Its default **Contribution heatmap** assigns each
direction its actual post-normalization, post-core-gating mass $c_{rji}$ for
the selected point. For hard voting this is the same argmax indicator used
by detection, including its smaller-raw-ID tie rule. For soft voting let
$h_{rji}$ be its sigmoid weight, with the configured support cutoff applied
only in sparse mode. Then

$$
c_{rji}=\mathbf1[i\in C_r(q)]\,
\frac{h_{rji}}{\sum_{k\in S_{r,c_i}(q)}h_{rjk}},\qquad
\sum_j c_{rji}=v_{r,i},\qquad
\Delta S_{rji}=\frac{c_{rji}}{J e_i}.
$$

Mass is normalized across search points for each direction, never across
directions for the selected point. The heatmap uses the same zero-gray,
linear Turbo mapping as raw-vote diagnostics, with a numeric per-point
range $[0,\max_j c_{rji}]$. Identical colors across different selected points
can represent different magnitudes; the colorbar and direction-mass sum
make that distinction explicit. All-zero heatmaps are gray and hide the
colorbar. The 256 samples are not interpolated onto new voting directions.
Padding/outside-search points receive zero actual mass. The UI reports the
reconstructed scale sum alongside its stored scale vote and total V/S;
small float32/accumulation differences are not coordinate or score changes.

The **Top 1% directions** menu option retains the earlier binary display:
gold points/rays mark $b_{ji}=1$; other directions are gray. This threshold
membership is distinct from the heatmap's continuous soft contributions,
direction normalization, the sparse cutoff and NMS eligibility. Q99 ties are
inclusive; small or tied supports can therefore include more than 1% of
their points. The selected point need not pass score thresholding or NMS.
Padding points can have gold directions but are explicitly labelled as
ineligible for output votes. Outside a selected scale's search radius the
diagnostic is labelled unevaluated, with no highlighted directions and no
threshold/projection arrays; multi-scale inspection defaults to the largest
radius and permits switching scales without moving the point.

The last inspected point retains its raw/ROI/query IDs, observed coordinates,
component, scale, support count, core/search status, direction vectors,
per-direction projection/threshold arrays, boolean membership mask, actual
`direction_votes` and `direction_scores` arrays in the
visual diagnostic state. Stored total raw vote and normalized score are
reported separately (across all configured scales). Soft diagnostics use
blocks of 64 directions; hard diagnostics retain the production full
point-by-direction multiplication shape to preserve argmax ties exactly.
Changing display mode reuses these arrays without projection/weight
recomputation. CED, clustering, production voting totals and NMS are not rerun
or modified. No new corner score or FP rejection rule is added.
The inspector is replaced on query reselection and closed on recapture/quit;
`--no-3d-view` suppresses it with the other 3-D diagnostics.

`python3 -m corner_detector.extremity_synthetic` uses the original CED settings
and identical noisy raw scenes for A (clustering/padding OFF), B (padding ON),
C (components/padding ON), D (C plus 12/30 mm scales), and ideal-label C/D
references. CED-produced and fixed-GT-near voxel queries are reported separately.
The default experiment runs 20 seeds on 16 scenes, base 3 mm/0.12 mm noise,
0.5 mm noise, and 6 mm sampling. It includes a genuine pinhole partial-depth
box observation. All observed finite physical patch vertices are listed as GT;
per-query recall uses only GT in that query's maximum core. Scene recall uses
all GT and reveals missing CED coverage. Coverage uses literal GT centres
inside the core, while 5 mm-tolerance matching may also credit a nearby GT just
outside it; that coverage fraction is not a strict tolerance-based recall
ceiling. Matching maximizes one-to-one matches
within 5 mm before minimizing their distances. Exact raw ID deduplication is
used only for scene metrics; adjacent hypotheses from different queries remain
distinct and may count as false positives. Localization errors are matched-only.

Reported timing separates CED scoring/NMS, preprocessing, shared spatial-index
preparation, local ROI, components, voting, aggregation/NMS, and added total
refinement. Generation, metrics, I/O and visualization are excluded. Means,
medians and P95 plus CPU/thread settings are exported. Repeatability is a
one-to-one candidate-set match to seed 0 divided by the smaller candidate count;
empty sets and seed 0 itself are excluded, so stable false positives are still
possible. This module adds time to CED and must not be described as a faster
replacement detector. Convex-hull-interior concave vertices, contact merges,
unobserved vertices and missing upstream queries are structural limitations.

### 13.1 Quantile soft voting and controlled comparison

The local module exposes four `ball_voting_mode` backends, documented beside
the setting in `SettingsDefaults`. `hard` preserves its original integer votes. `soft_dense`
is the literal quantile/sigmoid reference; `soft_partition` retains the same
equations using a specialized quantile; `soft_sparse` is an explicitly
approximate support-cutoff experiment. Existing user ROI/component/NMS settings
are not reset. `ball_voting_dtype` selects float64 (default) or float32 for
projection and weighting; final coordinates retain the original raw samples.

Within each scale/component search support $S$, soft voting uses

$$
s_{ji}=d_j^\top(X_i-q),\quad t_j=Q_{0.99}^{\rm linear}(s_{j,:}),\quad
w_{ji}=\operatorname{sigmoid}((s_{ji}-t_j)/\tau),\quad
\hat w_{ji}=w_{ji}/\sum_{k\in S}w_{jk}.
$$

Only after the **entire padded support** is normalized is the core gate applied:
$v_{r,i}=\mathbf1[i\in C_r]\sum_j\hat w_{ji}$. Padding mass is discarded,
not reassigned to the remaining core. The existing multi-scale score
$s_i=\sum_r v_{r,i}/(J e_i)$ and multi-peak local NMS still apply. There is no
Gaussian smoothing, weighted coordinate mean, synthetic vertex, extra query
or geometric refinement. `raw_votes` and candidate `raw_votes` become floating
vote mass for soft modes; `discarded_vote_mass` records discarded outer mass.
The hard-only integer `discarded_vote_count` is `None` in soft modes.

`ball_soft_tau_x` multiplies `voxel_3d_mm` into `ExtremityConfig.soft_tau_mm`.
Its default 0.025 corresponds to 0.25 mm at a 10 mm voxel. Quantile probability
is fixed at 0.99. `ball_soft_alpha` defaults to 4; sparse support is
$s_{ji}\ge t_j-\alpha\tau$, normalized on that truncated support before the
same core gate. Omitted tail weights can accumulate across many points. A
larger alpha is not assumed to preserve every peak, and sparse mode is not
selected automatically. All settings remain together in `SettingsDefaults`.

The implementation uses C-contiguous direction-by-point blocks of 64
directions, one projection generation per block, in-place dense sigmoid and
normalization, and vectorized sparse gathering/bincount. The partition helper
selects floor/ceil ranks of $0.99(n-1)$ in a copy, preserving the original
projection-column-to-point mapping, then uses the same stable linear
interpolation as NumPy. There is no single-rank quantile approximation. In
float32 both reference and optimized thresholds are cast to float32 before
weighting, matching the precision choice. Installed NumPy 1.21.5 itself already
uses partition internally; measured speedup is not attributed to eliminating
a full sort that did not exist.

`python3 -m corner_detector.soft_voting_benchmark --seeds 100` compares hard,
reference dense, exact partition dense, and sparse alpha 4/6/8, each in both
float precisions. Seven XYZ scenes include 1/2/3 faces, two same-object corners,
separate objects, near-vertex missing samples and outliers. Six noise sigmas
are 0/0.05/0.1/0.2/0.35/0.5 mm. The study explicitly fixes the requested 30 mm
core, 10 mm padding, 256 directions, 0.25 mm tau, 4.5 mm components, 5 mm NMS and
0.04 score gate. This leaves live/user settings untouched. Every variant reuses
the same raw cloud and unchanged CED NMS query indices. GT-near voxel queries
are reported separately. CED/PCA/NMS settings and code are unchanged.

CSV records preserve per-query projection, quantile, sigmoid/support-gather,
normalization, accumulation, NMS, ROI, component and total times. Frame timing
includes shared spatial-index preparation and all supplied queries; CED and
voxel preprocessing are recorded separately. Generation, matching, metrics,
I/O and rendering are excluded. Execution order rotates by seed, BLAS/OpenMP
pools are limited/recorded, and CPU/version/config provenance is saved.
One-to-one 5 mm matching reports recall/FP alongside matched-only median/P95
errors. Noise error curves pool matched observations across scenes at each
noise level; report tables explicitly label macro averages of condition-level
statistics. Repeated-position RMS is an evaluation statistic over matched
observations, never an averaged output position. Precision/sparse comparisons
must read peak agreement and misses as well as vote-map differences.

Research limitations include sparse-tail approximation, possible float32 tie
changes, per-point threshold dilution under soft mass, and singleton outlier
components receiving confident votes in every backend. Heatmap contrast alone
does not establish detection accuracy or geometric correctness.

In addition to the 100-seed unchanged-0.04-gate study, a separately labelled
30-seed `--min-vote-fraction 0` ablation applies the same zero gate to **all**
backends. It isolates peak behavior from soft-mass gate dilution without
changing live settings. It must report the additional false positives and NMS
cost; selecting a permissive gate only for soft voting would not be a fair
comparison. `paired_comparisons.csv` compares only the same GT/scene/noise/
seed/source across modes and precisions. Cross-gate conclusions and measured
trade-offs are in `outputs/soft_voting_20261029/CONCLUSION.md`.

## 14. Revision record

| Date | Revision |
|---|---|
| 2026-10-30 | Refined Boundary-only Corner groups with the existing translated, rank-checked all-Line 3-D LS solver, including after group merges, while retaining single-intersection constrained LS and all topology/extent/residual gates. Selected accepted Corners by shared-intersection presence, perpendicular Line RMS, total extension and query distance, with deterministic ID tie breaking and sub-1e-8 mm ranking roundoff suppression. Added per-vertex ranking metrics and restored experimental 30/30/30/50 mm limits to 5/5/5/10 mm. Guided validator and total share this policy; upstream scoring/RANSAC/Hull/TLS/strip/masking and disabled point support are unchanged. Only modified Python syntax was checked; no tests or GUI/device runs were performed. |
| 2026-10-30 | Changed only total mode's final Corner display markers to red and doubled their radius (4 mm at 5 mm validator voxel). Updated the final-window colour legend; contributing Edge styles and numerical detection results are unchanged. Python syntax passed; no tests or GUI/device run was performed. |
| 2026-10-30 | Added total mode: unchanged detector/NMS followed by the canonical guided validator for every scene seed, aggregating each ACCEPT query's selected Corner and contributing active Edge intervals. Added exactly two managed result views with Q99 Fusion and gray raw geometry, green Corner/Boundary, cyan shared bars and larger markers. Reused frame KD-trees, skipped unused organized mapping and core-only debug comparisons, retained compact query records, suppressed seed dumps and batched coloured meshes. Existing modes/numerical thresholds and seeded sampling are unchanged. Only modified Python syntax was checked; no tests, GUI/device runs or benchmark were performed. |
| 2026-10-30 | Swapped only the auxiliary display numbers: 4 is Hull/clipping/TLS/Before-After, 5 is plane-pair intersection, and 6 remains final. Kept each view's contents, layout, camera/label lifecycle and PNG export attached to its original role. Numerical execution order and results are unchanged; only modified Python syntax was checked, without tests or GUI/device runs. |
| 2026-10-30 | Removed the individual-plane Boundary window and moved the former window-7 Hull/clipping/TLS/Before-After views to window 5, including S-key export and camera/label lifecycle. Auxiliary order is now 4/5/6, with 6 last. Panel 6-4 distinguishes contributing shared intersections (cyan) from Boundary contributors/Corner (green) and reports type-specific IDs/counts or an explicit Boundary-only label. The numerical contributor list, Corner selection, geometry and configuration are unchanged. Only modified Python syntax was checked; no tests or GUI/device run was performed. |
| 2026-10-30 | Changed panel 6-3 from full endpoint allowance guides to actually used refined-vertex extrapolation, with adaptive short dashes, residual links and wire-halo markers. Added panel 6-4 containing only the selected Corner/incident active Edges on the original raw cloud, uploaded once through the manager/worker and retained across overlay updates. Window 6 uses a 2-by-2 layout; numeric geometry, thresholds and selected-query scope are unchanged. Only modified Python syntax was checked; no tests or GUI/device run was performed. |
| 2026-10-30 | Fixed ROI debug redraw for SceneWidget bindings without clear_3d_labels: retain Label3D handles in display panels and remove each with remove_3d_label before replacement. Numerical results, geometry and configuration are unchanged. Only Python syntax was checked; no tests or actual GUI/device runs were performed. |
| 2026-10-30 | Replaced outer-contact Hull exclusion with padded Hull segment clipping before unchanged TLS formulas/thresholds. Added stable ORIGINAL/CLIPPED Nodes, inherited endpoint states and source Hull Edge IDs; same-plane Boundary Corners now require a shared active endpoint. Preserved core RANSAC voxels/models and cross-plane intersection/LS paths with inactive endpoint safeguards. Consolidated three radius/margin controls into absolute `validator_roi_padding_mm=10`. Added selected-query debug window 7 (A/B/C/D), stored core-only comparison, synchronized Before/After cameras and S-key PNG export. Only modified Python syntax was checked; no tests or GUI/device runs were performed. |
| 2026-10-30 | Added explicit support padding and outer crop-contact margin (5x/2x RANSAC voxel). Hull support uses max(configured support radius, core + padding), while RANSAC retains the unchanged core cloud. Tagged pre-projection outer-contact hull segments, excluded them before TLS without bridging gaps, clipped all finite edges to the core sphere, and prohibited crop-created endpoints as vertex/extrapolation evidence. Preserved source endpoints and support/core overlap diagnostics; final vertices also require core membership. Point-support validation remains disabled. Only Python syntax was checked; no tests or GUI/device experiments were run. |
| 2026-10-30 | Consolidated duplicate vertex/corner controls into one minimum angle, one line-distance gate, one endpoint extension allowance, group span and query gate. Removed vertex parallel-angle and legacy voxel-scaled corner endpoint defaults/config/CLI and redundant min/max gating. Preserved current 30-degree and 5/5/5/10 mm effective values, boundary direction behaviour, LS and display metadata. Vertex distances are now controlled solely in absolute mm. Only modified Python syntax was checked; tests and GUI/device runs were not performed. |
| 2026-10-30 | Added a blue translucent line-distance cylinder over each finite shared edge in panel 6-2, using actual overlap settings stored as display metadata, plus end rings and angle/minimum-overlap captions. Kept masking and validation numerics unchanged. Only modified Python syntax was checked; no tests or actual GUI/device runs were performed. |
| 2026-10-30 | Corrected validator scope to one selected Coarse NMS query: removed automatic scene traversal, aggregate IPC/state and window 7. Replaced window 6 with original per-plane/shared edges, retained/removed interval masks, and endpoint-extension/pair-candidate/LS-corner panels. Added only the effective extension tolerance as display metadata; kept all graph, RANSAC, support, hull, mask and LS numerics unchanged. Only modified Python syntax was checked; tests and device/GUI runs were not performed. |
| 2026-10-30 | Added intersection/boundary duplicate masking with original TLS geometry and active intervals, anchor candidate span grouping, intersection-constrained and multi-line 3-D LS refinement with post-fit finite/residual gates, and independent overlap/vertex thresholds. Set the boundary TLS maximum-error default to the requested 2.5 mm. Retained boundary-only midpoint validation, current RANSAC/support/hull/strip/ROI gates and disabled point-support semantics. Added canonical all-NMS scene traversal and lazy final-only window 7 containing accepted corners/incident active edges. No cross-query NMS was added. Only modified Python syntax was checked; no tests, synthetic experiments or device/GUI runs were performed. |
| 2026-10-30 | Removed edge support radius/minimum-count settings, CLI and point-to-edge/near-vertex validation, including derived point metrics/quality. Retained geometric hull minimum, RANSAC, component selection, TLS/strip, length and corner gates. Boundary ROI cut now uses the actual support radius without the former radius-dependent inner band. Retained empty support slots, neutral quality and explicit unchecked flags; updated terminal/output semantics. Only modified Python syntax was checked; no tests or experiments were run. |
| 2026-10-30 | Replaced endpoint-chord boundary grouping with per-plane 2-D TLS over unique group vertices, maximum orthogonal residual gating, traversal-aligned original/group direction checks and projected min/max endpoints lifted to 3-D. Added independent absolute 2.5 mm boundary merge error and CLI. Preserved cyclic adjacent grouping, strip/intersection calculations, `_edge` validation and downstream schema. Only modified Python syntax was checked; no tests, synthetic data or benchmarks were run, as requested. |
| 2026-10-30 | Stored existing plane-pair intersection line origins/directions in diagnostics and added window-4 yellow infinite-line guides plus per-plane translucent tolerance strips and width caption, including zero accepted-edge results. Reused stored geometry without changing detection/support decisions or other views. Only modified Python syntax was checked; tests and actual GUI/device runs were not performed. |
| 2026-10-30 | Added per-plane RANSAC inlier counts with extraction-order colour labels and the effective minimum to the third-panel caption; reserved two lines. Read stored results only, preserving numerical results and plane acceptance. Only Python syntax was checked; no tests or device/GUI runs were performed. |
| 2026-10-30 | Added CornerVINS-inspired hull/strip clipping and common projected finite intervals for guided plane-pair candidates, with independent absolute 5 mm half-width and exact baseline at zero. Added per-plane interval/distance and recovery/support rejection diagnostics. Kept existing point-support decisions, boundary edges, downstream validator and visualization unchanged. Only modified Python syntax was checked; no tests, synthetic data, ablations or benchmarks were run, as requested. |
| 2026-10-30 | Removed the guided validator Fine scorer/NMS/Top-K/settings/picker and waiting/fallback states. Coarse NMS selection now keeps its coordinate, crops original raw core before local RANSAC voxelization, and immediately fits planes. Voxelize any geometry support shell separately; preserve core/native/scene IDs and expose local cloud/counts. Panel 2 now shows gray voxel input; panels 3–6 retain their roles. Coarse numerics and RANSAC/geometry rules are unchanged, while local sampling intentionally changes. Section-15 geometry experiments remain excluded. |
| 2026-10-30 | Unified detector/validator Coarse/Fine Fusion coloring on the existing full-Coarse-frame Q99 reference. Removed Fine ROI display rescaling, forwarded the shared upper limit through renderer startup and displayed it in both captions. Preserved numerical results and invalid-gray/Turbo mapping; verified equal-score RGB identity, invalid masking and renderer propagation with focused headless tests. |
| 2026-10-30 | Split guided geometry display into window 4 (finite plane-pair intersection edges), window 5 (separate per-plane boundary panels) and window 6 (integrated edges/vertices/decision). Preserved original panels 1/2/3 and all numerical results. Added read-only type/plane filters, independent window cleanup/keys and headless separation/colour/support/lifecycle checks; no synthetic geometry experiments were run. |
| 2026-10-30 | Added finite-valid Fine peak/count, strict-threshold eligible count and empty-candidate reason diagnostics. Fine panel/console now display the actual resolved threshold and explicit Coarse fallback alongside Q99. Preserved all scores, thresholds, NMS/Top-K and fallback numerics; added threshold/equality/invalid-score/upper-tail and headless label regression coverage. Made Fine numerical fixtures explicit about their Greedy/top-K=3 settings so user-edited live defaults do not alter those expectations. |
| 2026-10-30 | Restored the original plane-only third panel (core points, fitted disks/normals, alpha 0.60 and core-radius camera). Moved support hulls, finite edges, vertices and validation status to an independent fourth native window with manager-owned cleanup and R/Q/Esc handling. Preserved all numerical results; verified separation/lifecycle with headless UI tests, without running the requested-to-skip synthetic geometry experiments. |
| 2026-10-30 | Added optional guided-mode core/halo support reconstruction, connected plane hulls, observed finite intersection/boundary edges, two-edge corner reconstruction and Fine-centred validation. Preserved all fitted-plane/Fine/Coarse numerics and existing fine-mode algorithms. Added native/scene support provenance, final corner/incident-edge output, status/timers and manager-owned 3-D geometry. Synchronized parameter descriptions to existing workspace settings. Section-15 synthetic geometry tests were explicitly not executed at the user's request. |
| 2026-10-30 | Added post-Fine-NMS Fusion-ranked top-K selection (default 3, positive integer) and CLI/config validation. Preserved scores, threshold/NMS calculations, native/scene IDs and RANSAC fitting; existing nearest/manual query selection now uses the shortlist. Added original/retained count diagnostics and numerical ranking/tie/index/selection tests. Synchronized the Fine threshold description with the user's existing fixed workspace default. |
| 2026-10-30 | Disabled automatic post-NMS validation in CLI detector mode and default `DetectorConfig`. Preserved dense CED/bilateral/PCA/Fusion/threshold/NMS values and diagnostic panels; `--only` still selects the NMS-only display. Explicit library validation and fine/guided-validator modes remain available. |
| 2026-10-29 | Added Fine peak-relative Fusion gating with `T=max(absolute floor, ratio * finite valid Fine evaluation peak)`, default ratio 0.5/floor 0.10, and made it the Fine default. Kept strict eligibility, existing fixed/global-mean modes, score calculation, NMS geometry, Coarse defaults/CLI and RANSAC unchanged. Added CLI/config validation, resolved-threshold/ratio diagnostics and numerical edge-case tests. Preserved the user's current 5 mm Fine voxel and 3.5x CED radius and synchronized their documented lengths. |
| 2026-10-29 | Made Fine CED support, PCA radius/saturation, bilateral radius/spatial/range sigmas, Fusion threshold mode/value and NMS method/radius independent `validator_fine_*` settings. Spatial lengths use Fine-voxel multipliers; dimensionless values and neighbor count do not scale. Preserved current scoring defaults and user-set NMS radius of 3x; added an existing local-maximum NMS choice alongside default Greedy. Coarse settings/algorithms and RANSAC are unchanged. |
| 2026-10-29 | Added independent Fine/RANSAC voxel sizes and stage-specific `*_x` length defaults/CLI, cached full-cloud voxel preparation and explicit native/Coarse-scene index adapters. Fine scoring and RANSAC now exclusively use their downsampled clouds. Default 10 mm voxels preserve 90/30/15 and 60/5/10 mm lengths. Counts, ratios, seed, score formulas, three-stage UI and other mode algorithms are unchanged. |
| 2026-10-29 | Split the guided validator into three labeled Open3D panels: Coarse heatmap/NMS, cached Fine heatmap/NMS, and plane fitting. Interactive Coarse selection stops after Fine; a Fine click fits planes. Preserved full default library/headless calls and no-candidate Coarse fallback, scoring/NMS/RANSAC formulas, existing numeric settings and geometry semantics. Synced the defaults table with the current workspace values; fixed numerical fixtures explicitly retain their own thresholds/radii. |
| 2026-10-29 | Replaced the anchored validator with raw-cloud Fine CED/Greedy NMS and first-point-guided 3-point Sequential RANSAC. Added complete support halos, selectable Fine raw IDs/fallback, count/RMS selection, fixed original-ROI acceptance and hard inlier removal. Project plane disks onto their fitted planes. Adopted the supplied new validator defaults and removed anchored weights/angle gates; preserved all Coarse detector functions/settings and other mode paths. |
| 2026-10-29 | Improved anchored-plane visibility: increased disk opacity from 0.23 to 0.60, added a small emissive color and bright circular rims/two in-plane diameters. Preserved geometry, memberships, scores, queries and numerical configuration; synchronized the defaults table with the existing workspace threshold/sigma ratio/normal angle (30 mm/1.5/25 degrees). Numerical examples explicitly retain their 1.5 mm threshold and 0.6 sigma ratio. |
| 2026-10-29 | Added detector-only `--only`: stop after unchanged CED/PCA/fusion/NMS, skip shared validation, and display just the existing NMS Fusion heatmap with its markers. Preserved upstream scores, thresholds, scene IDs, Q99 colors and the standard detector path when omitted; no auxiliary RGB window in this mode. |
| 2026-10-29 | Changed the anchored validator's NMS overview from gray raw-cloud context to the original voxel-scene Fusion heatmap, with the detector's valid mask, linear Q99 range and color helper. Retained the original dense scores without recomputation; preserved query IDs, picking, anchored RANSAC and local plane colors. Only scene coordinates/colors are passed at viewer startup. |
| 2026-10-29 | Replaced the anchored validator's Matplotlib/Tk 3-D rendering with native Open3D overview/result panels, alpha disk materials and camera-based query picking. Upload the full scene once and replace only ROI geometry on selection. Isolated GUI lifetime in a display-only process with bounded shutdown after reproducing a Filament teardown stall in Open3D 0.19. Preserved CED/PCA/NMS, RANSAC, scene/raw IDs and current workspace settings (including the 50 mm radius); synchronized the radius table with that existing setting. |
| 2026-10-29 | Added a separate `validator` experiment: unchanged CED/PCA/NMS queries, click-only raw-ROI distance-weighted anchored RANSAC (0–3 planes), sequential suppression with tau equal to the inlier threshold, sign-invariant normal/unique-support gates, explicit raw IDs, RGB/depth and 3-D selection, and read-only translucent disk diagnostics. Existing fine/detector composition and detector numerics are unchanged. |
| 2026-10-29 | Renamed the interactive CLI mode from `validator` to `fine`, including its default, help groups, and dispatch. The internal shared validator pipeline, stage order, configuration values, and numerical outputs are unchanged. |
| 2026-10-29 | Added a default unit-sphere heatmap of the selected raw point's actual hard/soft/sparse direction contributions, with linear per-point colorbar, scale-total reconstruction and score contribution arrays. Retained selectable binary Q99 display. Preserved production voting, settings, coordinates and NMS; verified directional mass conservation and scale/score reconstruction for both precisions. |
| 2026-10-29 | Added per-query raw-point picking and a rotatable Q99 direction-membership sphere. Reused stored components/padded ROI and configured Fibonacci directions/precision, with inclusive ties, explicit core/padding/outside-scale states and query/window cleanup. Preserved all detector scores, vote totals, settings and NMS; added numerical, picking and Tk-event headless tests. |
| 2026-10-29 | Doubled live Ball marker display radii: queries 1.2 to 2.4 mm and hypotheses 1.5 to 3 mm, consistently in overview and local views. Preserved heatmaps, stored coordinates, settings and NMS behavior. |
| 2026-10-29 | Corrected the frame-wide Ball overview background to the original detector NMS Fusion heatmap, reusing its scores, valid mask, Q99 linear display range and color function. Preserved raw hypothesis coordinates, upstream/local settings, query-local views and all numerical results. |
| 2026-10-29 | Added a non-blocking frame-wide CED/PCA/NMS-to-Ball overview alongside the unchanged three local views. Reused cached results, preserved scene/raw index spaces and all numerical behavior; deduplicated exact raw IDs for display only and retained the overview during query reselection. |
| 2026-10-29 | Changed the live ball raw-vote heatmap from logarithmic positive-range colors to linear zero-to-maximum colors at user request. Zero remains gray; updated titles, terminal diagnostics and tests. Stored votes, normalized scores, NMS, settings and saved research exports are unchanged. |
| 2026-10-29 | Added reference/exact-partition/sparse quantile soft voting alongside unchanged hard voting, bounded direction blocks, precision controls and stage timers. Fixed raw heatmaps with explicit zero-gray/positive-log ranges and fractional vote support. Added same-query 7-scene/6-noise/100-seed comparisons; preserved CED/PCA/NMS, user geometry settings, raw coordinates and no-smoothing localization semantics. |
| 2026-10-29 | Converted ball length defaults/CLI to `voxel_3d_mm` multipliers: radii `(3.0,)`, padding `1.0`, component radius `0.45`, NMS radius `0.5`. Resolved algorithms/library configs remain mm. Default 10 mm voxel behavior is unchanged; other voxel sizes now scale all four lengths. Replaced the four old ball `--*-mm` length flags with `--*-x`. |
| 2026-10-29 | Split ball visualization into three simultaneous 3-D views: object components, full pre-NMS raw-vote heatmap, and final NMS hypotheses. Removed the C/H switch and automatic ball summary window; preserved stored numerical results and synthetic export semantics. |
| 2026-10-29 | Centralized all ball defaults in `settings.SettingsDefaults.ball_*` and moved the resolved `ExtremityConfig`/validation beside the other configs. Retained the old import alias and CLI options; numerical defaults, absolute-mm units, CED/PCA/NMS and local voting are unchanged. |
| 2026-10-29 | Added independent per-existing-query raw-XYZ extremity vertex hypotheses, optional local object components, padding, component-aware multi-peak NMS, multi-scale diagnostics, `ball` click inspection and CED-fed `extremity256` mode. Preserved existing CED/PCA/NMS equations, defaults and the detector/validator paths; added synthetic CED-versus-GT-near ablations and explicit raw/voxel index provenance. |
| 2026-10-28 | Restored the lazy Open3D proxy used by all active 3-D validator views, fixing the `o3d` startup `NameError` without forcing Open3D imports in headless numerical tests. |
| 2026-10-28 | Removed the preliminary OpenCV query-selection gate from validator startup; the frozen Tk RGB/depth window now opens directly and owns query selection/reselection. |
| 2026-10-28 | Removed the multiscale Gaussian/Harris, SAH affinity-graph, independent Edge3D, and surface-aware-normal diagnostic chains; preserved the main 2-D edge Gaussian boundary-weight equation in `boundary_support.py`. |
| 2026-10-28 | Removed the unreachable detector-only K-subspaces plane verifier, its private visualization/test chain, and three superseded package-local patch notes; `PIPELINE_METHODS.md` remains the canonical history. |
| 2026-10-28 | Reordered settings/default resolution/CLI groups by runtime stage and removed unconsumed corner-NMS, fixed-scope, and detector-only plane-verifier settings without changing retained defaults or unit conversions. |
| 2026-10-28 | Removed the cross-candidate Otsu threshold over `mean_quality`; every geometrically valid positive-score segment now enters final NMS. |
| 2026-10-28 | Added continuous Hough-seed refinement in $(\rho,\theta)$, profile-derived finite segments, validity filtering, overlap-aware final 2-D NMS, and 3g/3h pre/post-NMS diagnostics. |
| 2026-10-27 | Established shared detector-to-validator plane-edge composition and graph-local refinement. |
| 2026-10-27 | Added query-core organized CornerVINS AHC, RANSAC/refit, merging, convex hulls, and full-scene ROI views. |
| 2026-10-27 | Set the query-local Kinect defaults to a 3-pixel AHC block and 15 mm RANSAC inlier gate. |
| 2026-10-27 | Removed the selectable FPS/SAH reference-plane clustering backend. CornerVINS is now the sole automatic plane extractor for every query and detector NMS seed. |
| 2026-10-27 | Split plane diagnostics into coarse AHC, refined AHC, pre-merge RANSAC/refit, and final merge/hull views without changing extraction numerics. |
| 2026-10-27 | Added the explicit 50 mm `projection_halo_mm` setting: plane extraction remains inside the 100 mm query core while residual-compatible observed boundary context extends to 150 mm. |
