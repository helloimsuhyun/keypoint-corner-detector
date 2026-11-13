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

`--mode detector` defaults to the unchanged dense CED/bilateral/PCA/fusion-threshold/3-D NMS composition. Optional Q10 validation refines within the fixed NMS ball and applies Q10/opposite-support gates. Both the CLI detector and default `DetectorConfig` use `validator=None`; no plane, edge or corner computation runs. Scores, thresholding, NMS and scene IDs are unchanged. Library callers can still explicitly supply `ValidatorConfig` to invoke the shared validator.

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
kept detector seed runs the entire following composition immediately:

```text
Coarse NMS -> optional Q10 keep/filter + relocation
  -> selected seed center q_c (original NMS XYZ if Q10 OFF)
  -> inclusive radius crop from original raw XYZ
  -> query-local voxel downsampling
  -> first-point-guided sequential RANSAC (core voxels only)
  -> support recollection / hulls / finite edges / corner decision
```

There is no validator Fine score, second threshold/NMS, Top-K, validator-side query movement,
Fine picker, or fallback stage. `validator_fine.py`, `validator_fine_*` defaults
and CLI options, and resolved `fine_*` fields were removed. The separate
`--mode fine` organized AHC/Hough/refiner pipeline is unchanged.

**Crop before voxel.** For raw input $X$ and the selected seed center $q_c$,
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
$n=(v_b\times v_c)/\|v_b\times v_c\|$, $d=-n^Tp_a$. The selected seed center defines
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
`query_index` indexes the supplied Coarse scene. `coarse_query_position_mm`
is that original coordinate; `query_position_mm` is the actual start center,
optionally supplied separately as the Q10 refinement result. Without an override
the two coordinates remain equal. Support IDs still index the original scene.
`roi_raw_point_indices` addresses the original raw input before local voxelization.
`ransac_points_mm` stores $[P_c;P_h]$; native `roi_ransac_indices`, plane
`inlier_ransac_indices`, and geometry support IDs address this query-local array.
RANSAC masks address `roi_points_mm`, and accepted plane masks remain disjoint.
`core_voxel_count` and `halo_voxel_count` record the concatenation boundary.
Vertex `distance_to_query_mm` replaces the former Fine-distance name;
the selected seed center (Q10 refined when enabled) centers the local input crop; Corner generation has no query-distance gate.

`ransac_scene_indices` is the nearest-Coarse adapter for this local cloud.
It supplies scene-domain `roi_point_indices`, plane `inlier_point_indices`, and
geometry `support_point_indices` without replacing native coordinates, masks or
support counts. Mapping can be many-to-one. These adapters do not identify raw
contributors to a voxel. Local IDs from different queries are not interchangeable.
`planes_computed=True` includes empty-ROI/zero-plane runs; there is no Fine-wait
state. Report query, raw/voxel counts, voxel/core radius, plane diagnostics,
corner coordinates and incident finite edges, and voxel/RANSAC/geometry timers.

**Visualization.** The main native window keeps three panels:

1. Coarse Fusion heatmap and orange kept detector seed markers, using the detector's valid mask,
   linear full-frame Q99 range and color function. Only this panel picks queries.
2. Stored core voxel input in gray, the orange selected seed center, and its radius
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
unchanged Coarse NMS -> optional Q10 filtering/relocation -> raw seed crop -> local voxel input
  -> unchanged guided sequential RANSAC (core only)
  -> independent support recollection in core + halo
  -> connected per-plane tangent projection + convex hull
  -> hull/strip finite plane intersections
     + padded Hull segment clipping -> ACTIVE/INACTIVE Nodes -> TLS boundary lines
  -> intersection/boundary overlap masks + active finite-edge graph
  -> Edge-pair angle / 3-D line intersection / endpoint radius tests
  -> shared Edge + same Endpoint + complete-diameter graph integration
  -> all unverified Corner candidates (CANDIDATES / NO_CANDIDATES)
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
together. Only missing anchored support marks connectivity UNKNOWN; multiple
nearby anchored components do not invalidate the selected component.
The Corner radius does not participate in support selection or validity.
At least three noncollinear points are
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
Edges as provenance. The endpoint Corner generator uses supplied Edge endpoints
and does not apply the former Node-activity or same-plane adjacency gates.
Each clipped Edge retains its Plane ID and original Hull Edge ID. TLS groups
merge only through a shared Node ID, including true cyclic adjacency; separated
open chains cannot reconnect. Merged Lines retain ordered support Node IDs,
all source Hull Edge IDs, their original traversal start/end Node identities,
source and enabled states independently of fitted coordinates. Store the fitted
RMS and maximum TLS residuals; only RMS gates merging. The SVD formula,
complete-direction checks and greedy first-eligible merge policy are unchanged. Input clipping
can change fit values and merge outcomes.

Reclip fitted TLS extrema to the core if needed before minimum-length checks.
Any extra fit-created crop endpoint is CLIPPED/inactive with its parent Node ID;
projection never changes the stored clipped Node state. Intersection segments retain
their existing hull/strip computation and core clipping. Preserve source endpoints,
`roi_clipped_start/end`, pre-core support overlap and surviving core interval
diagnostics. The Corner generator adds no core/query gate or point-support check. `boundary_debug` stores original Hull Nodes/Edges,
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
intersection candidates and graph edges. Strip intervals remain
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
using the independent `boundary_merge_angle_deg` (current CLI default 0.1 degrees) without
absolute-dot reversal equivalence.
For the $N$ unique vertices, require
$\sqrt{\frac{1}{N}\sum_i\|(I-vv^T)(p_i-\mu)\|^2}\le\tau_{merge}$.
Each unique vertex has equal weight; no segment-length or raw-support weighting
is applied. The independent absolute-mm `validator_boundary_merge_rms_error_mm`
(current CLI default 5.0, finite/nonnegative; CLI `--validator-boundary-merge-rms-error-mm`)
replaces `validator_boundary_merge_max_error_mm` and its CLI option. Resolved
configs likewise use `boundary_merge_rms_error_mm`. Maximum residual no longer
rejects a merge: `tls_rms_error_mm` records the acceptance metric on Lines and
Boundary Edges, while `tls_max_error_mm` remains diagnostic. Keeping the same
threshold makes the residual gate more permissive and can absorb genuine bends;
the number and placement of unique Hull vertices affect this averaging.
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
neutral for all edges, so edge ordering falls back to length/source ID and it
provides no point-evidence ranking. Edge/result `point_support_checked=False`
and terminal `point support=disabled` make the new semantics explicit.

Every valid finite candidate becomes a graph Edge; there is no same-type
finite-edge deduplication. Preserve the existing sort by descending
`support_quality`, descending `length_mm`, then ascending `source_edge_id`, and
assign consecutive query-local `edge_id` values in that order. Each Edge keeps
its original geometry and provenance, with the compatibility `source_edge_ids`
list containing only its own source ID. Plane-pair enumeration produces at most
one intersection candidate per pair. Boundary Lines from distinct Hull chains
remain separate, including thin Hulls whose opposite chains share endpoint
Node IDs. Removing the former suppression can retain more Edges and change
subsequent IDs and Corner results. `validator_edge_duplicate_angle_deg`, its
resolved field and CLI option are removed.

**Intersection–boundary masking.** After finite-edge ordering and ID assignment,
compare each valid intersection $I$ with each boundary $B$ that owns one of its
planes. Require $\arccos(|u^Tv|)\le\theta_o$, both original boundary endpoints
within $d_o$ of the infinite intersection line, and projected overlap
$\max(0,\min(L_I,b_1)-\max(0,b_0))\ge\ell_o$ with sorted boundary projections.
Defaults are 25 degrees, 15 mm and 2.5 mm. Map each qualifying overlap back to the
original boundary's own parameter, accounting for direction sign, and union all
mask intervals from all intersections. Preserve original TLS start/end/direction
and source IDs. Store `masked_intervals_mm`, complementary
`active_intervals_mm` and `masked_by_intersection_ids`; a fully masked boundary
is INACTIVE and absent from active `edges`, while `all_edges` retains it.
`boundary_overlap_diagnostics` records masks. Intersections stay active.
A nonparallel crossing boundary is not masked. Active pieces are parameter
intervals of the original length-validated edge, not newly fitted segments;
mask cut endpoints are not materialized as new Edges or Corner endpoints.
Partially masked Edges retain their original endpoint coordinates; the new Corner
stage does not recheck masks or Node activity at those endpoints.

**Independent endpoint Corner generation.**
`validator_corner_graph.generate_corners(edges, min_angle_deg=20.0,
corner_radius_mm=5.0)` reads existing finite Edges and returns a list. It has no
query, scene cloud, plane, visualization or detector dependency. Pipeline config
supplies the two Corner controls; current user-edited CLI defaults are 45 degrees
and 15 mm, while the independent function keeps the requested 20-degree/5-mm
experimental reference defaults. Neither set is an accuracy calibration.

For each valid Edge pair, normalize the supplied endpoint difference to obtain
unit directions $u,v$ and require $\arccos(|u^Tv|)\ge\theta_{min}$.
Solve $[u,-v](t,s)^T=b-a$ for the two infinite Lines' closest points $p,q$.
Reject rank-deficient pairs. Their intersection residual is $\epsilon=\|p-q\|$;
require $\epsilon\le\max(\tau_i,\tau_j)+10^{-8}$ mm. The fixed $10^{-8}$ mm
term is numerical roundoff, not another user control. Each existing Edge carries
`fit_tolerance_mm`: Boundary Edges reuse the configured TLS RMS merge limit;
plane-intersection Edges reuse the plane inlier distance limit. These are existing
fitting bounds, not measured Corner uncertainty. External Edges without this
metadata use zero, requiring intersection to numerical precision. Adding this
metadata does not change any upstream Edge coordinates or acceptance.

Use $c=p+(q-p)/2$ for the pair position; for coplanar Lines this is their
intersection up to roundoff. For each Edge's supplied finite endpoints, compute
$g_i=\min(\|c-a_i\|,\|c-b_i\|)$ and retain the pair only when
$\max(g_i,g_j)\le r$. This tests endpoint proximity, not distance to the segment
interior or the minimum object size. Endpoint IDs are side indices 0=start,
1=end, qualified by the query-local Edge ID, not Hull Node IDs or scene point IDs.
Ties choose start deterministically. The candidate stores `position`, `edge_ids`,
`endpoint_ids`, `intersection_residual` and `endpoint_gap=max(g_i,g_j)`.
All positions/residuals/gaps are in mm.

Integrate candidate groups only when they share an Edge and use the same
Endpoint of that Edge. Reject any shared-Edge Endpoint conflict, since one
integrated Corner cannot assign two endpoints to one Edge. Require every
cross-group pair of original candidate positions to be within $r$; together
with the same condition already holding within groups this bounds the full
maximum diameter and prevents proximity-chain growth. Traverse pairs in sorted
Edge-ID order and greedily merge the first eligible groups until convergence.
Input enumeration order does not change that deterministic traversal.

Keep the candidate with the smallest `endpoint_gap` as the integrated position;
ties use `intersection_residual`, then the source Edge pair IDs. Do not average,
move, snap or refit this position. Union the member Edge IDs and consistent
Endpoint maps. The output residual/gap belong to that representative pair,
not a new fit or a recomputed maximum over every integrated Edge; a newly joined
Edge endpoint can therefore lie farther than $r$ from the representative. No
post-merge structural validation is performed.

The generator has no same-plane shared-Node requirement, ACTIVE/CLIPPED Node
gate, active-interval reachability test, endpoint extrapolation allowance,
anchor constraint, multi-Line LS, query/core acceptance gate or selected-Corner
ranking. All merged candidates are returned; a query may have zero, one or many.
It reads the endpoints of the valid Edges passed to it, including crop-created
or masked original endpoints; it does not establish physical endpoint existence.
The upstream finite Edge clipping and overlap masks remain numerically unchanged.

Pipeline output is `corners` plus `validation_status=CANDIDATES` or
`NO_CANDIDATES`; `verified=False` in both cases because structural validation is
deferred. Disabled geometry retains `NOT_COMPUTED` and an empty list. Remove
`corner_position_mm`, `corner_distance_mm`, `selected_vertex_id`, `corner_edge_ids`,
`corner_edges`, old vertex/LS fields and ACCEPT/REJECT/INCONCLUSIVE selection.
The separate fine-mode graph/refiner is unchanged.

**Initial settings.** Fields below live in `SettingsDefaults`; resolved fields
omit the `validator_` prefix in `GuidedValidatorConfig`. CLI uses hyphens.

| Setting | Default | Rationale / units |
|---|---:|---|
| `validator_geometry_enabled` | True | `--validator-geometry` / `--no-validator-geometry` |
| `validator_edge_overlap_angle_deg` | 25.0 | maximum unoriented intersection/boundary angle |
| `validator_edge_overlap_distance_mm` | 15.0 | both boundary endpoints to intersection line |
| `validator_edge_overlap_min_length_mm` | 2.5 | minimum finite overlap along the intersection |
| `validator_roi_padding_mm` | 10.0 | sole padding setting; absolute mm, support radius = core + padding; 0 supports core-only comparison |
| `validator_plane_connect_radius_x` | 4.0 | 20 mm at 5 mm RANSAC voxel; unchanged support component radius |
| `validator_plane_parallel_angle_deg` | 20 | small-angle intersection exclusion, not an orthogonality gate |
| `validator_intersection_strip_half_width_mm` | 15.0 | independent strip half-width in mm; 0 retains exact hull/line clipping |
| `validator_boundary_merge_angle_deg` | 0.1 | maximum directed angle for original segments and group TLS directions during adjacent boundary merging |
| `validator_boundary_merge_rms_error_mm` | 5.0 | RMS of unique 2-D TLS vertex residuals for adjacent boundary merges; absolute mm |
| `validator_edge_min_length_x` | 3.0 | 15 mm minimum finite edge length at the current 5 mm voxel |
| `validator_min_angle_deg` | 45.0 | current CLI Corner pair minimum angle; independent function reference default 20 degrees |
| `validator_corner_radius_mm` | 15.0 | current CLI endpoint/merge radius; independent function reference default 5 mm |

Boundary merge and Corner pair angles are independent, finite and in `(0,90]`
degrees. `validator_min_angle_deg` and `validator_corner_radius_mm` are the only
Corner controls, exposed as `--validator-min-angle-deg` and
`--validator-corner-radius-mm`. Removed controls are the former Corner minimum
angle/query gate and all `validator_vertex_*` distances. The radius is an absolute
positive mm distance; it does not scale with voxel size. Both validator and total
use the same independent generator.

All `*_x` geometry lengths multiply the RANSAC voxel; counts/angles do not scale.
These are initial experimental values, not calibrated accuracy claims. Existing
Coarse score/NMS settings and RANSAC fitting rules are preserved. The direct
query-local crop/voxel stage and its settings are described in section 1.1.
Positive lengths, boolean enable and (0,90]-degree
angle validation follow the existing config/CLI pattern.

**Display and timing.** The original main window retains unchanged Coarse
heatmap/query selection, local voxel input and RANSAC planes. All diagnostics
remain manager-owned, reading stored results without numerical mutation.
Auxiliary windows are 4: padded Hull / ROI clipping / TLS provenance / core-only
Before versus padded After, 5: finite plane-pair intersections and tolerance
strips, and 6: four process panels. Window 6 panels show original Edges,
unchanged overlap masks, endpoint Corner integration, and all merged Corner
candidates with their contributing Edges over the original gray raw cloud.
All Corners are green unverified hypotheses, contributing Boundaries green,
shared intersections cyan, and links from the recorded Edge Endpoint to each
Corner yellow. No selected-corner highlight, LS shifts, acceptance colour or
extension allowance is displayed. Panel 6-3 reports the common endpoint/merge
radius. Window 4 D panels show their stored Corner lists; only C adds TLS RMS
Line/error labels. Cameras, R/Q/Esc, S-key PNG export and per-panel cleanup retain
their existing lifecycle. No new scene query is created by visualization.

The optional core-only Before comparison calls the same generator on the stored
core-only Edge graph. It never affects the padded result; `roi_debug_before`
retains its own `corners`, and `include_debug=False` skips this comparison.
Stage timers retain voxel, RANSAC, support, Edge, Corner and debug work separately.
Headless numeric/UI-mock tests cannot establish physical Corner accuracy or
hardware/rendering behaviour. Device, GUI and accuracy calibration remain
unverified.

### 1.3 Whole-scene total composition

`--mode total` composes the unchanged scene detector/NMS, shared optional Q10 seed policy and canonical
guided query pipeline. Collect every generated `corners` entry, including
`verified=False` hypotheses, without per-query selection, structural validation,
cross-query merging or an additional NMS. Aggregate `final_corners` records use
`position_mm`, query-local `corner_id`, qualified `query_index`, contributing
`edge_ids`, `endpoint_ids`, representative residual and gap. `final_edges`
contains one contributor record per Corner/Edge relationship, preserving its
active intervals, source Edge ID and selected endpoint side. An Edge used by
multiple Corners can occur more than once with different Corner provenance.
Compact per-query records retain the full Corner list, status and stage timers,
without raw/local clouds or diagnostic graphs.

Reuse the existing raw/Coarse trees, raw crop before local voxelization,
independent core/halo grids and native/scene-ID adapters. The total path sets
`include_debug=False`, retaining the normal computation and omitting only the
core-only visual comparison. Geometry must be enabled. Progress and summary
report candidate/empty query counts, total unverified Corners, errors and stage
times. No detector, sampling, RANSAC, Hull, TLS or mask equation changes.

Manager still opens exactly two result windows: unchanged detector Fusion/NMS,
and original gray raw XYZ with every generated Corner in red, Boundary
contributors green, shared intersections cyan and endpoint-to-Corner links
yellow. Existing batched meshes, active-interval rendering, marker sizes and
R/Q/Esc/close lifecycle remain. There are no intermediate query windows in total
mode. These are candidate displays, not accepted physical Corners.

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

The mutable organized context caches extraction by `(query_index, actual query XYZ,
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

PCA support now has a configurable original-CED cutoff
`pca_ced_cutoff` / `--pca-ced-cutoff`, finite in [0,1]. For query i, retain

J_i = {j: ||p_j-p_i|| <= R_PCA, CED_valid_j, s_j>0,
          s_j>=pca_ced_cutoff},       w_j=s_j.

Default 0.0 preserves the previous positive-original-CED support and all
numerical outputs. A value of .10 excludes raw CED below .10, including when
bilateral filtering targets CED or Fusion. Equality is included; zero weights
remain excluded at cutoff zero. The query's own CED is not thresholded, so a
low-CED query can still have a valid weighted neighborhood. Original CED,
CED validity, bilateral-CED scores, coordinates and scene IDs are unchanged.
`pca_neighbor_count` remains the geometric radius count;
`weighted_neighbor_count`, PCA weight sum, effective sample size, weighted
centroid/covariance/eigenvalues and ratio describe only retained support.
The existing min-neighbor/eigenvalue gates still apply; no new effective-count
gate is introduced. Insufficient retained support invalidates PCA/Fusion and
sets Fusion to zero, rather than falling back to the excluded samples.
Positive cutoffs can therefore change Fusion, threshold eligibility and NMS.
All shared detector/validator/total/new-validator/ball/extremity Coarse consumers
and comparison callers use this setting; results/logs record `pca_ced_cutoff`.

For $\lambda_1\ge\lambda_2\ge\lambda_3\ge0$,

$$
r_i=\operatorname{clip}(\lambda_2/\lambda_1,0,1),\qquad
\widetilde r_i=\tau\tanh(r_i/\tau),\qquad
F_i=\operatorname{clip}(s_i^B\widetilde r_i,0,1).
$$

`SettingsDefaults.bilateral_mode` selects exactly one target:

- `"ced"`: filter original CED, then multiply by the configured PCA
  ratio: F_i=clip(Bilateral(s)_i * r_tilde_i,0,1). This preserves the previous
  boolean-True numerical outputs.
- `"fusion"`: form F_i_raw=clip(s_i * r_tilde_i,0,1), then
  F_i=Bilateral(F_raw)_i. All raw products exist before any filtered output is
  written; never filter in place or filter CED first in this mode.
- `"off"`: F_i=clip(s_i * r_tilde_i,0,1), preserving previous boolean-False
  outputs and skipping filter arithmetic.

For the selected source field z, the existing normalized bilateral kernel is

w_ij = exp(-||p_j-p_i||^2/(2*sigma_spatial^2))
       * exp(-(z_j-z_i)^2/(2*sigma_score^2)),
Bilateral(z)_i = clip(sum_j(w_ij*z_j)/sum_j(w_ij),0,1).

Use the inclusive configured spatial radius. CED-stage support contains only
CED-valid samples; Fusion-stage support contains only fusion-valid samples.
Invalid queries remain zero and never acquire validity through filtering.
The range sigma retains its historical `bilateral_sigma_ced` / CLI name but
measures selected CED or Fusion score differences, with no new sigma parameter.
Radius, spatial sigma and range validation applies to all three modes.

Settings/config/`--bilateral-mode {ced,fusion,off}` propagate through detector,
validator, total, new-validator and ball/extremity Coarse consumers, including
comparison callers. The boolean `bilateral_enabled` is removed. PCA always uses
the same original CED weights, support, traversal order and validity masks;
coordinates and scene IDs remain unchanged. Threshold eligibility and NMS use
the final `fusion_score`, so their results can change with the selected mode.

Results record `bilateral_mode` and `fusion_score_raw` (the product before the
optional Fusion-stage bilateral pass). `fusion_score_raw` and `fusion_score`
can reference the same array in ced/off modes; diagnostics never mutate either.
`bilateral_ced_score` copies original CED in fusion/off modes. Its historical
`bilateral_valid_mask` remains CED validity in every mode; final validity remains
`fusion_valid_mask`. Generic bilateral neighbor counts/weight sums describe
whichever field was filtered, and are zero for off. Fusion filtering reuses the
existing normalized-field helper and tree in one additional scalar-query pass;
the ced/off PCA neighborhood lookup/order stays unchanged. Shared lookup time
and Fusion filtering time are included in the bilateral timer, which can be
nonzero even in off mode because shared preparation still runs.

Logs show the selected mode. The third heatmap panel shows filtered CED for
ced mode, original CED/bypass for off, and pre-filter product `fusion_score_raw`
for fusion mode; the final panel explicitly labels Bilateral Fusion. Raw and
final Fusion share the final Fusion Q99 color scale. CED/filtered CED keep the
original CED Q99 scale. Display reads never recompute or change numerical data.

Omitting $\tau$ uses $\widetilde r_i=r_i$. Fusion thresholding is fixed or the
global mean over finite valid scores. Eligibility is strict, $F_i>T$.
Inclusive-radius 3-D NMS retains a point only when no eligible neighbor has a
larger score; equal scores favor the smaller scene index. CLI detector output stops at these unverified candidates. Explicit library validation passes NMS survivors to the same CornerVINS validator pipeline.

## 9. Default parameter mapping

| Setting | Meaning | Default |
|---|---|---:|
| `voxel_3d_mm` | scene voxel | 10 mm |
| `centroid_radius_mm` | detector CED support, independent of voxel | user-tuned in `settings.py` (mm) |
| `pca_radius_factor` | detector PCA support / CED support | 1.5 |
| `pca_ced_cutoff` | inclusive original-CED cutoff for positive PCA support weights | 0.0 |
| `bilateral_mode` | filter CED before product, Fusion after product, or skip | `"ced"` (`"fusion"` / `"off"`) |
| `bilateral_radius_factor` | bilateral spatial radius / CED support | 1.0 |
| `bilateral_sigma_spatial_factor` | bilateral spatial sigma / CED support | 0.5 |
| `bilateral_sigma_ced` | dimensionless range sigma for the selected CED/Fusion field | 0.05 |
| resolved `pca_radius_mm` | `pca_radius_factor * centroid_radius_mm` | derived from configured CED (mm) |
| `nms_radius_factor` | Local Maximum NMS radius / CED support | user-tuned in `settings.py` |
| resolved `nms_radius_mm` | `nms_radius_factor * centroid_radius_mm` | derived from configured CED (mm) |
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
graph-local refinement. `min_neighbors` remains shared with fine. Fine retains
its own `centroid_radius_x`, while validator/total/ball/extremity Coarse queries
reuse the CED-based detector radii above. The former multiscale-Harris,
SAH-graph, and independent Edge3D diagnostic settings have been removed.
Detector settings end at dense NMS and default to `validator=None`. An explicitly supplied shared `ValidatorConfig` validates every retained candidate. The superseded detector-only K-subspaces
plane-verifier implementation and its private diagnostics have been removed.

## 10. Visualization semantics

### Dataset Ahmed Python baseline (2026-11-09)

The dataset runner now resolves `--preset dataset` by default; `--preset author`
retains the earlier values described below. `baselines.config` owns the
sets and all overrides are recorded in the result. Dataset Sommer uses normal
radius 10 mm, dmin/dmax 5/100 mm, minimum votes 3 and refinement stride 5;
author uses 20 mm, 100/1000 mm, 5 and 10. Optional Sommer input voxel sampling
is off in both presets. When explicitly enabled, it selects the actual input
point nearest each voxel centroid (ties: smaller original order), preserving
sample-to-raw IDs and mapping normal IDs through that sample map. No new scene
coordinate is inserted, and visualization uses the complete saved input.

Dataset Ahmed uses 5 mm initial/edge grid, 2.5 mm analysis grid, edge K=100,
threshold multiplier 2, thinning K=30, corner K=20 and exported L=10. Author
values remain 10/5 mm, 400, 4.5, 150, 20 and 10. In both presets PCA uses
10-NN, patch span is 5 mm and strict angle bounds are 50/150 degrees.
Python overrides expose PCA K (both edge direction and cloud normal), patch
span and angle bounds, preserving the source formulas and strict comparisons.
Clustering, the size gate and L-loop RNG ownership are unchanged. Direct library/standalone port
defaults and the original MATLAB backend keep author parameters; numeric
Ahmed CLI overrides apply only to the Python port. No adaptive K, NMS,
prediction merging or GT-dependent run-time filtering is introduced.

Explicit `--preset scene2` (2026-11-10) uses the dataset Sommer parameters
unchanged. Ahmed Python uses voxel 5 mm / analysis voxel 2.5 mm, edge K=100,
multiplier 2, thinning K=5, corner K=27, PCA K=5, span=8 mm,
strict angles 53.3 < theta < 116 degrees and exported L=1, seed 1.
All ten passes still execute and are retained. Individual options override
the preset; the MATLAB backend rejects this Ahmed preset. On the same
scene_0002 ROI used for tuning, Sommer produces one prediction 8.803 mm from
GT #1; Ahmed produces four, with one-to-one GT #1..#4 errors
5.495/3.883/7.278/3.110 mm. These are in-sample results, with a sensitive
angle lower bound; no generalization or full-scene accuracy is established.
See `BASELINE_TUNING.md` for the search and saved reproduction command.
Dataset and author defaults, raw input/annotations and detector numerics
remain unchanged. No count target, GT snapping or postprocessing is applied.

The small ROI-only setting selection used five scenes and separate confirmation
on five scenes, seed 1 and maximum-cardinality one-to-one GT matching at
15 mm, with 10/20 mm sensitivity. Selection/confirmation counts, the 12+12
tested configurations, preset values, reproducibility artifacts and the very
low observed precision/recall are recorded in `BASELINE_TUNING.md`. This is
parameter selection, not a complete paper benchmark or independent-object test.
Sommer did not improve on confirmation scenes. Detector/Q10/validator and
Harris calculations are unchanged; source dataset/annotations are read-only.

`--method ahmed` now defaults to a NumPy/SciPy port of author commit
`e8aa7a4254cf67c3bdd6a82a7efb67d51c47994c`; `--ahmed-backend matlab`
still directly runs the author functions. The port is separate from active
CED/Q10/validator code and does not change their outputs or support IDs.
Input/output adapters retain the mm/m conversion and saved ROI/full-scene
selection. No GT enters computation. The original example's workpiece-specific
passthrough bounds and X=0 deletion remain replaced by that input selection.

Port stages: 10 mm grid centroids, 400-NN centroid displacement
`||x-mean(N400(x))|| > 4.5*d_NN2(x)` (self included), 150-NN edge averaging,
deletion of every member of each equal-position group, edge 10 mm/cloud 5 mm
grid averaging, nearest-cloud edge mapping, 10-NN PCA first/third components,
20-NN edge patches and a k=2/3 cluster count given by the number of coordinate
spans strictly greater than 5 mm. Grid origin is each input's bounding-box
minimum, cells are half-open and X varies fastest; KNN ties prefer smaller input
IDs. PCA centers twice as in the source and forces each loading's largest
absolute coordinate positive. K-means++ uses squared distances, one replicate,
100 iterations maximum and singleton empty-cluster repair.

For every L=1..10, rerun clustering for every eligible patch. Require every
pair angle `atan2(||u cross v||,u dot v)*180/3.142` strictly between configured
bounds (author defaults 50 and 150)
degrees. Require `|n1-n2|<L` for k=2; for k=3 keep the source's OR across all
three pairwise size differences. Store all ten masks/prediction lists and export
only `--ahmed-level` (default 10, scene2 preset 1) to the shared result. Do not assume monotone
counts across L, because clustering is rerun. No extra NMS or corner merging.

The Python seed uses RandomState/MT19937; MATLAB toolbox RNG consumption,
voxel ordering/boundaries and SVD degeneracies are not certified equivalent.
Results carry `implementation=python_port`, the port file hash and
`matlab_parity_verified=false`; comparison titles also identify the port.
`ahmed_stages.npz` stores diagnostic arrays without recomputation: initial IDs
address `initial_cloud_m`, mapped edge IDs address the separate `cloud_m`, not
raw/scene points. Common prediction coordinates remain unsnapped with separately
reported nearest raw IDs. Empty edge sets return empty predictions; other
fixed-K support shortages fail explicitly. Author MATLAB execution parity is
unverified; numerical tests cover formulas, ties, duplicate deletion, PCA signs,
cluster gates, L-loop ownership and a known straight-line result.

### Dataset baseline comparison

Dataset baseline comparisons (2026-11-09) read the run's stored input raw IDs,
GT and per-method predictions in mm. They do not read subsequently edited
annotations or recompute a detector. `--visualize` opens one manager-owned
window per completed scene, with GT first and method panels in CLI order;
`view_baselines.py` reopens saved runs through the same manager. Every panel
shows the identical input ROI/full-scene cloud. GT markers are green and each
method's predictions red, at their original stored coordinates without snapping,
merging, clipping or score filtering. Zero detections are labelled `0 corners`;
failed/prepared runs retain distinct status labels. Camera view and projection
are synchronized from any manipulated panel. Each scene owns a fresh spawned
GUI process; closing its window advances to the next saved scene. Existing
detector/validator equations, output coordinates, IDs and support are unchanged.

As of 2026-11-10, the shared baseline method list defaults to Sommer / Ahmed /
PCL Harris3D. Omitting `--method` runs all three. `--visualize` always appends
any missing baseline names before runtime checks and computation, preserving
the explicitly requested panel order. Nonvisual runs still allow a method
subset. Saved-result comparison also always adds missing method panels;
an absent record has status `NOT_SAVED`, empty marker geometry and a rerun
instruction, never a fabricated zero-detection result or implicit recomputation.
All three use the same stored input/GT and existing manager-owned camera sync.
Backend parameters, detection coordinates, seed/order within each method and
unit/index contracts are unchanged.

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
| Optional Q10 refinement / antipodal validation | `corner_detector/refiner.py`, `corner_detector/q10_validator.py`, `corner_detector/spherical_antipodal.py` |
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

## Q10 relocation and final antipodal Lite

### Active Lite bidirectionality: cleaned per-plane MaxGap (2026-11-11)

The sole active final test now reproduces the current standalone lab's cleaned
angular span, replacing N+/N-/B and per-plane axis scanning. Stage order is
unchanged dense CED/PCA/Fusion -> threshold/NMS -> unchanged fixed-NMS-ball
Q10 relocation and cached quality gate -> unit-sphere great-circle RANSAC ->
angular connected group selection -> sparse end-bin trimming -> cleaned span
verdict. `spherical_antipodal.py` owns angular models and MaxGap kernels;
`q10_validator._final_evidence` composes them without importing the lab GUI.
Physical mm plane extraction remains in its independent validator layer.

At final refined p, collect the full inclusive R analysis-voxel ROI, exclude
lengths <=1e-7 mm, and normalize u_i=(x_i-p)/||x_i-p||. Use all nonzero R samples
regardless of Q10 CC. There is no radial weighting, expanded Tube ROI, final
spatial CC or NN-spacing gate. Fewer than three directions gives unavailable
evidence. Existing RANSAC configuration is retained, including the user-tuned
detector tolerance in settings.py; lab function defaults may differ.

Fit up to K origin-constrained planes n.u=0, ||n||=1. RANSAC samples two
unexplained directions; reject cross-product magnitude <.16. Angular residual
is asin(|n.u|) in degrees, inclusive at the configured tolerance. Each round
scores at most 4000 randomly selected unexplained directions, preferring larger
inlier count then smaller sum of truncated squared dot residuals. Refit against
all directions with Huber IRLS origin-TLS scatter sum(w*u*u^T), without centroid
subtraction: up to 8 iterations, Huber transition max(.5 degrees,.5*tolerance),
convergence <1e-5 degrees. Reject near-duplicate models at absolute normal
cosine >cos(max(12 degrees,1.5*tolerance)). Require
max(min_support,ceil(min_fraction*N)) newly explained directions; stop when
unsupported, never force K. Final refit/classification tests each direction
against every plane independently, retaining overlaps. Fix normal sign by
making its largest-magnitude coordinate positive. These RANSAC numerics are
unchanged by the MaxGap replacement.

Project each plane's inliers onto its great circle, renormalize, and measure
azimuth theta_i in [0,360) in the deterministic least-normal-aligned coordinate
basis. Sort circularly, including the last-to-first gap through 360/0. Connect
neighbors within `gap_link_deg` (20 degrees); choose the group with greatest
point count, then smallest angular span, then smallest start angle. This is
angular connectivity, independent of physical distance/3-D connectivity.

`trim_sparse_arc_ends` optionally cleans this dominant group. Unwrap at its
largest empty gap and bin on a fixed angular grid of width h=10 degrees. The
last endpoint is clamped inside the observed arc by 1e-8 degrees to avoid a
phantom endpoint bin. With occupied-bin counts c_b, define

T_density = max(gap_tail_min_bin_count,
                ceil(gap_tail_density_fraction * median(c_b for c_b>0))).

Defaults are minimum 2 and fraction .5. Remove bins with counts <T_density only
from the two ends until a supported bin is reached; never delete sparse
interior bins. Disabled trimming or a single-bin arc keeps the original group.
A fully trimmed group has no retained samples. Disconnected groups and trimmed
end samples both count as discarded; masks preserve the original plane order.
The filter does not guarantee density at every angle inside the kept arc.

For sorted retained azimuths phi_1,...,phi_m, define circular gaps between
successive angles, including phi_1+360-phi_m, and compute

G_k = max(circular gaps),       S_k = 360 - G_k.

With no retained points use G_k=360, S_k=0 and no arc endpoints. A plane is
measurable (the stored `reliable` field) iff kept_count>0. No retained-count or
retained-fraction threshold is applied; even a single retained point has S_k=0.
Any measurable plane with S_k>=gap_edge_span_deg (equality rejects) makes the aggregate
REJECT, even if another plane has an empty arc. Otherwise no fitted planes or
any empty cleaned arc yields UNKNOWN; all fitted planes nonempty and below the span gate gives
CORNER. The removed count/fraction gates no longer produce UNKNOWN, so small
or minority arcs can now pass. This is experimental edge-versus-corner angular evidence, not proof of
a physical corner. The PCL-style boundary flag compares the dominant group's
pre-trim maximum gap to 90 degrees; it is diagnostic only, never the veto.
Unassigned directions do not introduce a new failure condition.

When refinement is ON, read converged trace Q10 before collecting final evidence. Finite Q10 below
`q10_min_q10` immediately REJECTs with `q10_below_threshold`; equality passes.
Undefined Q10 stays INDETERMINATE. Both cases skip final ROI, RANSAC and MaxGap.
Only a finite passing Q10 reaches the final angular gate. Cache final evidence
once per reached analysis ID, without recalculating Q10. MaxGap REJECT produces
`angular_span`; CORNER produces ACCEPT / `compact_angular_span`; UNKNOWN keeps
the candidate INDETERMINATE. Existing ACCEPT-only refined-seed handoff to the
canonical plane validator remains unchanged.

Established container keys `spherical_planes`, `plane_antipodal`, `antipodal`,
`antipodal_reject` and `antipodal_evaluated`, and kind `q10_antipodal`, remain
for the existing UI/callers. Fixed `antipodal_mode="sphere_max_gap"` identifies
the new semantics; there is no method selector. `plane_antipodal` now contains
all `PlaneGapTest` records: raw/kept/discarded counts, kept masks, group count,
max gap, cleaned span/endpoints, PCL-style flag, reliability, edge-like verdict,
pre-trim span, tail-removed count/bin count and density reference/threshold.
The summary `antipodal` selects by rejecting first, reliable first, widest
cleaned span, greatest kept count, then lowest plane ID. Its retained/discarded
masks align with all `vector_analysis_indices` and are false outside that plane.
Old axis, B score and positive/negative counts/masks are removed. Original scene
NMS IDs/positions, Fusion, Q10 trace/support and analysis IDs remain immutable.
`geometry_analysis_indices` stays the union of plane memberships. Full-ROI
component IDs/all-true masks, `cc_evaluated=False`, NN/epsilon=None assert no
final spatial graph. Excluded angular samples remain in stored models/ROI.

Settings are centralized in `SettingsDefaults` / `Q10ValidatorConfig`. Existing
Q10/RANSAC controls remain. `q10_gap_min_points` and
`q10_gap_min_retained_fraction`, their resolved config/result metadata and CLI
flags are removed. RANSAC minimum new support/fraction and end-bin density
filtering remain unchanged. This detector change does not alter the standalone
lab's retained-count/fraction gates; angular grouping, tail masks and span
equations still match it. The old `q10_beta_deg`, `q10_antipodal_threshold`,
`q10_min_side_points`, `q10_sphere_axis_step_deg`, their CLI flags and the
`--q10-threshold` alias are removed rather than reinterpreted. New settings
resolve through matching `--q10-gap-*` CLI flags (tail toggle uses
`--[no-]q10-gap-tail`):

| Setting (`q10_...`) | Default / constraint | Meaning |
|---|---|---|
| gap_link_deg | 20; (0,180) degrees | neighboring azimuth connection gap |
| gap_edge_span_deg | 160; (0,360] degrees | cleaned span rejection, inclusive |
| gap_tail_enabled | True; boolean | trim sparse end bins |
| gap_tail_bin_width_deg | 10; (0,90] degrees | fixed-grid bin width |
| gap_tail_density_fraction | .5; [0,1] | occupied-bin median multiplier |
| gap_tail_min_bin_count | 2; integer >=1 | minimum supported end-bin count |

Manager-owned diagnostics read stored evidence only. Six Q10/path/support/
history overview panels and per-plane rose histograms remain. Raw cached
azimuths populate 36 ten-degree histogram bins; arcs use the corresponding
plane color and dim crosses mark discarded angles. Cards show cached
span, maximum gap, kept/raw counts, discarded/tail counts, reliability, RMS and
new support. Empty-arc evidence shows UNKNOWN; the removed count/fraction gate
text is omitted from the footer. Early Q10 rejection shows
RANSAC SKIPPED. No signed axes or antipodal cap shading remain. Plot basis,
circle colors and overlap memberships match the lab; projected display radii
remain 1.11+.025*k, circles 1.08. Display thinning affects only rendering.
`q10_spherical_plane_primitives` reconstructs arcs from cached endpoints in the
stored model basis. The native mm view retains one analysis cloud and .8R arcs
at the final origin. As of 2026-11-11, voxel display reads all cached plane
inlier/kept masks, independent of the selected summary: P1 orange (255,130,62),
P2 green (81,218,129), P3 purple (159,130,255), normalized by 255. A point with
multiple fitted memberships is white; a point discarded by all its member
planes uses 30% brightness. Retention in any member plane keeps it bright.
Blue denotes unscored context and gray denotes final directions without a
fitted plane. Display-only membership/retention matrices align to unique
analysis IDs; points are not moved or duplicated. Projected points/cleaned arcs
share the plane palette, with discarded projections/crosses dimmed per plane.
The Q10 arrow stays cyan; native/Tk titles and legends describe MaxGap and
these color meanings. Summary evidence and per-plane numerical masks/counts
remain unchanged.
Physical panels use equal mm scales; step navigation leaves final evidence
fixed. No diagnostics mutate results or rerun fitting, Q10 or MaxGap.

### Q10 relocation and cache contract

Dense threshold/NMS feeds the original seeds into `run_q10_validator`; the
public `lite_validator` result alias is retained. The canonical plane-validator
seed handoff is unchanged. The final angular check is described above.
`--q10-validator` follows shared settings in detector, validator, total and
new-validator; library configs retain their explicit optional Q10 field.

As of 2026-11-11, `q10_refine_enabled=True` / `--[no-]q10-refine` independently
controls relocation and the min-Q10 gate while `q10_validator_enabled` remains
the overall validator master. ON reproduces the previous path; the current initial switch follows settings.py. OFF
skips all nearest-analysis snapping, `refine`/`evaluate`, Q10 direction/quantile
computation, spherical Q10 CC, support-search gates and `q10_min_q10` comparison.
Keep final origin exactly at the original supplied NMS scene XYZ, even if no
analysis sample exists inside the NMS search ball. Use the unchanged global
analysis voxelization and radius-R full support for RANSAC/MaxGap. Its CORNER
means ACCEPT, REJECT means REJECT, UNKNOWN means INDETERMINATE; changing the
configured min-Q10 value has no effect in this mode. Configuration values still
undergo their normal type/range validation. No additional CC or physical-plane
stage is introduced.

OFF stores `refine_enabled=False` on each record and result, with effective
`refine_cc_enabled=False`, `q10_value=None`, `q10_reject=None`, and
`refined_analysis_index=None`; `refined_position_mm` equals original scene XYZ.
For the existing inspector, store a one-position trace, empty analysis IDs and
Q10 support, NaN Q10/axis/CC arrays, zero support/tested counts, actual angular
ROI count, `converged=False` and `stop_reason="refine_disabled"`. This does not
claim a successful Q10 search. Cache angular evidence by the exact XYZ tuple,
so different origins cannot alias through a missing analysis ID; equal origins
can reuse evidence. Q10 caches remain unused, early-reject count is zero, and
`antipodal_evaluated` describes the reached angular stage as before. Diagnostics
label refinement/min gate OFF, draw no min-Q10 line/arrow, and keep existing
cached MaxGap panels/colors. The configured threshold is retained in metadata
but is inactive. Master OFF still removes the whole validator stage.

With refinement ON, globally voxelize raw XYZ once at the configured
`q10_voxel_mm`. Preserve each original
scene NMS coordinate p0 and restrict all selectable analysis voxel centroids
to |p-p0|<=`nms_radius_mm` (+1e-8 mm boundary tolerance). Snap the baseline to
the nearest analysis centroid within that fixed ball. If none exists, keep
p0 as INDETERMINATE with `refined_analysis_index=None`, no analysis IDs in
the trace, and no CC/antipodal evaluation. `refiner.Q10PositionRefiner` owns relocation.
At each uncached analysis point p, query the full inclusive R-neighborhood
(`q10_radius_mm`, tolerance +1e-8 mm). Compute non-self median NN using all of
those analysis voxels and epsilon=`q10_cc_factor` times that median. Run one
epsilon graph inside the sphere, appending the exact query vertex and retaining
its component. Both NN and graph edges use voxel XYZ in mm. The query-anchored graph stays inside the spherical R region. The graph uses the shared
`connectivity.radius_components` implementation, moved without changing graph arithmetic. No halo or path outside R is admitted.

Only query-connected samples supply directions
u_i=(q_i-p)/|q_i-p| for distances >1e-7 mm. With >=10 connected samples,
>=3 nonzero directions and a defined selected axis, the unit_mean mode sets

$$m=\frac{\operatorname{mean}(u_i)}{|\operatorname{mean}(u_i)|},\qquad
Q(p)=\operatorname{quantile}_{0.1}(\operatorname{clip}(u_i^T m,-1,1)).$$

Use ordinary NumPy quantile/interpolation after spherical CC; no radial/density weights,
CED-axis sampling or antipodal objective enters Q. Cache Q, neighbor count
(including the query itself) and m by analysis ID for the frame's immutable
cloud, R, CC factor and direction mode. Cache the spherical component IDs, full ROI count,
median NN and epsilon too. Invalid responses are cached; degenerate spacing,
fewer than ten connected points or an undefined selected direction leaves Q invalid.
A new frame/cloud/R/CC-factor/direction-mode creates a new instance/cache; no approximate or
stale reuse is allowed. Existing three-argument `Q10PositionRefiner` calls remain
accepted, with a 4.0 CC-factor default; detector calls pass the configured factor.
`q10_refine_cc_enabled` follows current user tuning in settings.py. OFF skips both NN spacing and graph
construction, and uses the full-R support for the same mean/Q10 formula and
80% support guard. Cache scope additionally includes that fixed switch. Trace
support IDs then refer to full-R support and NN/epsilon are NaN; no connectedness
is asserted. The constructor accepts a trailing boolean `cc_enabled=True`.

Perform one search over that fixed original-NMS ball, recalculating each
candidate's spherical CC and Q10 when uncached. Require connected
count>=max(10,ceil(.8*initial count)). Select greatest Q, with reference 1e-9
tie tolerance, then smaller distance to p0 and analysis ID. Replace the baseline
only for strict improvement >1e-9. Never recenter or repeat the search around
the selected point. A valid completed search sets `converged=True` to mean
the fixed-ball selection is complete; it is not a moving-window local maximum.
More than 3500 fixed-ball candidates is INDETERMINATE, without silent thinning.
The trace stores the initial analysis position and at most one selected
replacement, along with their IDs, Q, connected counts and mean axes; tested-position
counts describe the single search. The retained `step_mm` field now denotes
this fixed NMS search radius. Total displacement from p0 is bounded by that
radius (+1e-8 mm), including the initial snap. `support_analysis_indices` stores
the spherical CC IDs per trace position. `neighbor_counts` now means connected
count including the query, `roi_counts` means full-R count, and trace arrays
`cc_median_nn_mm` / `cc_radii_mm` store the spherical CC spacing and epsilon.

The final angular evidence is evaluated once per distinct Q10-passing converged
analysis ID and cached. The first quality gate reuses trace Q10, never a Q1 or
post-final recalculation. `antipodal_evaluated` is false on Q10 early exits and
true when final evidence is requested/reused, even if that stage later reports
insufficient support. On early exits, `antipodal`/`antipodal_reject` stay None;
there are no fabricated plane masks or second failure reason. The sole known
failure is `q10_below_threshold`. `q10_early_reject_count` counts rejected seeds,
and `final_evaluation_count` counts distinct origins that reached evidence.
Q10 formulas, traces, passing-point RANSAC equations and scene/analysis IDs stay
unchanged. Original NMS coordinates are immutable; refined positions are separate.
The retained `raw_tree` argument is accepted but not consulted by Q10 evidence.
Debug is read-only: manager-owned plots/native views use stored Q10 support,
models, axes and masks, without rerunning gates or mutating numerical results.

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

## Removed detector experiments (2026-11-09)

The independent `--mode heatmap` multiscale experiment and Weighted Local Cone
backend are removed, including their configuration fields/CLI flags, exclusive
numerical helpers, rendering functions and tests. Shared inclusive Local Maximum
NMS, dense CED/bilateral/PCA/Fusion computations and heatmap screens,
Q10 relocation/spherical-RANSAC support and the standalone lab remain.
The Euclidean CC helper remains in `connectivity.py` for the Q10 refiner;
final spherical support does not use a graph. The native-view worker remains Q10-only, with its
existing spawned-process isolation, step messages and bounded shutdown.

The Scale/PCA Lite experiment was also removed on 2026-11-09: delete all eight
`lite_validator_*` defaults, `LiteValidatorConfig`, its `DetectorConfig` field,
CLI flags and composition branch, `lite_validator.py`, exclusive
`StableRadiusPCA`, scale plots/sliders/status styling and feature-only tests.
Retain dense `ced_weighted_pca`, Q10 refinement/CC/direction code and spherical
RANSAC final support. The Q10 result alias `lite_validator`, clicked inspector, status colors,
refined-coordinate markers and registered-RGB ACCEPT-only display remain.
`--no-q10-validator` now selects the unchanged dense/NMS detector without any
Lite validator. Fine/validator/total/ball pipelines keep their existing owners.

## Detector computation reuse and pruning (2026-11-09)

These optimizations preserve numeric scores, NMS IDs, refinement positions,
support masks, CC rules and all tie keys; configured radii, thresholds, voxel
sizes and Q10 direction modes are not changed.

1. Runtime detector mode omits `build_organized_plane_context`: it has no plane
   consumer for raw organized XYZ or pixel-to-scene nearest-voxel mapping. Total
   already omits it; fine, validator and ball modes retain their original input.
2. `edge_suppression` batches CED lookups in groups of 64. Once all original CED
   values exist, bilateral and PCA share a lookup at the larger of their radii
   and the same squared distances for each valid query, with at most 64 queries
   per batch. Keep unsorted scalar KD-tree traversal order with
   `return_sorted=False`; filter strict CED and inclusive bilateral/PCA supports
   exactly as before. Near a smaller sphere boundary (absolute difference from
   radius squared <= 64*float64 epsilon*radius squared), repeat the original
   smaller-radius scalar lookup before filtering: cKDTree/NumPy rounding can
   otherwise disagree at the boundary. Do not cache a whole-scene neighborhood
   graph. The bilateral timer includes the shared query/distance preparation;
   PCA time measures its kernel. Array/result/index contracts stay unchanged.
3. Q10 candidate evaluation first applies the existing required support count
   max(10,ceil(.8*initial count)). Reuse stored support/CC data when available;
   defer the quantile for insufficient support. Deferred does not mean invalid:
   keep that candidate out of the numerical response cache so another seed with
   a lower count requirement can calculate it, reusing the same support/CC.
   Normal cached scores and intrinsically invalid responses keep their semantics.
   Cache misses count uncached requests, including deferred requests.
Verification covers dense numeric fields against the prior three-pass
implementation and exhaustive Q10 evaluation across both Q10 CC settings and
all three direction modes, including floating-point sphere boundaries,
support-cache deferral and source dispatch. The removed projection-axis
pruning no longer runs. Saved-capture kernel timing is not a live sensor or GUI
test and excludes acquisition, Open3D voxelization and rendering.

Historical pre-removal benchmark (2026-11-09): the Projection rows below describe
the implementation that has since been deleted; they do not describe an active
Lite option or current spherical-RANSAC runtime.

Kernel benchmark on `captures/capture_20261107_083438_550378/cloud.npz`, using
`points_mm[::16]` converted to float64 (10,000 fixed raw samples), compares the
saved pre-change implementation with the optimized implementation. Timings are
medians of five alternating old/new executions. Dense uses CED/bilateral=45 mm,
PCA=67.5 mm, min-neighbors=5 and PCA saturation tau=.6. Projection uses R=45 mm,
Tube=10 mm, gap-factor=2.5 and 642 signed directions on 23 non-sparse ROIs.
Refinement uses 24 fixed seeds, R=45 mm, NMS=11.25 mm, centroid direction and CC OFF.

| Kernel | Before | After |
|---|---:|---:|
| Dense scores, all 10,000 samples | 1.516 s | 1.257 s |
| Projection, per candidate (batch average) | 7.874 ms | 3.243 ms |
| Refinement, per seed (batch average) | 1.281 ms | 1.265 ms |

All dense numeric fields, full Projection results/masks and refinement traces
match the saved implementation on these inputs. Refinement's support-gate
benefit is small on this input; gains depend on how many candidates fail that
gate. These are kernel timings on a fixed subsample, not whole-frame latency or
sensor FPS. Fifty-five focused numeric/dispatch/architecture tests pass.
The full suite runs 244 tests with 23 failures and 38 errors in the current
environment/configuration; no newly failing test names appear versus the
pre-change run. Open3D/device paths are not exercised here.

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
OFF still selects only angular-ACCEPT IDs at their exact NMS centers. It does not
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


## Saved dataset execution (2026-11-10)

`--source dataset --mode detector --input PATH` resolves a single saved
`points.npz` scene, a dataset root, or `corner_dataset_tool/dataset` beneath
the supplied tool directory. The existing `SceneData.load` validates XYZ/RGB,
pixel indices and saved annotation structure; `scene_sort_key` orders scenes
numerically. `dataset_input.run_dataset_detector` owns this input composition,
not any copied detector/Q10/validator algorithm.

`dataset_region` / `--dataset-region` is `roi` initially. It requires the saved
annotation AABB and uses its inclusive original raw-row selection. `scene`
retains all raw rows without requiring an ROI. Empty/missing/insufficient input
fails explicitly. Coordinates remain mm, RGB follows the same selected raw rows,
and GT corners do not affect scores, Q10, RANSAC or parameter choice. Pass the
selected original raw XYZ through the existing Open3D scene voxelization and
also as `raw_points_mm` for Q10's independently configured analysis voxelization.
Algorithms keep their native scene and analysis IDs; ROI-local IDs are not
original full-capture IDs, and virtual voxel centroids are never relabeled GT.

Every scene calls `run_dense_heatmap_mode` with the same configured detector/Lite
and manager-owned RGB/marker/inspector views. `--no-3d-view` suppresses display;
root execution processes every saved scene sequentially and releases its result
before proceeding. No results export, GT evaluation or automatic retuning is
introduced. Input points/ROI/labels/images are never rewritten. Synthetic depth
perturbation is rejected for this saved-XYZ source instead of reconstructing or
faking organized depth. Existing camera/STL/Kinect paths retain their behavior.

## Shared lab demo execution (2026-11-11)

`--source demo --mode detector` uses the exact standalone lab generator now
shared in `pointcloud_inputs.demo.make_demo_cloud`. NumPy `default_rng(7)`
creates 4,200 samples on each positive-octant coordinate plane: the two in-plane
coordinates are uniform in [0,85) mm, the perpendicular coordinate has Gaussian
noise with sigma 0.25 mm, and all samples are translated by (0,0,700) mm.
The original per-plane RGB is retained. Append the white known corner at
(0,0,700) as raw row 12,600, returning 12,601 float32 XYZ / uint8 RGB rows in
the unchanged order. The known corner ID is ignored by detection; it is not
inserted as an NMS seed or used for score/validator decisions.

The generator now accepts keyword-only `noise_sigma_mm=0.25`,
`roughness_amplitude_mm=0`, `roughness_wavelength_mm=20`,
`corner_fillet_1face_radius_mm=0`, `corner_fillet_2face_radius_mm=0`,
`corner_fillet_3face_radius_mm=0`, `edge_fillet_radius_mm=0` and `seed=7`.
These defaults preserve every original float32 XYZ/uint8 RGB row byte-for-byte.
For each plane's sampled in-plane coordinates u,v, its normal offset is

\[
h(u,v)=\epsilon+A\sin(2\pi u/w)\sin(2\pi v/w),\quad
\epsilon\sim\mathcal N(0,\sigma^2).
\]

Sigma and A are finite nonnegative mm values; w is finite positive mm; seed
is a nonnegative integer. Normal draws are still consumed when sigma=0, and
the sinusoid consumes no RNG values, preserving same-seed sampling and noise
realizations when strengths change. A=0 skips the bump addition entirely.
With all radii zero the appended white corner stays at (0,0,700) mm. The smooth product sinusoid
is zero on the two ideal origin edges; it models surface waviness, not a
calibrated sensor noise process or arbitrary stochastic roughness. Wavelengths
below the chosen analysis voxel spacing can be undersampled.

### Independent demo corner/edge fillets (2026-11-12)

The former common corner-radius keyword/control is replaced by independent
finite radii C1, C2, C3 in [0,42.5] mm. C1 rounds the three one-face outer
vertices, C2 the three two-face shared-edge endpoints, and C3 the central
three-face vertex. Edge radius E in [0,85] mm rounds the three shared edges.
Each may be zero independently; all zero skip rounding/clipping and preserve
the original geometry byte-for-byte. The corner upper bound is half the
85-mm face width, preventing opposite outline arcs from overlapping.
Setting C1=C2=C3 reproduces the previous common-corner-radius XYZ/RGB and
reference ID exactly, including noise/bumps and the maximum supported radii.

For each noiseless face sample p relative to (0,0,700), define D=||p||,
W=2*max(C3,E), t=clip((D-sqrt(2)*C3)/W,0,1) and
r=C3+(E-C3)*t²*(3-2*t). With C3 or E positive W is positive; otherwise
skip the surface deformation. Set
c=max(p,r) coordinatewise, n=(c-p)/||c-p|| and q=c-r*n; r=0 leaves p
unchanged. The implementation first scales c-p by r to avoid tiny-r norm
underflow. Thus the central spherical patch has radius C3, the distant edge
cylinders have radius E, and a cubic transition joins their differing radii.
With C3=0 the vertex/reference remains at the original origin while the edge
radius grows away from it; with E=0 the distant shared edges remain sharp.
The transition is a synthetic variable-radius blend, not a constant-radius
CAD rolling-ball fillet. Large settings can make it span much of the scene.
Where C3=E, the untrimmed surface matches the former constant-radius mapping.

For each face's original u,v, clip its other three corners k=(0,85),
(85,0),(85,85) using radii C2, C2, C1 respectively. For each positive local
radius Ck retain the sample only if sum(max(1-abs((u,v)-k)/Ck,0)²)<=1;
zero skips that cut. This is a circular outline cut in the original face
coordinates. The first two cuts round the two-face edge ends; the last rounds
the one-face outer vertex. Adjacent face cuts meet
at the same shared-edge parameter. The (0,0) outline corner is not clipped,
because the central spherical patch fills that three-face corner. No replacement
samples are drawn after clipping; the 3-D contours follow the surface deformation
where they overlap its transition. Unaffected straight boundary portions stay.

Apply the same pre-drawn scalar noise/bump height h along the actual surface
normal, obtained from the analytic tangents of q, including the derivative of
r. No additional RNG draws occur. Clipping uses noiseless original u,v and is
independent of C3, E, sigma and bump settings. C1/C2 change no retained sample's
coordinates or noise realization; C3 changes no clipping decision. Large C3/E
transitions can still geometrically overlap outer cuts. Curved patches inherit planar
sampling density and original sinusoid coordinates, not uniform surface-area
sampling or surface arc length. Nonzero C1/C2 remove samples, so raw row counts
and the final reference ID change; RGB and surviving sample order stay aligned.
For example, seed 7 and a single 20-mm corner radius enabled yields 12,440
rows for C1, 12,282 for C2, and the unchanged 12,601 for C3. With all three
at 20 mm there are 12,121 total rows, with reference ID 12,120.

Append the noiseless white surface reference
(0,0,700)+C3*(1-1/sqrt(3))*(1,1,1) at len(retained samples). The saved
`corner_001` uses this returned index; for C3>0 it is an inspection reference,
not physical sharp-corner GT. Existing captures/labels are not reinterpreted.
Detector scoring equations and no-argument detector demo output remain
unchanged; nonzero rounding intentionally changes input geometry and scores.

The lab Input tab exposes three separate 1-face/2-face/3-face corner sliders,
each 0--42.5 mm, plus edge radius 0--85 mm,
sigma 0--5 mm, amplitude 0--10 mm, wavelength
2--100 mm and editable integer seed, with an explicit Generate/apply button.
Controls do not regenerate or recompute on every slider update. Both Generate
and the existing top Demo button use the selected values. Exact defaults keep
the old demo_corner path/labels; all nondefault generations get fresh
demo_corner_* folders containing cloud.npz/cloud.ply, all four radii in metadata.json
and the known corner label. Repeated variants do not overwrite prior labels.
Raw camera/saved dataset inputs and all score/NMS/Q10/MaxGap algorithms are
unchanged. Detector --source demo continues to generate its original defaults;
no new detector CLI perturbation path is introduced.

Runner calls the existing scene voxelizer with configured `voxel_3d_mm`, then
the canonical dense detector with original raw XYZ for independently voxelized
Q10 support and aligned raw XYZ/RGB for managed visualization. Scoring, NMS,
thresholds, bilateral mode and Q10/MaxGap use current detector settings/CLI,
without adopting lab defaults. Generated data are in memory only; no camera,
depth reconstruction, label loading or captures-directory writes occur. Existing
default lab Demo saving remains unchanged. Unsupported non-detector modes, supplied
input files, organized depth-noise settings and insufficient voxel support fail
explicitly. `--only` and `--no-3d-view` keep their existing detector behavior.

## Standalone lab saved-scene input (2026-11-10)

`corner_lab_open3d_cone.py --scene N [--dataset ROOT]` opens the saved
scene_NNNN/points.npz by numeric ID, defaulting to the dataset beside the script.
The tool root containing dataset/ is also accepted. Invalid/missing scene IDs
and conflicts with --demo/--open fail before creating a GUI. --camera remains
the live RealSense entry point. --open also accepts dataset points.npz or its
scene directory; existing cloud.npz loading remains compatible.

The adapter loads every original raw row in mm and its aligned RGB; no saved
ROI crop or reprojection is applied, so original annotation point IDs remain
valid. When a lab corners.json does not exist, validated dataset annotations
are imported as gt_NNN labels with the original raw point index. Otherwise the
lab file's existing label/refinement semantics take precedence. Loading is
read-only; later explicit lab interactions retain the existing separate
corners.json saving path, leaving points.npz/annotations.json unchanged. Invalid
GT fails before replacing the current cloud. The existing geometry-only
Open3D importer is reused; no optional ML dependency or new fitting algorithm
is introduced. This changes standalone input/selection only, preserving all
Q10, voxelization, spherical fitting and coverage equations/thresholds.

### Standalone lab heatmap/NMS presentation (2026-11-11)

The lab additionally accepts `--dataset-region {scene,roi}` (default scene),
also controlled by `Dataset: saved ROI only` in the common header. Dataset ROI
uses the existing validated, inclusive saved AABB. Require annotations and at
least one selected raw sample. Raw XYZ/RGB storage and label IDs still index
the complete original capture; a separate selected-raw-ID map limits the cloud
display, picking KD-tree, shared analysis voxelization and whole-scene score
input. Picked tree rows map back through that array. Labels whose original raw
anchor lies outside the ROI are hidden and preserved on explicit lab saves.
Toggle/reload invalidates existing workers/caches through the existing scene
reload path. Failed ROI validation leaves the prior scene and checkbox state
intact. This does not modify points.npz, annotations.json or the saved ROI.
Legacy cloud/demo input remains complete. ROI-only scores use only ROI support,
so cropping can change neighborhoods and create extrema at the crop boundary;
there is no new boundary suppression or retained outside-ROI context.

The default `Heatmap/NMS` workspace associates each scalar field with the
extrema already computed for that field, using the existing full-scene or
local comparison result:

| Criterion | Cached field | Cached NMS indices |
|---|---|---|
| Fusion | `fusion_scores` | `fusion_maxima` (MAX) |
| Raw CED | `ced_scores` | `ced_maxima` (MAX) |
| CED-weighted PCA ratio | `ced_weighted_ratio` | `ratio_maxima` (MAX) |
| CED-weighted lambda1 | `ced_weighted_lambda1_mm2` | `weighted_lambda1_minima` (MIN) |
| Normal PCA lambda1 | `normal_lambda1_mm2` | `lambda1_minima` (MIN) |
| CED × local lambda1 minimum | `ced_local_min_scores` | `ced_local_min_maxima` (MAX) |
| Local lambda1 minimum | `local_minimum_scores` (L itself) | none (heatmap only) |
| CED × Median lambda1 minimum | `ced_median_min_scores` | `ced_median_min_maxima` (MAX) |
| Median lambda1 minimum | `median_minimum_scores` | none (heatmap only) |
| Harris 3D (NumPy) | `harris_scores` | `harris_maxima` (MAX) |
| CED normal component (PCA) | `ced_normal_component` (normal magnitude / R_CED) | none (heatmap only) |
| CED tangent component (PCA) | `ced_tangent_component` (tangent magnitude / R_CED) | none (heatmap only) |
| CED / sqrt(Normal PCA lambda1 + eps), selected Activation | `ced_inverse_std_scores` | `ced_inverse_std_maxima` (MAX) |

The sixth field is a separate contrast-regularized alternative to original
Fusion; it does not replace the five existing fields. Let lambda1(p) be the
largest eigenvalue of the existing unweighted Normal-PCA covariance in mm².
Let R_PCA=`pca_radius_mm` set the eigenvalue support and scale regularizer.
Use independent rho=`local_minimum_radius_mm` for the inclusive spatial
min/max neighborhood, defaulting to R_PCA when omitted. Among finite lambda1
values at evaluated voxel positions q within rho of p (including p), define
m(p)=min lambda1(q), M(p)=max lambda1(q).
Then compute

\[
L(p)=\frac{M(p)-\lambda_1(p)}{M(p)-m(p)+\tau_c R_{PCA}^2},\qquad
S_{\min}(p)=\operatorname{CED}_{raw}(p)L(p).
\]

`contrast_tau` (GUI `Local minimum contrast tau`) is dimensionless, finite and
strictly positive, initially 0.005. It is independent of the original Fusion
tanh saturation parameter `tau`. Increasing contrast_tau lowers nonzero
responses. Scaling geometry, both radii and PCA support by a multiplies
covariance eigenvalues and R_PCA² by a², preserving L and S when normalized
CED is unchanged. No minimum absolute
variance gate is added: a zero-valued local minimum can still score positively
if its neighbors differ. Constant eigenvalue neighborhoods give L=0; local
maxima give L=0. Invalid query lambda1 or CED yields invalid S, and invalid
neighbor eigenvalues are excluded. All arithmetic uses native mm/mm² values,
with no percentile or min-max preprocessing of eigenvalues or CED.

`compute_local_minimum_fusion` owns this equation; `with_local_minimum_fusion`
attaches new arrays and independent MAX NMS while preserving all original
arrays/indices/matches. NMS retains the existing full-scene or local support,
tie-breaking, local interior margin and local 25-extrema cap; remove zero-score
peaks after NMS. Full-scene computation remains cancellable. Min/max support
is the evaluated field: the selected dataset region for scene mode and the
evaluated local view in corner mode. These boundaries truncate neighborhoods;
no additional eigenvalue halo, boundary correction or interpolation is added.
Changing contrast_tau or the neighborhood radius changes the cache key and
requires recomputation. The neighborhood radius does not change eigenvalues,
the PCA scale regularizer, shared NMS radius or another criterion's statistics.

By default the new heatmap uses fixed [0,1] Viridis limits instead of data percentiles, so
small absolute responses stay dark rather than being stretched to high colors.
This is a display scale, not an additional score normalization. All invalid
new-score samples remain grey. Existing heatmaps retain their original scales.
The optional `Auto Contrast` checkbox, initially OFF and enabled for CED*L and
L itself, selects the existing finite-value P2--P98 display scale. Constant
values use the existing nonzero display-range fallback; all-invalid values
retain grey with [0,1] limits. The legend reports the scale and actual limits.
This changes only cached heatmap colors: no score normalization/recomputation
or changes to NMS/gates/matches, workers or the comparison key occur.

`Local lambda1 minimum | HEATMAP` shows the existing cached L(p) array without
CED multiplication. It uses the same fixed [0,1] / optional Auto Contrast
display scale. Selection redraws measured positions only, skips NMS gating
and disables NMS controls. No independent L extrema, threshold or pairings are
added; all nine scored/NMS criteria and their results remain intact.

### Additional Median-based local minimum (2026-11-11)

Keep the Min-Max score unchanged and additionally use the ordinary NumPy
median d(p) of finite lambda1 samples in its own inclusive radius-rho_med ball,
where rho_med=`median_minimum_radius_mm` defaults to R_PCA:

\[
L_{med}(p)=\max\left(0,\frac{d(p)-\lambda_1(p)}
 {d(p)-m(p)+\tau_{med}R_{PCA}^2}\right),\qquad
S_{med}(p)=\operatorname{CED}_{raw}(p)L_{med}(p).
\]

Here m is the minimum in that same Median neighborhood, independently of the
Min-Max neighborhood. For even sample counts d is the average of the two middle values. Above-median
queries and constant neighborhoods have zero response. Invalid query lambda1
remains invalid; invalid neighbors are excluded. L can be valid where query CED
is invalid, but S remains invalid there. No absolute minimum-variance gate,
ratio factor, tanh or other weighting is added to this equation.

`compute_local_minimum_fusion(..., statistic='median')` shares the original
validation, radius support and cancellation, taking an explicit Median tau
from `with_local_minimum_fusion`. `median_contrast_tau` / GUI `Median contrast
tau` is finite, strictly positive and initially 0.005; it is independent of
Min-Max's `contrast_tau` (also initially 0.005). Both taus and both neighborhood
radii join comparison cache keys and are snapshotted for full/ROI and local
workers. Changing any requires recompute.
Derived results cache Median L/S, positive-only MAX NMS and its tau in separate
fields, preserving every previous scalar array/extremum/match. The Median NMS
uses the same existing radius, finite-support, tie-breaking, local interior
margin and local top-25 rules, with zero peaks removed after NMS.

### Independent lab Min-Max/Median neighborhood radii (2026-11-12)

Both comparison entry points and `with_local_minimum_fusion` accept
`local_minimum_radius_mm=None` and `median_minimum_radius_mm=None`. None
resolves separately to the current PCA radius; explicit radii must be finite
and strictly positive mm values. Results store the two resolved float radii.
The Heatmap/NMS UI exposes `Min-Max neighborhood radius` and `Median
neighborhood radius` text fields; blank means follow PCA, including later
PCA-radius changes. Explicit values remain independent when PCA changes.
Validate both inputs before starting the worker, include both in full/local
cache keys, snapshot them into either worker path, and discard results if
the UI settings change during calculation. Summaries report the resolved
neighborhood radius alongside the unchanged PCA radius and tau.

These radii select finite cached Normal-PCA lambda1 observations only.
`compute_local_minimum_fusion(..., regularization_radius_mm=R_PCA)` keeps
tau*R_PCA² while selecting observations using its `radius_mm` argument.
Direct callers omitting the new keyword retain the original tau*radius_mm²
behavior. Min-Max's radius changes only Min-Max L/S and its derived NMS;
Median's radius changes only Median L/S and its derived NMS. Input XYZ,
Normal/weighted PCA, CED, original Fusion, Harris, shared NMS settings and
label IDs remain unchanged. No new support halo is added: the selected
scene/ROI or evaluated local view still bounds the available eigenvalue
samples, even if the chosen neighborhood radius is larger. Defaults and
explicit radii equal to PCA preserve all previous numerical arrays/indices.
This is a standalone lab control; the main detector CLI/config still uses
its existing PCA-linked local-minimum radius.

Median L is a standalone heatmap-only criterion; Median CED*L has independent
MAX NMS and an inclusive lower score threshold. Both use fixed [0,1] colors by
default and optional display-only Auto Contrast. They retain measured XYZ,
region/analysis IDs, ROI support truncation and invalid grey colors. Existing
ratio diagnostics, paired comparisons and Q10/MaxGap are unchanged. This
implementation/test work establishes the equation and its outlier response;
it does not establish RGB-D localization accuracy or resolve line-like Edge
minima versus point-like Corner minima.

`NMS score thresholds` provides an independent optional limit for each NMS row.
Blank disables the gate; finite nonnegative values use inclusive `score >= t`
for the seven MAX fields and `score <= t` for the two MIN fields. Lambda1
thresholds are mm²; inverse-std thresholds use 1/mm; the other thresholds are
dimensionless. Apply validates all nine inputs before committing, filters
existing extrema into a new result and
reruns the existing nearest-first, one-to-one matching on the surviving lists.
Thus another eligible pair can replace a removed match. Base scalar arrays,
unthresholded extrema, positions, workers and heatmap scales remain unchanged.
Changing thresholds is a post-NMS gate with no new NMS or top-K pass. Counts
show selected-field kept/original NMS and filtered comparison/pair totals.

The five original heatmaps retain the existing Viridis 2nd--98th percentile display scale,
grey invalid samples and original positions; lambda1 units remain mm².
Registry lookup uses `colormaps['viridis']` for compatibility with the installed
Matplotlib. Cyan markers show the selected criterion's extrema. Independent
heatmap/NMS visibility toggles change geometry only; cached values, NMS indices
and reported counts remain intact. The collapsed paired-method option retains
the original Fusion-only or CED-plus-PCA comparisons, pink secondary extrema
and green matched pairs with unchanged distance matching.
Matched pairs use a 28-pixel lime midpoint marker, larger than the 15/13-pixel
primary/secondary markers, and a 6-pixel green link (all scaled by window DPI).
The midpoint remains visible when both extrema share a position and the link
has zero length. Marker centers, pair endpoints, distances and counts stay
unchanged; the enlarged marker footprint is a display cue, not a match radius.
Full-scene heatmaps hide the underlying RGB cloud while visible, restoring it
when cleared/hidden; local views retain RGB context for raw-point depth picking.
The shared CED radius stays editable in unweighted PCA refinement mode because
it still controls the independent CED/Fusion comparison fields. Dynamic summary
updates request layout again to prevent text overlap.

`Corner` shows the existing cloud/unit-sphere/histogram inspection layout and
Q10, spherical-plane and cleaned MaxGap controls. `PCA` retains optional
eigen-field refinement, while `Input` holds camera/capture controls. Switching
tabs only changes layout/inspection visibility. Outside the visible NMS inspection
path in Heatmap/NMS, successful Ctrl+click selects the Corner tab after the
existing label/inspection operation. Previous score equations,
extrema computation and thresholds, Q10 refinement, spherical RANSAC, MaxGap,
worker/cache behavior, raw/analysis IDs and saved-label semantics are unchanged.
Optional unit-sphere/histogram widgets are registered after permanent controls,
keeping the tab panel's Open3D/ImGui window ID stable when their visibility
changes. Registering them first changes visible-child numbering and resets tab
selection. The tab callback does not reassign an already selected native tab;
programmatic navigation assigns the index only when it differs. Tab switching
does not cancel/restart an active computation or replace its cached result.

### Lab raw CED and ordinary-PCA inverse-standard-deviation score (2026-11-12)

Every standalone lab comparison criterion now uses ordinary raw CED,
CED(p)=||mean(radius-neighbors)-p||/R_CED, with the existing inclusive lookup
and minimum of three points. `ced_scores` stores this field without bilateral
smoothing. Delete the full-scene bilateral pass and local per-query filter;
retire their `sigma_ced` argument/result field and label the existing CED row
`Raw CED`. Keep raw CED support/NaNs, positional PCA, raw-CED-weighted PCA,
Harris, CED component decomposition, input positions and label IDs unchanged.
The main detector CLI's bilateral implementation/settings remain unchanged.

Original lab Fusion becomes raw CED*tau*tanh(weighted_ratio/tau), retaining
its original support, variance gate, clipping and tau. Min-Max and Median
products become raw CED*L using their unchanged L fields, independent radii
and tau*R_PCA² regularizers. CED-dependent scores, their NMS and associated
pairings intentionally change; NMS algorithms/settings are unchanged.

The base ratio evaluates the supplied equation literally:
r(p)=CED_raw(p)/sqrt(lambda1_NormalPCA(p)+epsilon).

Ordinary PCA is the unweighted centered covariance
C=mean((x-mean(x))*(x-mean(x))^T) in the existing PCA-radius ball, with the
existing positional min_neighbors. lambda1 is its largest eigenvalue in mm²,
read from `normal_lambda1_mm2`. Do not use weighted eigenvalues, the smallest
normal eigenvalue, local-minimum statistics or lambda1/R_PCA². Epsilon is
inside the square root and is finite positive mm², initially 1e-6.

`compute_inverse_std_fusion` evaluates aligned cached arrays. Defined zero CED
produces zero; lambda1=0 uses sqrt(epsilon). Nonfinite/negative CED or lambda1
stays NaN. No new normalization, clipping, saturation or variance gate is
introduced. Base r has units 1/mm and may exceed 1. Increasing epsilon lowers
positive scores. Scaling positions/radii by a and epsilon by a² scales S by
1/a. With Activation None, S=r exactly. `with_inverse_std_fusion` attaches the selected `ced_inverse_std_scores`, positive-only
`ced_inverse_std_maxima` and resolved `inverse_std_epsilon_mm2` to both local
and full results. Keep shared MAX NMS radius/ties/support rules, local interior
margin/top-25 cap, and remove zero responses after NMS. Valid inverse-std
output remains available when weighted PCA/Harris fields lack support; it
still requires the existing ordinary-PCA and CED support.

Heatmap/NMS appends the criterion and an epsilon field. Both cache keys and
worker paths snapshot epsilon; validate before starting a worker and discard
output after mid-job setting changes. Reuse ordinary finite P2--P98 Viridis
colors, gray invalid values, an independent inclusive lower threshold in
1/mm, and the cached Ctrl+NMS-click plot with MAX direction and units. The
summary reports raw CED, unweighted largest eigenvalue, epsilon and actual
color limits. No new neighborhood pass, covariance fit, position refinement
or main detector CLI/config path is introduced. Recompute after changes.

### Selectable inverse-std score Activation (2026-11-12)

`INVERSE_STD_ACTIVATIONS` lists the lab's options in one place, shared by the
selector and validation. `apply_inverse_std_activation` owns the pointwise
implementations. Add a future option there and an explicit implementation
branch; an unimplemented/unknown option fails instead of silently selecting
another function. This applies only to the CED/ordinary-PCA inverse-std
criterion; other scores and the main detector CLI remain unchanged.

| Activation key | Definition | Output scale |
|---|---|---|
| none (default) | S=r | Existing unbounded 1/mm ratio |
| exponential | S=1-exp(-r) | [0,1] |
| median_density | S=1-exp(-r*N_median/N) | [0,1] |

Compute exponential outputs with `-expm1(-x)` for small-x accuracy. Saturation
may reach 1 in floating point. Keep original r in `inverse_std_raw_scores`;
`ced_inverse_std_scores` always contains the selected output used by heatmap,
MAX NMS, thresholding and the cached click plot. Apply Activation before the
existing NMS and positive-score filter, rather than recoloring old peaks.
Density weighting can change peak order; exponential saturation/tolerance
can also affect floating-point NMS ties. None preserves all previous arrays
and NMS results exactly. Do not invent a new learned probability/calibration.

The user-selected density definition uses the current PCA radius for both
counts and median support. `pca_neighbor_counts[i]` is the geometric number
of observed voxel support samples in query i's inclusive PCA ball, including
the query itself, recorded during the existing PCA lookup before any
min_neighbors, CED-weight/cutoff or PCA-validity test. No extra lookup is
needed for this count. For `median_density` only,
`compute_neighbor_count_medians` looks up neighboring evaluated Query rows
within the same radius (+existing 1e-8-mm lookup tolerance), excludes only the
current Query row, and takes the ordinary unweighted median of positive finite
counts. Other coincident Query rows remain. Include count evidence even where
the neighboring Query's score/PCA is undefined. Even-sized medians average
the two middle values. There is no count floor, pseudocount, mean, absolute
density threshold or fallback median.

Results cache N in `pca_neighbor_counts`, selected mode in
`inverse_std_activation`, and N_median in `inverse_std_neighbor_medians`.
Medians are empty when unused; they are recomputed from the supplied result
when the density mode is calculated and then read-only during display.
Undefined raw ratios remain NaN. Nonpositive/nonfinite N or N_median, or no
other eligible Query in the ball, leaves density-activated S undefined/gray.
Defined r=0 with valid evidence remains zero. Scaling all counts by a common
positive factor preserves the ratio and Activation for fixed r; the base r
can still change with sampling, geometry and units. Use literal numerical r
in the existing mm convention: relative density does not make the full score
scale invariant or add an activation temperature.

Full/ROI mode's median population is the compared voxel Query set. Local
mode's N uses the unchanged halo-backed positional-PCA support, while the
median population is the returned local-view Query set. Thus local/ROI/scan
boundaries can truncate the median neighborhood; no new Query-count halo is
added. If an entire corner region is similarly sparse, N_median/N can remain
near one. The method does not certify physical corners or eliminate that
limitation. The median pass checks cancellation and stores no neighbor graph.

Heatmap/NMS adds an ACTIVATION combobox under the ratio epsilon. Default None
retains the original percentile scale and 1/mm units. Exponential modes use
fixed [0,1] colors initially and the existing optional display-only Auto
Contrast; summary/legend/click plots identify the mode and omit 1/mm units.
Both full/local cache keys and workers snapshot the mode. A mode change
cancels/invalidates the comparison and clears only this criterion's threshold,
since its scale changes; keep other gates intact. Recompute with Compute/show.
Subsequent redraws only read cached output/evidence. The NMS gate acts on the
selected score. Other CED/PCA/Harris/label calculations are unchanged.

### Lab CED normal/tangent component magnitude heatmaps (2026-11-12)

The two additional criteria show scalar component magnitudes of the same raw
CED vector, replacing the earlier signed-XYZ direction-color interpretation.
During each existing raw-CED lookup retain d=mean(neighbors)-query in mm, with
the same CED radius R_CED and minimum of three samples. During the existing
unweighted Normal-PCA pass retain its centered positional covariance C and
compute a separate `eigh(C)` for the smallest-eigenvalue eigenvector n. Keep
the original scalar `eigvalsh` call and its arithmetic unchanged. Normal
support uses the existing PCA radius and positional min_neighbors, independently
of CED weights/cutoffs, weighted PCA and Harris normal estimation.

`direction_fields.pca_normal_direction` requires lambda_middle>epsilon and
lambda_middle-lambda_smallest>epsilon, with
 epsilon=64*float64_eps*max(lambda_largest,0). Collinear/coincident support and
a tied smallest eigenspace give undefined normals, without changing any old
scalar validity/score/NMS decision. Sharp multi-face support need not have a
physical single-plane normal; the decomposition follows the estimated plane.

`ced_component_magnitudes` normalizes each finite nonzero supplied normal to
unit n, then computes the nonnegative dimensionless scalar fields

C_normal=abs(d dot n)/R_CED,
C_tangent=||d-(d dot n)*n||/R_CED.

The vectors d and its tangent projection are not normalized to unit length.
Consequently smaller CED offsets produce proportionally smaller component
values. For defined planes, zero offsets yield two valid zeros; pure normal
or tangent offsets yield zero in the other component. Both fields stay NaN
when the CED support or PCA normal is undefined. There is no minimum tangent
length gate. Normal sign/scale and joint rotations do not change magnitudes;
scaling positions and R_CED together also preserves them. The identity
C_normal²+C_tangent²=||d||²/R_CED² holds up to floating-point error. The right
side equals the squared raw scalar `ced_scores`.
No componentwise bilateral filtering, clipping or new score/NMS is introduced.

`ced_normal_component` and `ced_tangent_component` are shape-(N,) arrays
aligned with `result.positions_mm`. Local mode uses the original ordered query
subset and existing CED/PCA support halos; full/ROI mode uses all compared
voxels. No new neighborhood lookup or parameter is needed. An extra 3x3
eigendecomposition supplies normal evidence per valid positional-PCA query.
The original centroid/scalar eigenvalues, all previous score/NMS arrays,
thresholds, matching, labels/IDs and main detector CLI remain unchanged.

Both components use the existing scalar Viridis heatmap path: purple low,
yellow high, fixed [0,1] limits initially, gray (95,103,115) for NaN. Thus
normal/tangent strengths share a color scale. Existing optional Auto Contrast
uses each field's finite P2--P98 limits for display only; enable it with care
when comparing absolute strengths between the two components. Summary/legend
report the selected component formula, defined count, CED/PCA radii and actual
color scale. The former direction RGB mapper is removed.

These are heatmap-only criteria without NMS markers, threshold inputs or NMS
peak plots. Selection skips scalar NMS gates and clears paired-method display;
redraw reads cached scalar magnitudes and never recomputes PCA or components.
Hide/clear restores the existing RGB context. Restart/recompute to populate
the new scalar caches. The first ten criteria and worker/cache keys retain
their existing semantics.

### Lab Harris 3D and cached NMS neighborhood plot (2026-11-12)

`corner_detector.harris3d.compute_harris3d` owns a NumPy implementation of the
[PCL HARRIS response equation](https://pointclouds.org/documentation/harris__3d_8hpp_source.html),
separate from positional PCA/CED/Fusion. In mm, estimate each support point's
unit normal as the smallest-eigenvalue eigenvector of its centered unweighted
radius-R covariance. Require at least three samples and second eigenvalue
>1e-12 times the largest, with largest>0; reject coincident/collinear support.
R is the lab's existing PCA support radius for both normal estimation and
response support; no separate hidden radius or CLI setting is introduced.
Use finite normals in the inclusive response ball, with at least three,
and require the query's normal to be defined. With M=mean(n*n^T), compute
H=0.04+det(M)-0.04*trace(M)^2. M is uncentered; normal signs do not matter.
Unit normals give trace(M)=1 up to roundoff. Ideal planar/two-orthogonal-normal
support gives zero, and balanced three-orthogonal-normal support gives 1/27.
Undefined responses stay NaN. No CED cutoff, weighting, tanh or positional
covariance determinant enters Harris. Scale/rigid rotation leave H unchanged
when radius/support are transformed consistently; H is dimensionless.

`with_harris3d` attaches aligned `harris_scores` and MAX `harris_maxima` to both
comparison entry points via dataclass replacement. Keep the existing full-scene
or local NMS, tie-breaking, local interior margin/top-25 behavior, then remove
H<=1e-12 numerical-zero peaks. The GUI adds Harris as criterion ten, including
its own inclusive post-NMS lower threshold and ordinary P2--P98 colors. All
other scalar arrays/extrema/matches and their arithmetic remain unchanged.
Harris has no position refinement, and is labelled NumPy: PCL normal solver,
float precision, degenerate-support rules and native NMS parity are unverified.
The saved dataset PCL backend remains separate and unchanged.

Full-scene Harris uses all compared voxels without a whole-scene neighbor
graph. Local Harris uses original analysis samples out to view radius+2R,
so both its normal and response neighborhoods have halo support; returned
responses address exactly the existing ordered local query positions. This
extra support does not change the old scoring crop. Both computation passes
check cancellation. Computation adds normal-eigendecomposition and normal-
moment radius passes. A valid Harris or inverse-std field permits comparison even when the
legacy Fusion support-count check fails; otherwise the insufficient-
support error remains. Existing positional scores/gates are not relaxed.

In Heatmap/NMS, Ctrl+left click with NMS markers visible keeps the overlay
active and depth-picks the nearest displayed primary/secondary NMS voxel within
12 mm, honoring current post-NMS thresholds. Coincident paired fields prefer
the displayed heatmap criterion. IDs address `result.positions_mm`, never the
raw label tree. No raw-point snapping, label writes or Corner-tab navigation
occurs. Outside that path, existing Ctrl+click labelling/inspection remains.
Ignore callbacks after the result, criterion or workspace changes.

`visualization.manager.show_nms_peak_plot` owns rendering/window dispatch;
`plots.make_nms_peak_figure` produces a distance-vs-score scatter and histogram,
and `view3d.open_nms_peak_view` reuses one native image window. The orange marker
identifies the picked position without altering it. Selection uses cached
scores only: no scoring, voxelization, NMS or refinement is rerun. Redraw,
hide/invalidation and parent close dispose the diagnostic; closing its own
window allows the next click to recreate it.

The plot compares all finite samples in the selected NMS-radius ball, inclusive
with the lab's 1e-8-mm lookup tolerance, excluding only the selected row (other
coincident rows remain). Raw scores retain original units. Let a=+1 for MAX,
-1 for MIN, selected score s, neighbor scores v, count m, mean mu and population
standard deviation sigma. Show rank=1+count(a*v>a*s), percentile=
100*(count(a*v<a*s)+0.5*count(v=s))/m, mean advantage a*(s-mu), relative mean
advantage 100*a*(s-mu)/abs(mu), best-neighbor margin a*s-max(a*v), and
z=a*(s-mu)/sigma. Positive advantage always means better, including MIN.
Empty support makes every comparison N/A; mu=0 makes relative percent N/A;
sigma=0 makes z N/A; equal-score neighbors have percentile 50 and margin zero.
Near-zero means can produce large percentages. These are local contrasts,
not detection probabilities or physical-corner confidence. Actual scan/ROI
boundaries can truncate support; diagnostics add no boundary correction.

## Detector Min-Max Local Minimum Fusion and post-NMS edge gate (2026-11-11)

`fusion_mode={pca_ratio,local_minimum}` selects a dense product, independently
of the existing bilateral stage. `pca_ratio` preserves the original CED ×
optional-tanh weighted-PCA ratio, support rules and NMS; the current user
setting selects `local_minimum`.
For `local_minimum`, compute ordinary Normal-PCA covariance at each voxel p:

\[
C_N(p)=\frac1{n}\sum_{q\in B_R(p)}(q-\bar q)(q-\bar q)^T,
\quad a(p)=\lambda_{\max}(C_N(p)),\quad R=R_{PCA}.
\]

Require configured min_neighbors geometric samples in the inclusive ball;
otherwise a(p) is undefined. Normal eigenvalues are evaluated even where query
CED/weighted PCA is invalid, since they can support a neighboring query's local
range. Among finite a(q) at scene voxels within inclusive R of p, let m and M
be the minimum and maximum. Then

\[
L(p)=\frac{M(p)-a(p)}{M(p)-m(p)+\tau_LR^2},\quad
S_{raw}(p)=CED_{selected}(p)L(p),\quad \tau_L=0.005.
\]

This is the lab's **maximum-based** equation, not its Median variant. R serves
both covariance/local-range support and the mm² regularizer. No tanh, weighted
lambda ratio, minimum absolute variance or effective-support condition is
added to L. Constant neighborhoods and maxima have L=0. Invalid Normal lambda1
neighbors are ignored. Fusion requires valid query CED and finite L; weighted
PCA need not be valid unless the optional later gate is enabled. Dense fields
store zero for invalid L/S with separate masks; raw Normal lambda1 retains NaN
for invalid support. Scaling all geometry/radii by k scales a and R² by k²,
preserving L, Fusion and normalized gate decisions apart from floating error.

The dense scorer owns Normal PCA/local min-max, reusing the bounded shared
neighborhood pass and adding a bounded radius pass after eigenvalues exist.
Original CED, weighted covariance/cutoff, ratio diagnostics, coordinates and
scene IDs are preserved. `bilateral_mode=ced` uses existing filtered CED;
off uses original CED; fusion filters the selected complete product with the
existing valid-support/range/spatial rules. `fusion_score_raw` is that selected
pre-filter product, `fusion_score` is the final score consumed downstream.

Settings/CLI expose `fusion_mode`, `local_minimum_tau` and
`local_minimum_fusion_threshold` (0.2). Legacy `fusion_score_threshold` remains
its own setting. Configuration resolves the appropriate active threshold into
DetectorConfig; strict score>threshold and existing fixed/global-mean selection
remain unchanged. Thus 0.2 is the new mode's configured fixed default, not a
new lower bound for the optional adaptive threshold. Shared detector/validator/
total/new-validator/ball and comparison callers propagate the same mode/tau.

After unchanged thresholding/local-max NMS, optional
`post_nms_lambda1_ratio_max=t` (None=OFF; current user setting 0.20) applies

\[
\lambda_{1,W}(p)/R_{PCA}^2\le t
\]

only at NMS survivors, using the **CED-weighted** largest covariance eigenvalue
already cached by the existing PCA kernel. Values are dimensionless; configured
t must be finite in [0,1]. Equality passes. Invalid weighted PCA/nonfinite or
negative lambda1 rejects when enabled, never passing as an artificial zero.
No NMS rerun, neighbor promotion, new covariance, position change or rescoring
occurs. This gate applies to either Fusion mode and is independent of both
Fusion thresholds and Q10/MaxGap. It is an experimental edge-response filter,
not proof that an accepted point is a Corner.

`nms_candidate_indices/mask` retain original NMS scene IDs. The existing
`candidate_indices/mask/candidates/count` refer to gate survivors in unchanged
NMS order. `post_nms_lambda1_gate` records enable state, threshold, raw NMS IDs,
aligned normalized lambda1, pass/invalid masks and rejected IDs. Kept candidate
records carry the mode and, when gated, the normalized value/threshold.
Q10 and canonical validator seed handoff consume only these gate survivors.
Gate OFF preserves previous candidates and requires no extra eigenvalue data.

Managed visualization retains six panels, substituting cached L for the
ratio-used panel in local_minimum mode. Existing linear Q99 display scales and
raw scene-indexed positions remain; color maps do not alter score/gate values.
Console logs distinguish mode, actual threshold, raw NMS and retained counts.
No real sensor performance or optimal gate threshold is established here.

## 14. Revision record

| Date | Revision |
|---|---|
| 2026-11-12 | Added selectable inverse-std Activation: None (default raw ratio), exponential, and 1-exp(-r*N_median/N). Use PCA-radius geometric counts including self and positive finite neighboring Query-count medians excluding self; cache raw ratio/counts/medians and selected output. Apply before MAX NMS, update display/plot units and fixed-[0,1]/Auto Contrast semantics, include mode in cache/worker snapshots, and clear only its gate on mode changes. Preserve all None-mode and other-criterion numerics; add exact density/median/stability/rank/default/UI/worker regressions. Local/ROI median boundary truncation remains explicit. |
| 2026-11-12 | Added S=raw CED/sqrt(largest unweighted Normal-PCA lambda1 + epsilon), heatmap/MAX NMS/threshold/click diagnostics and editable finite positive epsilon (default 1e-6 mm²) across full/local results, caches and workers. At user request, replace all lab bilateral CED uses with ordinary raw centroid-distance CED, remove filtering/sigma_ced and relabel CED controls. CED-dependent scores/NMS/matches intentionally change; preserve positional/weighted PCA, Harris, component magnitudes, NMS algorithms, coordinates/labels and the separate main-detector bilateral path. Add formula/unit/invalid/epsilon/cache/worker/display and direct raw-CED product regressions. |
| 2026-11-12 | Corrected the two new lab views to scalar CED normal/tangent component magnitudes relative to the estimated unweighted PCA normal. Replace direction vectors/RGB colors with abs(d dot n)/R_CED and tangent length/R_CED, fixed shared [0,1] Viridis and optional existing Auto Contrast; valid zero offsets remain zero. Remove the direction RGB mapper; retain existing scores/NMS/IDs and no component NMS/gates. Test exact magnitude/zero/strength, Pythagorean/sign/rotation/scale invariants, plane support and cached scalar display. |
| 2026-11-12 | Added lab CED tangent and unweighted positional-PCA normal direction caches/heatmaps using existing lookup support. Preserve raw centroid/scalar eigvalsh arithmetic and all original scores/NMS/IDs; reject degenerate normal eigenspaces and undefined projected tangents only in the new fields. Add manager/view3d fixed signed-XYZ RGB with sign-canonical normal colors, gray undefined evidence and direction-only controls without NMS/gates/Auto Contrast/recomputation. Test expected plane/boundary directions, projection/rotation/sign/scale/color invariants, local/full alignment and cached display/no-mutation behavior. |
| 2026-11-12 | Added separate lab Min-Max/Median lambda1-statistic neighborhood radii, blank/None following PCA by default. Propagate resolved radii through comparison entry points, result metadata, cache keys, GUI validation, worker snapshots and summaries. Preserve positional eigenvalue support, tau*R_PCA² regularizers, shared NMS settings, all other criteria and exact default local/full arrays. Add exact independent-radius/scale/regularizer numerics, cache/input/worker and unchanged-field regressions; existing evaluated-field boundary truncation remains. |
| 2026-11-12 | Split the demo common corner radius into independent 1-face outer-vertex, 2-face edge-end and 3-face central-vertex keywords/sliders, each finite [0,42.5] mm. Keep edge radius independent; C1/C2 control only their outline cuts, C3 only the existing central/edge blend and reference position. Preserve equal-three-radius parity, zero/default geometry, noise/RNG, retained RGB/order and returned label IDs. Save all four radii separately; add isolated-type numerical/count/region tests, equal-radius hash regression and independent GUI/save/validation checks. |
| 2026-11-12 | Added lab Harris 3D (NumPy PCL-response equation, PCA-radius normals/response, separate positive MAX NMS/gate and local 2R support halo) as criterion ten. Preserve old fields/NMS/labels, while permitting valid Harris output when Fusion support is insufficient. Added manager-owned cached NMS click plots with score-distance/histogram, direction-aware rank/percentile/mean/best/z contrasts, native window reuse/cleanup and voxel-ID-safe picking without recomputation or label writes. Added numerical/support/scale/rotation/degenerate, cached display, threshold, callback and window lifecycle tests; native PCL/GUI/hardware parity remains unverified. |
| 2026-11-12 | Split the demo shared fillet into independent corner [0,42.5] and edge [0,85]-mm controls/keywords. Round all one-face outer corners, two-face edge ends and the three-face origin, using outline cuts plus a cubic central-to-edge radius blend and analytic surface normals. Preserve both-zero/default detector output and RNG draws; retain surviving row/RGB order, return the post-clip reference ID, save both radii in fresh variant metadata, and test independent controls, all corner types, exact constant-radius patches, transition tangents, variable counts and save/label isolation. |
| 2026-11-12 | Added shared-radius cylindrical edge and spherical corner fillets to the lab Demo surface, with finite [0,85]-mm validation, normal-directed noise/bumps and a diagonal on-surface reference row. Preserve radius-zero XYZ/RGB/seed/default detector behavior; GUI applies explicitly and saves radius metadata in separate variant folders. Reuse the geometry-only Open3D loader for standalone startup. Add exact radius/flat-face/reference/noise/seed and GUI/save-isolation regressions. |
| 2026-11-11 | Removed detector Q10 MaxGap minimum retained-count and retained-fraction gates, their settings/config/CLI/result fields and debug footer. Every nonempty cleaned arc now uses span alone; no-plane/empty-arc UNKNOWN and rejecting-plane precedence remain. Preserve RANSAC support gates, grouping, tail trimming, span equality, Q10 refinement, scores/NMS and scene IDs. Add small/minority/single-point/empty-arc numeric and cached-display/removed-option regressions; standalone lab gates remain unchanged. |
| 2026-11-11 | Added selectable detector pca_ratio/local_minimum Fusion with Lab Min-Max Normal-PCA L, tau 0.005 and separate new-mode fixed threshold 0.2. Added optional post-NMS CED-weighted lambda1/R_PCA² upper gate (None=OFF), preserving raw NMS IDs and filtering before Q10/seed handoff without neighbor promotion. Retain user-selected local_minimum mode and gate 0.20. Propagate shared config/CLI/callers, cached managed panels/logs, and test equation/covariance/scaling/gate equality/order/validity/legacy regressions. |
| 2026-11-11 | Set both lab Min-Max and Median contrast tau defaults to 0.005 across GUI, result metadata and numeric entry points. Preserve independent overrides and all equations; add exact default-score and GUI-default assertions. Only default Min-Max responses/NMS may change; Median defaults stay unchanged. |
| 2026-11-11 | Added GUI Demo surface controls for normal Gaussian sigma, sinusoidal bump amplitude/wavelength and seed through the shared generator. Preserve exact no-argument XYZ/RGB/corner-ID defaults and detector demo behavior; apply only on Generate/Demo, save modified generations separately with parameter metadata and fresh labels, and leave camera/scoring/validator paths unchanged. Add numeric shape/noise/wave/seed/validation and save-isolation/UI regressions. |
| 2026-11-11 | Added Median local-minimum L=max(0,(median-lambda1)/(median-min+tau_med*R²)) and bilateral CED×L as independent lab criteria, with Median tau=0.005 separate from unchanged Min-Max tau=0.01. Add Median L heatmap, CED×Median MAX NMS/gate, Auto Contrast and worker/cache propagation. Preserve existing scores/NMS/ROI/indices/Q10/ratio/matching; test exact/clamped/even/flat/invalid/scaled/outlier behavior, independent taus, NMS and GUI controls without accuracy claims. |
| 2026-11-11 | Exposed cached Local lambda1 minimum L(p) as a standalone heatmap-only view with fixed [0,1] and optional Auto Contrast. Disable NMS controls and bypass unrelated score gates in this view; preserve all score/NMS/matching/worker/cache behavior. Add cached-field/color/no-recompute/no-gating display regressions. |
| 2026-11-11 | Added display-only Auto Contrast for the standalone lab CED × local lambda1 minimum heatmap. Default OFF preserves [0,1]; ON uses existing P2--P98 colors with invalid/constant handling and a scale-aware legend. Preserve all scores, NMS/gates/matches and cache/worker state; add cached-color, unchanged-NMS/state and invalid-field regressions. |
| 2026-11-11 | Added standalone lab CED × contrast-regularized Normal-PCA lambda1 local-minimum field with separate positive-response MAX NMS, fixed [0,1] heatmap and per-field gate. Use PCA radius R for finite-neighbor min/max and tau_c*R², expose contrast_tau=0.01 separately from Fusion saturation, and include it in cache/worker inputs. Preserve original five fields/NMS, ROI/raw IDs, matching and Q10/MaxGap; test exact formula, flat/max/invalid behavior, scale invariance, cancellation, NMS and GUI integration. |
| 2026-11-11 | Added standalone lab saved-ROI input selection with full raw-ID preservation, ROI-only display/scoring/analysis/picking and excluded-label preservation. Added five independent inclusive post-NMS score gates and rematching of surviving extrema, with blank/OFF preserving cached numerical behavior. Validate ROI/thresholds before state changes; add boundary, indexing, save-safety, MAX/MIN equality, cache and rematching regressions. |
| 2026-11-11 | Fixed standalone lab mouse tab switching by registering optional inspection widgets after permanent controls, preserving the tab panel's Open3D/ImGui window ID as views appear/disappear. Also avoid redundant native tab-index assignment. Preserve programmatic Corner navigation, layout, active worker/cache state and all numerical behavior; add child-order/native-callback regressions and verify native tab changes during computation and with cached results. |
| 2026-11-11 | Added detector `--source demo` using a shared extraction of the unchanged lab three-plane generator. Preserve exact seed-7 raw XYZ/RGB/known-corner ID, current detector settings, scene/Q10 voxel spaces and all detector/Lite algorithms. Add exact geometry regression, input/config/dispatch checks and unsupported-input validation; no GUI import or file writes in generation. |
| 2026-11-11 | Enlarged standalone lab matched-pair midpoint markers from 9 to 28 pixels and links from 3 to 6 pixels, with explicit legend text. Preserve match selection, endpoints, midpoint coordinates, distances and counts; verify marker prominence and zero-distance pairs without mutating cached results. |
| 2026-11-11 | Reorganized the standalone lab into Heatmap/NMS, Corner, PCA and Input tabs. Associate each of five cached scalar heatmaps with its own MAX/MIN NMS; retain optional paired-method comparison and add independent display toggles. Preserve all score/extrema/Q10/RANSAC/MaxGap calculations, defaults, caches and saved indices; add cached association, layout, visibility and raw-ID interaction tests. Use compatible Viridis registry lookup without changing colors or scaling. |
| 2026-11-11 | Added `pca_ced_cutoff` / `--pca-ced-cutoff` to shared dense PCA and comparison callers. Default zero preserves positive-original-CED support; positive values retain only CED-valid weights at or above the cutoff, never thresholding query CED or substituting bilateral CED. Keep geometric counts/coordinates/IDs and existing support gates; record cutoff and add inclusive-boundary, weighted-covariance, invalid-support, propagation and default-output regressions. |
| 2026-11-11 | Added independent `q10_refine_enabled` / `--[no-]q10-refine` under the existing validator master. OFF skips relocation, snapping, Q10/CC and min-Q10 gating, evaluates unchanged RANSAC/MaxGap at exact NMS XYZ and caches by coordinate. Preserve refinement-ON numerics and dense/NMS output; add explicit skipped trace/metadata/UI state and `maxgap_nms` seed provenance, with numerical, gate/cache/config and read-only display regressions. |
| 2026-11-11 | Replaced `bilateral_enabled` with `bilateral_mode={ced,fusion,off}` in settings/config/CLI and shared scoring/comparison callers. Preserve ced/off numerics; fusion filters the completed original-CED × PCA product using fusion-valid support and the existing radius/sigmas. Add pre-filter Fusion diagnostics and mode-aware managed panels; retain original CED/PCA/validity/IDs and verify weighting, invalid-support exclusion, propagation and final-score consumption. |
| 2026-11-11 | Changed Q10 diagnostics from selected-summary coloring to all-plane membership coloring: P1 orange, P2 green, P3 purple, overlap white and all-member exclusions dimmed. Matched arc/projection/rose colors and corrected stale Tk/native legends. Preserve one voxel per analysis ID, cached numerical evidence, detector/Q10/MaxGap decisions and all coordinates; added headless color/overlap/mask/no-mutation checks. |
| 2026-11-11 | Replaced final per-plane antipodal axis scanning/N+/N-/B with the current standalone lab MaxGap logic: dominant circular group, optional sparse end-bin trimming, retained-support reliability and inclusive cleaned-span veto. Removed obsolete cap/axis/side/B settings and CLI; added gap/tail controls and cached masks/diagnostics. Kept detector RANSAC tuning, Q10 refinement/early gate, dense scores/NMS and index contracts; updated managed plots/native views and numerical/reference/config/cache tests. |
| 2026-11-11 | Added settings-only `bilateral_enabled=True` switch to shared dense scoring and comparison callers. OFF skips the filter and uses original CED in Fusion, preserving PCA, validity, coordinates and scene IDs; default ON retains existing numerical behavior. Added switch/result/bypass diagnostics and numerical/configuration propagation regressions. |
| 2026-11-10 | Added standalone lab --scene/--dataset selection and points.npz loading, full raw-ID-preserving GT import with separate lab sidecar precedence, and geometry-only Open3D startup. Kept live-camera/demo/legacy-cloud behavior and every numerical function unchanged; validated input before replacing scenes and documented read-only loading. |
| 2026-11-10 | Moved the cached converged Q10 gate before final unit-sphere evidence: early reject below threshold, skip undefined Q10, retain equality and passing-point numerics, record evaluated/skipped semantics and actual final-evaluation counts. Added read-only dataset detector input for single scenes or numeric-order roots, saved ROI/full-scene selection, aligned RGB and the existing Lite views with shared settings/algorithms. |
| 2026-11-10 | Improved cached Lite diagnostics using the standalone spherical lab conventions: matching plane circles/projected dots/rose plots, exact per-plane score cards, signed axes/cap shading, overlap counts and explicit unknown states. Retained Q10 path/history, added equal mm display scales and screen-fitting layout, and preserved every numerical result and index without recomputation. |
| 2026-11-10 | Removed the remaining Cone/Tube/projection-continuity implementations, axis-grid module, mode selector and related settings/CLI/result parameters and display branches. Retained only spherical RANSAC final support, with unchanged Q10 relocation, angular equations, masks, decisions and scene/analysis IDs; replaced obsolete mode tests with removal guards. |
| 2026-11-10 | Replaced default Lite bidirectionality with reference unit-sphere origin-plane RANSAC and per-plane angular antipodal tests; centralized sphere controls, set B threshold to .08, preserved Q10/NMS/voxel/index contracts and explicit older methods, and added cached model/axis diagnostics and numerical parity tests. |
| 2026-11-09 | Added new-validator per-plane 2-D centroid-direction cosine Q10 and independent common min_q10=.10. Require all fitted planes to pass (1/1, 2/2, 3/3); reject known failures, never accept undefined/no-plane cases. Cache vectors/cosines/quantiles for plot/log diagnostics; retain prior RANSAC/support/UVs and all old validator behavior, with no Hull/Edge/Corner computation. |
| 2026-11-10 | Made Sommer / Ahmed / PCL Harris3D the default baseline set and mandatory in visual comparisons, appending omitted methods while preserving requested order. Saved views show missing records as NOT_SAVED without recomputation; preserved backend parameters, numeric outputs, scene IDs, units and manager-owned visualization. |
| 2026-11-10 | Added explicit scene2 ROI baseline preset after in-sample parameter selection: unchanged dataset Sommer yields one GT-near corner; Ahmed Python yields four. Exposed PCA K, patch span and strict angle bounds, preserving author/dataset defaults, original equations/order, all L passes, raw-ID/unit contracts and GT-free runtime. Recorded distances, sensitivity and lack of held-out validation in BASELINE_TUNING.md. |
| 2026-11-09 | Added dataset/author baseline presets after an ROI-only 5-scene selection and separate 5-scene confirmation. Exposed Ahmed K/space/threshold overrides and optional Sommer observed-point voxel sampling with raw-ID mapping. Changed dataset CLI defaults while preserving author/library defaults, original algorithms, equations, unit/index contracts and GT-free run-time computation; recorded tuning outcomes and non-generalization in BASELINE_TUNING.md. |
| 2026-11-09 | Added independent new-validator mode and new_validator_* plane/projection settings. Reused canonical raw crop, guided RANSAC, support CC and mm UV projection, then stopped before rank/Hull/Edge/Corner computation. Preserved original validator numerics/stages, scene/native indexing and Q10 ACCEPT-only seed policy; added cached 2-D projection views and equivalence/configuration/no-later-stage regressions. |
| 2026-11-09 | Added explicit Ahmed MATLAB-to-Python baseline and made it the default `ahmed` backend, retaining original MATLAB execution by option. Preserved author stage constants/order, strict gates, PCA convention and all ten stochastic L passes; recorded port identity/parity limits and separate analysis-index namespaces. Existing detector, Harris and Sommer computations remain unchanged. |
| 2026-11-09 | Added read-only dataset baseline comparison through the visualization manager: GT plus ordered method panels, shared saved input and synchronized cameras, explicit zero/failed/prepared statuses, saved-result reopening and a separate GUI process per scene. Detector/validator and author-baseline numerical behavior is unchanged. |
| 2026-11-09 | Restricted downstream Q10 validator/total seeds to accepted_indices (ACCEPT only). Excluded INDETERMINATE even when Q10 keeps it for diagnostics; preserved Q10 scores/statuses/refinement, original scene IDs, refined centers and Q10-OFF NMS behavior. Added mixed-status and all-INDETERMINATE/no-fallback regressions. |
| 2026-11-09 | Connected detector Q10 ON/OFF to interactive validator and total: original NMS seeds when OFF, kept ACCEPT/INDETERMINATE scene IDs and stored refined centers when ON. Added shared seed bundles and explicit canonical query-center arguments, propagated through raw/organized crop caches and GUI/RGB picking without moving scene points or changing support-ID spaces. Preserved detector/Q10/validator equations; enabled shared Q10 CLI/config in all three modes and added numeric crop/index/cache/dispatch/view regressions. |
| 2026-11-09 | Removed Scale/PCA Lite defaults/config/CLI/runtime, its exclusive selected-radius PCA class and debug views/tests. Q10 remains the optional detector Lite path; retained its result alias, numerical calculations, cached inspectors, refined-position status markers and final green RGB overlay. Dense scoring/NMS and other mode pipelines are unchanged. |
| 2026-11-09 | Optimized detector computation without changing scores or decisions: omit unused organized mapping; batch CED and share bounded bilateral/PCA neighborhoods with scalar sphere-boundary fallback; defer Q10 quantiles below the existing support gate without poisoning caches; prune Projection sorts using strict support-length bounds and preserve all ties. Added numeric/ordering/cache/dispatch regressions; stage timing now assigns shared lookup preparation to bilateral. |
| 2026-11-09 | Removed separate multiscale heatmap mode and Weighted Local Cone backend/settings/exclusive code/views/tests; moved shared radius-graph CC unchanged. Added unit_mean (unchanged default), coordinatewise unit_median and original-XYZ centroid Q10 axes via settings/CLI, propagated through cached refinement/final quality and physical 3-D cyan arrows. Dense scoring/NMS, CC rules, antipodal objectives and thresholds remain unchanged; nondefault Q10 axes may change relocation/acceptance. |
| 2026-11-09 | Corrected final quality rejection to use the converged cached Q10 (10th-percentile cosine), matching the refinement objective. Removed separate Q1 computation, renamed its lower-bound setting/CLI to q10_min_q10 / --q10-min-q10 while preserving .01, and changed result fields/rejection/debug labels to Q10. Kept B threshold .65, projection continuity, CC switches, axes and NMS unchanged. Added negative-1%-outlier/high-Q10 acceptance and final-Q10 boundary/cause regressions; diagnoses reuse stored values without recomputation. |
| 2026-11-09 | Renamed the detector's B threshold setting to q10_antipodal_threshold, the Q10 config/result/debug field to antipodal_threshold and the canonical CLI to --q10-antipodal-threshold. Kept --q10-threshold as a same-destination CLI alias and preserved the tuned value and all numerical predicates, including the independent Q1 lower bound. Updated callers, debug captions, tests and method references. |
| 2026-11-09 | Added optional projection_continuity antipodal support on the existing unique-axis grid: finite Tube projections, query-origin gap prefixes, full-R positive 3-D median NN with default gap factor 2.5, and max min(L+,L−)/R. Expand spatial lookup to hypot(R,tau), retain analysis IDs through diagnostics, and expose settings/CLI plus lab mode/gap GUI controls. Preserve original Cone/Tube counts, Q10/Q1, NMS and final gate composition; the old final 3-D CC applies to count modes while new mode uses only its specified 1-D continuity. Added exact numerical, full-signed reference, end-coverage, cache, configuration and headless GUI/debug regressions. |
| 2026-11-09 | Added independent default-ON switches for Q10 spherical CC and final selected Cone/Tube CC, including settings/CLI and cached debug state. Disabled stages skip NN/graph work; final CC OFF preserves pre-CC support exactly. Added final Q1=lower-1% cosine on the active Q10 support/mean, with user-selected lower bound .15, without changing Q10 relocation. Final rejection is Q1 failure OR opposite support; store all causes and show gate results/cause labels in debug. Added numerical OFF/ON-combination, Q1 boundary/cache and reason-display regressions. Existing tuned mode/radii/B thresholds remain unchanged; decisions can change with Q1 or CC disabled. |
| 2026-11-07 | Added spherical query CC before every uncached Q10 evaluation on the analysis voxels, using full-R median NN times the existing CC factor and the same query-anchor/graph rules as final Cone/Tube CC. Q10 mean/quantile and the 80% support guard now use the connected component. Cache component IDs/counts/NN/epsilon with responses; record step supports and display CC-only Q10 directions/histograms with connected/full-R counts. Final axis sampling, full-R denominator N, selected-region CC and decision thresholds are unchanged. Localization/decisions and cache-miss cost can change; old weighted Cone and standalone diagnostics are unchanged. |
| 2026-11-07 | Moved Q10 Lite median-NN spacing and selected-axis query CC from raw observations to the post-voxel analysis cloud for both Cone and Tube. Removed nearest-raw membership transfer; ROI/geometry/component IDs now explicitly index analysis voxels. Keep physical mm distances, full-R median, fixed pre-CC axis, one graph per final point, fixed denominator N, Q10 refinement and existing configuration values. Display voxel NN/epsilon and add raw-density/bridge-invariance regressions; changed spacing/connectivity can change final decisions. The separate weighted Cone raw CC is unchanged. |
| 2026-11-07 | Moved final detector CC after antipodal axis selection. Median NN still uses all raw Q10-radius neighbors; run one query-anchored graph inside the selected double Cone/Tube intersected with R, with no halo detours. Filter side masks by transferred raw connectivity while retaining all nonzero R voxels in N and keeping the pre-CC axis/ties. Store pre/post evidence, show side-count changes and update numerical/order/cache regressions. Q10, candidate grids, thresholds and standalone raw antipodal diagnostics remain unchanged; post-CC global optimality is not guaranteed. |
| 2026-11-07 | Made local Q10 diagnostics display exact analysis voxels, with one native point cloud keyed by analysis ID and shared final-support plot colors from immutable support masks. Removed dense raw/duplicate gray overlays, distinguished unscored context from scored non-support and retained the independent RGB final view. Restored the implemented Q10 settings block/4 mm Tube radius while preserving detector/user scales; classifier equations, Q10/CC and scene/raw/analysis index spaces remain unchanged. |
| 2026-11-07 | Shared Cone/Tube antipodal grids between the detector and standalone lab; preserved 42/64 and added nested 162/642/2562 icospheres, default 642. Evaluate only earliest +/- representatives in bounded batches, preserving min-side/total/first-signed-index ties, masks and score normalization. Added detector Cone/Tube and tube-radius controls, expanded standalone GUI choices and Tube cylinder diagnostics. Kept Q10, voxelization, final-only CC, existing rejection thresholds and RGB markers unchanged; denser default sampling can change final decisions. |
| 2026-11-07 | Display final accepted corners in green over the registered RealSense RGB point cloud, using Q10 refined XYZ and cached ACCEPT decisions. Reuse the final Lite view, including under --only; omit duplicate RGB/NMS auxiliary output and retain ACCEPT-only refresh/picking. Preserve scoring, fixed-NMS refinement, CC/antipodal decisions, source RGB and index spaces. |
| 2026-11-07 | Bounded Q10 refinement to one search in the original scene NMS ball, including voxel-snap displacement. Removed moving-window iteration; retained frame/radius Q10 caching, initial-relative 80% support, final-only CC/antipodal equations and original scene IDs/scores. Empty fixed balls keep the original NMS position as INDETERMINATE without CC. Updated traces, debug/log/help terminology and regression checks for outward drift, unsnapped centers and missing local support. |
| 2026-11-07 | Added default detector Q10 relocation with NMS-radius steps and frame/cloud/radius response caching; final-origin-only raw CC followed by the reference antipodal cap decision, with shared final evidence for duplicate convergence points. Kept independent OFF-by-default weighted Cone and Scale/PCA switches and immutable upstream scores/scene IDs. Added stage/status markers, cached step/sphere/histogram/CC/opposite-support inspector and isolated native path view. The .08/3/25-degree antipodal defaults are experimental; local Q10 convergence does not certify corner accuracy. |
| 2026-11-06 | Made Density ON + Local ON use the cached Local weighted optimum for public score/direction/cosines/angle/coverage, threshold decisions, accepted/kept scene IDs and blue NMS markers. Retained CED, Plane comparison records, unchanged upstream Fusion/NMS/CC/query positions and all three other option combinations. Updated debug decision labels and tests, including Local-pass/Plane-fail, inclusive threshold, rejection, diagnostic mode and original-coordinate blue marker checks. Preserved the user's ON/ON and threshold settings; pinned legacy numerical fixtures to explicit Density OFF. All 107 related tests passed; full suite 289 tests retained 3 previous failures/15 errors. No actual GUI/sensor run. |
| 2026-11-06 | Added default-OFF optional direction-density weighted inverse-CDF quantiles, bounded angular-neighbor blocks and once-per-query cached weights. Preserved OFF NumPy behavior and unchanged CED/PCA/Fusion/NMS/CC/query contracts; ON fixes the origin and skips legacy Back retry. Reused axis searches, Plane decision ownership and cached views; added same-axis raw/weighted Q and separate raw/weighted coverage. Controlled comparisons expose sparse-face exclusion and an edge false positive. All 106 related checks passed; full suite 288 tests retained the same existing 3 failures/15 errors. No new device/GUI run; see the comparison report. |
| 2026-11-06 | Limited Lite Cone and legacy Lite result markers and picking to final ACCEPT NMS records, with ACCEPT/total titles and empty-marker refresh handling. Preserved scene heatmaps, original NMS diagnostic panels, numerical results, index spaces and fail-open forwarding. |
| 2026-11-06 | Removed Global/icosphere settings, computation and mode switching; Local-only is enabled by default, with unchanged Local arithmetic and Plane-baseline acceptance. Added a single-window spawned Open3D inspector to isolate native GPU/GLFW lifetime from detector/Tk, bounded worker cleanup, camera clipboard-paste interception, numeric roll and invalid-camera sync guards. Passed 63 targeted tests and four actual Tk/parent/child open-close cycles across system/laser_handeye Python. Native sensor segfault cause remains unconfirmed; no shape simulation, full suite or live sensor test was run. |
| 2026-11-06 | Added fixed-NMS minimum enclosing Cone comparisons: raw-quantile Global icosphere search and independent Local cap, centroid/CED initialization and an identical-support Plane baseline. Preserve default legacy behavior; comparison modes bypass Back movement and retain baseline acceptance. Add batched/cached geometry, raw-negative ranking, angle/coverage/rotation/reduction metrics, comparison table and Open3D 1/2/3 switching with fixed camera/apex. Existing user threshold 0.25 and disabled Back default are preserved. Nine new plus 46 existing targeted tests passed (55 total), plus imports/CLI/syntax and actual Open3D three-mode render/switch checks. Shape simulations, full suite and sensor tests excluded. |
| 2026-11-06 | Added click-owned Open3D visualization of the stored raw quantile angular boundary, actual coverage and inside/outside points at the final evaluation origin. Handle negative quantiles and finite-sample/tie coverage without score clipping; align Matplotlib cone ring with raw Q rather than decision threshold. Added nonblocking window polling/cleanup and oblique startup camera. Passed 46 numerical/UI/architecture checks plus imports/CLI and actual hidden Open3D positive/negative render inspection. Numerical detector/validator results are unchanged; no scene simulations or sensor run. |
| 2026-11-06 | Added one conditional Plane retry at the deepest strict-back raw point. Preserve NMS IDs/positions, first-pass results, original CC/core/CED and normal cap/spacing; no extra NN tree or repeated connectivity. Adopt feasible retry Cone results, otherwise keep the first pass. Add a default-ON retry toggle inside default-OFF Plane mode, explicit evaluation origin/shift/retry records and existing-window geometry/score comparison. Passed 38 numerical/configuration/headless/architecture checks plus syntax/import/CLI checks; no shape simulations, full suite or device GUI run. Synced descriptions to the existing user-set 0.20 threshold and 45-degree comparison-B cap without changing them. |
| 2026-11-06 | Added default-OFF Plane-based CED direction refinement: minimize Gaussian contact subject to a back-ratio bound over a bounded spherical cap. Planes always pass through the unchanged NMS query; remove the prior minimum-projection display translation. Preserve A and comparison-only B, with C precedence, unchanged CC/CED/Fusion/PCA/NMS and final Cone formula. Add unique-support NN thickness/floor, feasibility fallback, plane/cone diagnostics and existing-window plane-side visualization. Passed 29 arithmetic/configuration/UI/architecture checks plus syntax/import/CLI checks. Shape simulations, full-suite and hardware/GUI execution excluded. |
| 2026-11-06 | Moved refinement debug panel 2's green arrow/cone to the selected core CC minimum-projection height and added its perpendicular display plane. Keep initial arrow/query at NMS, and keep all numerical results, cosine colors and score origins unchanged. Nine Cone and four architecture tests passed, including headless anchor/CC exclusion/no-mutation and OFF-display checks. Synchronized descriptions with the user's existing 40 mm radius/0.15 threshold without changing settings. No scene simulation, full-suite or device GUI run. |
| 2026-11-06 | Removed neighboring-CED direction averaging and restored the candidate's raw CED d0. Added optional finite spherical-cap direction refinement (OFF, 20/5/15 degrees), vectorized unchanged Cone quantile maximization with deterministic ties and no rotation penalty. Preserved CC/support, raw CED, Fusion/PCA/NMS and threshold/fail-open policy. Added before/after directions/scores/angle/count and extended existing cached debug. Seven refinement, eight Lite wiring/UI and four architecture tests passed, plus syntax/import/CLI checks. Scene simulations, full-suite and sensor/GUI runs excluded. |
| 2026-11-06 | Changed the Cone axis to the normalized, unweighted arithmetic mean of valid neighboring raw CED unit directions in the seed component. Each local CED uses its complete same-component radius-R support from the existing 2R halo. Added direction count/coherence and debug provenance; insufficient/cancelled directions remain Indeterminate. Preserved query coordinates, CC policy, Fusion/NMS, Cone quantile/threshold and old Lite backend. Three direction arithmetic checks and eight wiring/UI contract tests passed; scene simulations, full suite and sensor/GUI runs were skipped. |
| 2026-11-06 | Made `lite_validator_enabled` the detector Lite master switch with Cone as its default backend, experimental threshold 0.5 and debug enabled. GUI/headless runs evaluate every NMS candidate before display; clicking only inspects cached evidence. Preserve the old Scale/PCA backend through `--no-cone-extremality`, explicit library deferral, all scoring formulas and upstream NMS. Scope implicit defaults to detector mode. Eight configuration/delegation/UI contract tests and syntax/import/CLI checks passed; no numerical simulation or sensor/GUI run. |
| 2026-11-06 | Replaced the unfinished directional-length experiment with opt-in Cone Extremality after unchanged detector NMS. Added raw halo-aware seed-component selection, signed-cosine lower-quantile scoring, None-threshold diagnostics and fail-open Indeterminate states, with click caching and CC/cone/histogram debug views. Preserved old Lite and upstream numeric stages. Simulation/test execution and sensor/GUI validation were skipped at the user's request; syntax/import/CLI only. |
| 2026-11-06 | Added three representative raw single-radius Heatmap references at configured minimum/geometric midpoint/maximum (30/51.9615/90 mm for 30–90 mm), reusing raw CED × saturated weighted-PCA without bilateral, Local Support or Window scoring. Store independent scores/NMS; retain four multiscale results and primary selection. Show all seven with red NMS markers, shared max-Q99 colors/cameras and a three-column layout. |
| 2026-11-06 | Show each Local Support comparison's stored NMS candidates as red sphere markers in its own Heatmap, with dense-detector sizing, per-mode counts and empty-set handling. Keep shared max-Q99 colors/cameras and all score/support/window/NMS calculations unchanged. Add headless marker position/provenance/color and no-mutation coverage across four distinct candidate sets. |
| 2026-11-06 | Audited four-mode Heatmap scoring on recorded Kinect redkitchen/seq-03 frame 740 (46,080 voxel points): arrays are independent, Local Support has real neighbors, scores and zero-gate NMS results differ. Replaced independent per-window Q99 normalization with shared max-Q99 colors so equal scores map equally and magnitude differences are preserved; added raw-score/delta logs and a proportional-score rendering regression. Preserved all numerical scores, windows and NMS; no live RealSense/GUI test was performed. |
| 2026-11-06 | Replaced fixed 10 mm Local Support with a scale-relative factor, default 1/3: ced/fusion use each CED radius, multiscale uses each query's existing selected representative radius. Added scalar/per-query helper radii and stored resolved radius arrays; replaced the fixed-mm CLI/config and updated labels/tests. Preserve baseline scores, raw CED/PCA/directions, RMS/window formulas, final-stage Window selection and NMS configuration. |
| 2026-11-06 | Added independent spatial Local Support G(X)=sqrt(X*valid-neighbor mean), 10 mm support, and none/ced/fusion/multiscale primary-result selection. Reuse unchanged raw CED/PCA, directions, saturation, RMS windows and NMS; honor supplied scene component labels. Store four comparison results and NMS coordinates/scores, display four synchronized pre-NMS Heatmaps, and remove NMS markers as requested. Preserve user-tuned voxel/threshold defaults and all other detector modes. |
| 2026-11-06 | Show stored Local Maximum NMS candidates as red sphere markers over the multiscale pre-NMS Heatmap, using existing dense-detector marker sizing. Preserve Raw CED/PCA/Fusion, stability scores, thresholds, NMS results, Q99 colors and scene coordinates; update labels and headless renderer tests, including empty NMS. |
| 2026-11-06 | Added independent `--mode heatmap`: raw CED × existing saturated weighted-PCA ratio without bilateral calls, 1.2 log scales and best contiguous five-scale mean-magnitude minus RMS vector deviation. Added chosen-window radius metadata, independent final threshold/18 mm Local Maximum NMS and one pre-NMS-only Heatmap view without markers. Preserved existing mode numerics and scene IDs; reused shared NMS and retained the old HeatmapConfig alias. |
| 2026-11-05 | Simplified Lite to a CED prefix anchored at the detector radius: running-peak loss and all-pairs direction diameter stop at the first failure/invalid sample, without restart or a three-sample minimum. CED passes iff last valid R* >= minimum scale. Keep one raw weighted-PCA gate after CED at R*, current threshold 0.35, separate stage outcomes and unchanged detector/NMS/final-validator calculations. Store only the evaluated prefix and simplify cached debug to four panels with a bounded slider. |
| 2026-11-05 | Replaced the CED prefix with overlapping joint stable-interval search: interval running peaks, maximum pairwise direction diameter, at least three valid samples and exact minimum containment. Select the farthest-ending interval and apply raw CED-weighted PCA >=0.10 once at its R*, preserving original bilateral/PCA arithmetic with lazy full-scene dependency caches. Store CED-only versus final outcomes, prior-response-loss diagnostics and PCA rejects with retained scale; add six-panel cached debug. Detector/Fusion/NMS/final-validator behavior is preserved and no Fusion scale sweep is added. |
| 2026-11-05 | Simplified Lite to continuous CED-scale estimation: one-sided score drop <=0.15 and directed angle <=20 degrees, first failed transition stops without grace, exact minimum scale inserted, and only one/two initially undersupported samples may be skipped. Renamed minimum-scale/drop settings, added maximum-reach flag and accepted radius proposals, preserved normal/debug decisions with normal early exit, and retained fixed final-validator ROI interface with an explicit applied=False flag. Updated ACCEPT/REJECT diagnostics, tests and methods; original detector/NMS/CED equations remain unchanged. |
| 2026-11-05 | Reverted Fusion Score Scale Persistence and its subsequent no-bilateral variant to the saved pre-Fusion CED magnitude/direction implementation. Restored 0.20/20-degree controls, eight log samples, two-consecutive-failure prefix/rollback, maximum-radius warning policy and four-panel CED debug. Removed sparse multi-scale PCA/Fusion helpers and ratio settings; retained earlier two-window click/camera fixes and current user detector/enable settings. |
| 2026-11-05 | Removed bilateral recomputation and its spatial dependencies from Lite only, per user request. Lite now uses original CED times saturated PCA with its own base denominator; original detector bilateral/Fusion/NMS remain unchanged. Updated base comparison to unfiltered detector components, removed bilateral debug response, relabeled Lite plots and added a no-bilateral-call regression. |
| 2026-11-05 | Replaced CED magnitude/direction/two-failure persistence with raw Fusion/base-Fusion ratio and first-failure prefix. Insert the exact required radius and reject out-of-range settings. Share existing dense kernels and cache full-scene dependency CED while evaluating PCA/Fusion only at NMS queries; preserve dense/NMS outputs and final-validator behavior. Updated six-panel cached debug decomposition and regression/real-data equivalence and timing checks; synthetic-cloud experiments remain excluded. |
| 2026-11-05 | Fixed Lite debug interaction: enabled candidate clicks in both original NMS and extra Lite windows, restored camera drag handling replaced by Open3D custom callbacks, suppressed return-to-start drag clicks, queued Tk construction outside GLFW dispatch and raised the inspector. Added terminal click diagnostics and UI regressions; detector/NMS/Lite numerical results remain unchanged. |
| 2026-11-05 | Added optional detector-only CED scale-persistence Lite Validator after unchanged NMS, sharing original CED arithmetic, caching radius responses and filtering explicit final-validator seeds. Added KEEP/REJECT display and cached NMS-click debug inspector; preserve original score/NMS records and disabled behavior. Documented two-failure prefix semantics, invalid responses, unreachable 120 mm at 30x3 mm defaults, and no automatic final-ROI override. Excluded synthetic experiments as requested. |
| 2026-11-05 | Converted detector NMS to `nms_radius_factor` / `--nms-radius-factor`, removing optional absolute-mm defaults/CLI and the `None` fallback. Chose factor 1.0 to preserve the user's current CED 30 mm / NMS 30 mm settings. Keep metric numerical interfaces and the unchanged Local Maximum algorithm. Updated all shared detector consumers and factor/invalid-value/removed-CLI/scale-invariance tests; ball's separate hypothesis NMS remains unchanged. |
| 2026-11-05 | Unified bilateral spatial settings as CED multipliers: `bilateral_radius_factor=1.0` and `bilateral_sigma_spatial_factor=0.5`, replacing optional spatial-mm defaults/CLI and their `None` resolution branches. Keep `bilateral_sigma_ced=0.05` dimensionless, existing resolved-mm library interfaces and unchanged filter equations/default numerical outputs. Extended multiplier/invalid-value/removed-option regressions and numerical scale-invariance checks. |
| 2026-11-05 | Made detector PCA follow the absolute CED radius through `pca_radius_factor=1.5` and restored `--pca-radius-factor` in place of the independent PCA-mm option. Bilateral defaults continue to follow CED at radius 1x / spatial sigma 0.5x, retaining explicit mm overrides. Kept CED independent of voxel, resolved metric library interfaces, default 80/120/80/40 mm behaviour and NMS settings. Updated CLI/config/scale-regression tests and docs; scaling synthetic points and CED/NMS radii preserves expected scores and candidate IDs. |
| 2026-11-05 | Decoupled detector support radii from voxel size: absolute `centroid_radius_mm=80` and independent `pca_radius_mm=120`, exposed as `--centroid-radius-mm` / `--pca-radius-mm`. Removed detector PCA-factor conversion/config/logging; kept bilateral defaults tied to CED radius, explicit overrides and NMS behaviour. Fine retains its original voxel-multiplied CED setting and fine-only flag. Updated shared Coarse consumers, CLI/fixtures/docs and numerical score/NMS regressions across voxel settings. No scoring, thresholding, sampling or validator geometry equations changed; no multi-scale loop was added. |
| 2026-11-05 | Replaced guided Corner validation with independent endpoint candidate generation and complete-diameter shared-Edge/same-Endpoint integration. The standalone function defaults to 20 degrees / 5 mm; preserved the user's subsequent live CLI edits to 45 degrees / 15 mm. Removed vertex distances, query/core/Node gates, LS/anchor refinement and selected-Corner ranking. Added existing Edge fitting-tolerance metadata for 3-D skew residuals; Edge detection/masks remain unchanged. Return every unverified Corner with endpoint provenance and representative residual/gap; adapt validator, total, terminal and manager-owned views. Added numerical boundary/skew/merge/anti-chain/multi-output/validation tests and headless display checks. |
| 2026-11-05 | Removed the plane-support `nearby` component count and its multi-component UNKNOWN gate. Support validity no longer depends on `corner_gate_radius_mm`; retain the nearest original-inlier-anchored component, original tie breaks and single-component Hull. Preserve missing-anchor UNKNOWN, residual/radius filtering, native IDs, plane models and Hull degeneracy checks. Added numerical tests for two nearby disconnected anchored components, Corner-gate independence, anchor preference and missing-anchor rejection. With the reference 20 mm connectivity / 10 mm Corner gate, the removed multi-component condition was unreachable in exact geometry; alternative settings may now retain previously UNKNOWN patches. |
| 2026-11-05 | Removed same-type finite-edge deduplication and `validator_edge_duplicate_angle_deg` from defaults, resolved config, validation and CLI. Retain every valid finite candidate in the existing quality/length/source order, with consecutive Edge IDs and singleton `source_edge_ids`. Preserve TLS RMS fitting, Node provenance, core clipping, minimum length and intersection–boundary masking. Distinct opposite Hull chains now survive even when they share endpoint Node IDs; more retained Edges can change downstream IDs/Corner results. Updated angle-independence tests and added numerical thin-Hull retention and unchanged intersection-masking regressions. |
| 2026-11-05 | Changed adjacent boundary TLS merging from maximum orthogonal residual to RMS over equally weighted unique group vertices, retaining the 3.5 mm threshold. Renamed the default/resolved setting and CLI to `validator_boundary_merge_rms_error_mm` / `boundary_merge_rms_error_mm` / `--validator-boundary-merge-rms-error-mm`. Added `tls_rms_error_mm` to Lines/Boundary Edges, retained maximum error as diagnostics, and changed live/exported TLS labels to RMS. Preserved Node adjacency/provenance, independent direction gates, SVD fitting, ROI clipping, minimum length and upstream plane fitting. Added numerical RMS-versus-max, exact threshold, propagation and configuration regressions. |
| 2026-11-05 | Split the formerly shared guided geometry angle into independent boundary TLS merge, same-type finite-edge duplicate and Corner minimum angles, each defaulting to 30 degrees. Added settings, CLI propagation, finite `(0, 90]` validation and numerical independence regression tests. Preserved TLS formulas, the current 3.5 mm maximum residual, endpoint provenance, deduplication conditions and default numerical behavior in validator/total. |
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
