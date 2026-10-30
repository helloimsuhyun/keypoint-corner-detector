# Active detector and validator methods

This is the canonical ledger for the active numerical pipeline. All coordinates
and metric thresholds are in millimetres unless stated otherwise.

## 1. Stage order

Interactive fine queries and detector NMS seeds share one entry point:

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
extraction is unconditional for fine/detector configurations. There is no backend selector and no FPS/SAH
reference-plane clustering fallback.

`--mode detector --only` is an explicit stop after the unchanged dense
CED/bilateral/PCA/fusion-threshold/3-D NMS composition. It resolves
`DetectorConfig.only=True`, `validator=None` and omits all validator, plane,
edge and corner computation. Returned dense scores and NMS provenance are the
same as the normal detector before validation; they contain no
`validator_results` or post-validation candidate statuses. Rendering selects
only the existing NMS Corner Candidates panel, preserving its Fusion valid
mask, linear Q99 range, red scene-indexed markers and M marker toggle.
No geometry/CED/PCA/Fusion-only or auxiliary registered RGB windows open.
`--no-3d-view` suppresses the remaining window too. The option is valid only
with `--mode detector`; the default detector still validates every NMS seed.

### 1.1 Coarse-to-fine guided sequential validator

`--mode validator` uses the original voxel-scene CED → bilateral CED → weighted
PCA ratio → Fusion threshold → radius-local-maximum NMS to create Coarse queries.
The existing Coarse functions, order, parameters and outputs are unchanged;
the plan's schematic ordering/Greedy label does not replace the active Coarse
implementation. `prepare_validator_queries` retains the original detector result
in `validator_ced_scores`; its Fusion heatmap remains the left overview.

Only a selected Coarse query starts the following independent experiment:

```text
Coarse scene-indexed query
  -> Fine-voxel evaluation ball with complete CED/PCA/bilateral support halo
  -> existing dense CED/bilateral/PCA/Fusion functions at the Fine scale
  -> reused eligibility threshold -> deterministic Greedy 3-D NMS
  -> closest Fine candidate (or unchanged Coarse fallback)
  -> Fine-centred RANSAC-voxel ball
  -> first-point-guided, three-point hypotheses
  -> count/RMS selection -> hard inlier removal -> at most three planes
```

Defaults live in `settings.SettingsDefaults` and resolve through
`GuidedValidatorConfig`; CLI options use `--validator-*`. Fine/RANSAC voxel sizes are absolute mm;
all nine CLI/default length values are stage-specific voxel multipliers (`*_x`).
Numerical functions receive resolved mm lengths in `GuidedValidatorConfig`;
direct library mm overrides retain their explicit-mm contract. Obsolete anchored weighting/normal-angle
settings and their CLI options are removed, along with the anchored extractor.

| Setting (`SettingsDefaults` field) | Default | Length basis |
|---|---|---|
| `validator_fine_voxel_mm` | 5.0 mm | independent Fine voxel |
| `validator_fine_search_radius_x` | 9.0 | Fine voxel (45 mm) |
| `validator_fine_ced_radius_x` | 3.5 | Fine voxel (17.5 mm) |
| `validator_fine_min_neighbors` | 6 | minimum neighborhood support |
| `validator_fine_bilateral_radius_x` | 3.0 | Fine voxel (15 mm) |
| `validator_fine_bilateral_sigma_spatial_x` | 1.5 | Fine voxel (7.5 mm) |
| `validator_fine_bilateral_sigma_ced` | 0.05 | normalized CED range sigma |
| `validator_fine_pca_radius_x` | 4.5 | Fine voxel (22.5 mm) |
| `validator_fine_pca_ratio_saturation_tau` | 0.5 | dimensionless; None disables saturation |
| `validator_fine_fusion_threshold_mode` | peak-relative | peak-relative, global-mean or fixed |
| `validator_fine_fusion_score_threshold` | 0.10 | fixed cutoff; absolute floor in peak-relative mode |
| `validator_fine_fusion_peak_relative_ratio` | 0.5 | fraction of finite valid Fine evaluation maximum, range [0, 1] |
| `validator_fine_nms_method` | greedy | greedy or local_maximum |
| `validator_fine_nms_radius_x` | 3.0 | Fine voxel (15 mm, current workspace setting) |
| `validator_ransac_voxel_mm` | 10.0 mm | independent RANSAC voxel |
| `validator_ransac_radius_x` | 6.0 | RANSAC voxel (60 mm) |
| `validator_inlier_threshold_x` | 0.5 | RANSAC voxel (5 mm) |
| `validator_guided_sigma_x` | 1.0 | RANSAC voxel (10 mm) |
| `validator_max_planes` | 3 (valid range 0–3) | count, unchanged |
| `validator_ransac_iterations` | 256 per round | count, unchanged |
| `validator_min_inliers` | 20 | RANSAC-voxel sample count, unchanged |
| `validator_min_inlier_ratio` | 0.08 | dimensionless, unchanged |
| `validator_random_seed` | 42 | unchanged |

`prepare_validator_voxel_context` constructs Fine and RANSAC clouds from the full
input with the existing voxel downsampler, caching each distinct voxel size/tree
once per frame. Matching sizes share the same cloud/tree; the supplied Coarse
scene is reused when its voxel size matches. Input raw samples are used only to
prepare new clouds. Fine evaluation/neighborhoods use only Fine voxel centroids;
RANSAC sampling/evaluation/support counts use only RANSAC voxel centroids. The
selected Fine query need not coincide with a RANSAC voxel centroid.

**Fine neighborhoods and NMS.** Evaluation samples are Fine voxel centroids in the
inclusive search ball around $q_c$. All scoring/NMS values now come from the
independent `validator_fine_*` settings. CED, PCA, bilateral radius/spatial sigma
and NMS lengths each multiply the Fine voxel size. Bilateral range sigma is in
normalized CED units; PCA saturation tau and Fusion thresholds are dimensionless.
Fine minimum neighbors is a separately validated integer of at least three.
Coarse CED/PCA/bilateral/threshold changes do not change Fine scoring parameters.

The unchanged scorer still applies CED → simultaneous bilateral CED →
original-CED-weighted PCA → optional tanh ratio saturation → product Fusion.
At the current workspace's 5 mm Fine voxel, dedicated defaults give
17.5/22.5/15/7.5 mm CED/PCA/bilateral-radius/spatial-sigma values, range
sigma 0.05, saturation tau 0.5 and six neighbors. Changing CED
radius no longer implicitly changes Fine PCA or bilateral lengths; each is
explicitly configured. Setting Fine saturation tau to None uses the raw ratio.

`validator_fine.find_fine_queries` extracts Fine voxel samples out to
$R_{support}=R_{search}+\max(r_p,r_b)+r_c$. The last term includes the CED
neighborhood of every PCA/bilateral support point. This is a computational halo,
not a clipped evaluation ROI: every evaluation sample's dependencies are the
same Fine-voxel samples as a full Fine-cloud calculation. Halo-edge scores are
never candidates. All public Fine score arrays are sliced to the evaluation
ball, and `original_point_indices` explicitly contains Fine voxel IDs. The existing
threshold path supplies strict eligible $S_i>T$ masks. Fine independently chooses
fixed, global-mean or peak-relative (default) thresholding. With $V$ the finite
Fusion-valid evaluation samples, absolute threshold $T_0$ and dimensionless
ratio $\alpha\in[0,1]$, peak-relative uses
$$
T=\max\left(T_0,\;\alpha\max_{i\in V}S_i\right).
$$
The default is $\alpha=0.5$, $T_0=0.10$. The support halo and invalid/nonfinite
scores do not contribute to the peak. Empty $V$ retains $T_0$ with no eligible
points. All-zero support is rejected, and a ratio of one rejects even the peak
because eligibility remains strict. Fixed mode uses only $T_0$; global-mean
uses the mean over $V$ and ignores $T_0$. A single extreme peak can suppress
other real corners; neither a relative threshold nor its floor proves that a
corner exists. The ratio/floor are starting values, not calibrated accuracy
claims. `fine_nms_method="greedy"` ranks
eligible points by decreasing Fusion score, breaks ties by Fine voxel ID, and
suppresses within the inclusive configured radius of retained peaks.
`fine_nms_method="local_maximum"` reuses the existing deterministic local-maximum
candidate indices instead. Coarse scoring, configuration and its extractor are
unchanged. Fine results record the threshold mode/value, peak-relative ratio and
NMS method. Only Fine changes its default eligibility rule; Coarse keeps its
existing fixed/global-mean choices, and RANSAC is unchanged.

The default highlighted Fine query is the candidate closest to $q_c$ (ties: Fine voxel
ID). All Fine NMS candidates are retained. In the interactive 3-D workflow, a
Coarse click requests `stop_after_fine=True`: the entry returns the Fine result
with `planes_computed=False`, without RANSAC. A Fine NMS click runs RANSAC using
the stored Fine result. Later Fine selections reuse these same scores. Default
library calls and `--no-3d-view` retain the full automatic nearest-Fine pipeline. If no candidate exists, $q_f=q_c$ and the console
and result explicitly report `fine_fallback=True`.

**Guided sequential RANSAC.** For the inclusive RANSAC-voxel ball
$I=\{i:\|p_i-q_f\|\le R\}$, compute the fixed acceptance count
$N_{min}=\max(N_{configured},\lceil\rho |I|\rceil)$. It never shrinks as planes
are removed. In each round let $A$ be the remaining samples and set
$g_i=\exp[-\|p_i-q_f\|^2/(2\sigma_s^2)]$. The first sample is drawn with
$P(a=i)=g_i/\sum_{j\in A}g_j$, using log-weight subtraction for stable
normalization and uniform fallback when no valid normalization survives.
Probabilities are renormalized on remaining points every round. Draw the other
two points uniformly without replacement from $A\setminus\{a\}$.

With $v_b=p_b-p_a$, $v_c=p_c-p_a$, reject any pair separation at or below
$10^{-9}$ mm or cross-product norm at or below
$10^{-12}\max(1,\|v_b\|\|v_c\|)$. Otherwise set
$n=(v_b\times v_c)/\|v_b\times v_c\|$, $d=-n^Tp_a$. The Fine query defines
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
cfg, raw_points_mm=...)` remains the canonical composition entry. The raw input
is used only for context construction; cached calls pass `voxel_context`.
`ValidatorDetectorConfig` selects this experiment; existing `ValidatorConfig`
retains the unchanged fine/detector composition. `query_index` is the Coarse
scene ID. `fine_query_index`, Fine `candidate_fine_indices`,
`evaluation_fine_indices` and `support_fine_indices` address the Fine voxel cloud.
`roi_ransac_indices` and `inlier_ransac_indices` address the RANSAC voxel cloud;
plane masks address the stored RANSAC ROI and remain disjoint. A fallback has
`fine_query_index=None` and uses the original Coarse position.

Each native cloud has a nearest-Coarse scene adapter, computed once. It supplies
`fine_query_scene_index`, Fine `candidate_scene_indices`, `evaluation_scene_indices`,
`support_scene_indices`, and public `roi_point_indices`/`inlier_point_indices` in
the supplied Coarse scene space. Mapping may be many-to-one. Adapter IDs never
replace native positions, masks or support counts. Other mode indexing contracts
and the scene-indexed projection/edge/refiner pipeline are unchanged. Cached Fine
results are checked against the Coarse query and Fine voxel size.

Report Coarse/Fine positions, displacement, candidate count/fallback, both voxel sizes, Fine CED
radius, RANSAC radius, guided sigma, ROI count, plane normal/offset, inlier count,
inlier ratio to the **original ROI**, RMS and RANSAC total time. The absolute
query-to-plane distance $|n^Tq_f+d|$ is diagnostic only.

**Visualization.** One native Open3D window now contains three labeled panels:

1. Coarse Fusion heatmap plus orange Coarse NMS markers, uploaded once.
2. Fine Fusion heatmap of the stored Fine-voxel evaluation points plus yellow Fine NMS
   markers. The nearest/selected Fine query is a larger red sphere. Colors use
   the stored Fine Fusion values/valid mask, an independent valid-score linear
   Q99 range and the same shared heatmap color function as Coarse.
3. Plane fitting after a Fine NMS click: gray RANSAC-voxel ROI/outliers, orange Coarse
   sphere, large red Fine sphere, Fine-centred wire RANSAC ball, and RGB plane
   disks, inliers and normal arrows. Fine candidates stay in panel 2.

A Coarse click refreshes panel 2 and clears previous plane fits; panel 3 waits for
Fine selection. `planes_computed=False` distinguishes this from a computed
zero-plane result. When no Fine candidate exists, Coarse fallback automatically
runs RANSAC as before. Fine selection replaces only the red marker in panel 2
and fitting geometry in panel 3; it preserves Coarse/Fine clouds, heatmaps and
cameras. `fine_result.fine_search_radius_mm` supplies the display framing without
changing score evaluation. Fine picking occurs only in panel 2 and queues its
Fine voxel ID with the Coarse scene ID; stale selections are ignored. No display step
recomputes or mutates numerical scores/memberships. R/Q/Esc work in all panels.

For each unit plane $(n,d)$, project the selected query to
$c=q_f-(n^Tq_f+d)n$. Draw the disk at $c$ with radius
$\sqrt{\max(0,R^2-|n^Tq_f+d|^2)}$, so it is the plane/sphere intersection and
stays inside the RANSAC ball. These disks illustrate infinite plane models, not
observed patch boundaries. Planes/inliers use extraction-order red, green and
blue. Retain the alpha-0.60 lit-transparency material, small emissive color,
3-pixel rims/two diameters and normal arrows (starting on the projected plane).
Rims and diameters are display guides, not extracted edges or intersections.

The existing display-only worker, startup-only scene upload, camera picking,
R/Q/Esc handling and bounded Filament shutdown remain. Subsequent IPC messages
carry results/selected IDs; `--no-3d-view` retains RGB/depth Coarse selection and
console reporting. There is no plane/corner refinement, edge detection,
intersection or corner validation in this experiment. Sparse Fine-voxel neighborhoods
can yield no Fine candidate, finite RANSAC can miss weak planes, and hard removal
assigns intersection support to the earliest accepted plane. Downsampling reduces
support counts and changes centroids, so fixed `min_inliers` may require adjustment
for sparse scenes. This is the intended change from pre-voxel support.

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
larger score; equal scores favor the smaller scene index. Every NMS survivor is
passed to the same CornerVINS validator pipeline.

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

`settings.py` presents defaults, resolved validator fields, and CLI groups in
the same execution order: query geometry, CornerVINS extraction, padded
projection/boundary, weighted Hough, continuous 2-D line refinement, 3-D line
restoration/merging, and graph-local refinement. The former multiscale-Harris,
SAH-graph, and independent Edge3D diagnostic settings have been removed.
Detector settings end at dense NMS and carry the shared `ValidatorConfig` used
for every retained candidate. The superseded detector-only K-subspaces
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
post-NMS validator automatically. `--mode detector` retains that old behavior.
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
