"""Pure 2-D diagnostics for the active plane/edge pipeline."""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

matplotlib.use("Agg", force=True)


def save_soft_study_plots(summary, stability, output, *, matches=None):
    """Measured research curves. Every localization error is matched-only."""
    labels = ("A_hard_f64", "B_dense_f64", "C_partition_f64", "D_sparse4_f64",
              "D_sparse6_f64", "D_sparse8_f64", "C_partition_f32")
    noise = sorted({r["sigma_mm"] for r in summary})
    matched_errors = {}
    for row in matches or ():
        if row["query_source"] == "gt_near":
            matched_errors.setdefault((row["variant"], row["sigma_mm"]), []).append(row["error_mm"])

    def average(rows, field):
        values = np.asarray([r[field] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else np.nan

    def rows_for(label, sigma=None, source="gt_near"):
        return [r for r in summary if r["variant"] == label and r["query_source"] == source
                and (sigma is None or r["sigma_mm"] == sigma)]

    def make_curves(filename, metrics, titles):
        fig = Figure(figsize=(13, 5), constrained_layout=True)
        FigureCanvasAgg(fig)
        for index, (metric, title) in enumerate(zip(metrics, titles)):
            ax = fig.add_subplot(1, len(metrics), index + 1)
            for label in labels:
                if not rows_for(label):
                    continue
                values = []
                for sigma in noise:
                    samples = matched_errors.get((label, sigma), [])
                    if matches is not None and metric in ("error_median_mm", "error_p95_mm"):
                        value = (np.percentile(samples, 50 if metric == "error_median_mm" else 95)
                                 if samples else np.nan)
                    else:
                        value = average(rows_for(label, sigma), metric)
                    values.append(value)
                ax.plot(noise, values,
                        marker="o", label=label, linestyle="--" if label.endswith("f32") else "-")
            ax.set(xlabel="Coordinate noise sigma (mm)", ylabel=title, title=title)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        statistic = "Pooled matched observations; read recall alongside errors" if filename == "noise_localization.png" else "Macro mean across scenes"
        fig.suptitle(f"Fixed GT-near diagnostic; {statistic}; {int(max(r['seeds'] for r in summary))} seeds per condition")
        fig.savefig(output / filename, dpi=150)
        fig.clear()

    make_curves("noise_localization.png", ("error_median_mm", "error_p95_mm"),
                ("Matched median error (mm)", "Matched P95 error (mm)"))
    make_curves("noise_detection.png", ("recall_5mm", "false_positives_mean"),
                ("1:1 detection recall within 5 mm", "False positives per query"))
    fig = Figure(figsize=(10, 5), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for label in labels:
        ax.plot(noise, [average([r for r in stability if r["variant"] == label
            and r["query_source"] == "gt_near" and r["sigma_mm"] == sigma
            and r["observations"] >= 2], "position_rms_mm") for sigma in noise], marker="o", label=label)
    ax.set(xlabel="Coordinate noise sigma (mm)", ylabel="Repeated-position RMS (mm)",
           title="Matched observations only; mean is used for evaluation, never localization")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output / "repeatability.png", dpi=150)
    fig.clear()
    fig = Figure(figsize=(15, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(121)
    variants = sorted({r["variant"] for r in summary})
    bottom = np.zeros(len(variants))
    for stage in ("roi", "clustering", "projection", "quantile", "sigmoid", "normalization", "accumulation", "nms"):
        values = np.asarray([1000 * average(rows_for(label, source="ced"), stage + "_s_mean") for label in variants])
        ax.barh(variants, values, left=bottom, label=stage)
        bottom += values
    ax.set(xlabel="Mean ms per actual CED query", title="Disjoint instrumented numerical stages")
    ax.legend(fontsize=8, ncol=2)
    ax = fig.add_subplot(122)
    ax.barh(variants, [1000 * average(rows_for(label, source="ced"), "frame_refinement_s_mean") for label in variants])
    ax.set(xlabel="Mean additional ms per frame (all CED queries)", title="Includes shared spatial index and per-query overhead")
    fig.savefig(output / "stage_times.png", dpi=150)
    fig.clear()
    fig = Figure(figsize=(10, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    errors = []
    for label in variants:
        rows = rows_for(label, source="ced")
        x, y = 1000 * average(rows, "total_s_mean"), average(rows, "error_median_mm")
        ax.scatter(x, y, s=30 + 100 * average(rows, "recall_5mm"), label=label,
                   marker="x" if label.endswith("f32") else "o")
        errors.append(y)
    ax.set(xlabel="Mean ms per CED query", ylabel="Matched median error (mm)",
           title="Size encodes recall; missing detections are excluded from error")
    ax.grid(alpha=0.25)
    finite = np.asarray(errors)[np.isfinite(errors)]
    ax.set_ylim(0, max(0.5, 1.2 * float(finite.max())) if len(finite) else 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.savefig(output / "tradeoff.png", dpi=150)
    fig.clear()


def make_extremity_summary_figure(result):
    """Display stored votes and candidate evidence; never rerun the detector."""
    fig = Figure(figsize=(10, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(211)
    ids = np.flatnonzero(result["raw_votes"] > 0)
    ax.scatter(result["raw_votes"][ids], result["normalized_score"][ids], s=12,
               c=result["opportunity_count"][ids], cmap="viridis")
    chosen = result["candidate_local_indices"]
    ax.scatter(result["raw_votes"][chosen], result["normalized_score"][chosen],
               facecolors="none", edgecolors="magenta", s=70, label="NMS candidates")
    ax.axhline(result["config"].min_vote_fraction, color="gray", linestyle="--")
    ax.set(xlabel="Raw votes", ylabel="Votes / (directions * opportunities)",
           title=f"BALL: {len(chosen)} unverified corner candidates | color: opportunity count")
    ax.legend()
    text = fig.add_subplot(212)
    text.axis("off")
    rows = ["scene / raw index   XYZ (mm)                    votes fraction component offset(mm)"]
    for candidate in result["candidates"][:9]:
        rows.append(f"{candidate['point_index']:5d}/{candidate['raw_point_index']:<5d}  {np.round(candidate['position_mm'], 2).tolist()}"
                    f"  {candidate['raw_votes']:7.3g}  {candidate['normalized_score']:.4f}"
                    f"  {candidate['cluster_id']:5d}  {candidate['query_distance_mm']:7.2f}")
    if len(chosen) > 9:
        rows.append(f"... {len(chosen) - 9} more; full list printed in terminal")
    rows.append("Candidates are hypotheses, not geometrically verified corners.")
    rows.append("Yellow: query core; gray: search padding. Each object component votes independently.")
    text.text(0, 1, "\n".join(rows), va="top", fontsize=9, family="monospace")
    return fig


def make_plane_edge_debug_figure(result, clustering, points_mm, edge_config,
                                 plane_index=0, peak_index=0):
    """Read-only review of stored padded-boundary/Hough/3-D outputs."""
    from ..plane_edge_graph_pipeline import _plane_basis

    raw_planes = clustering.get('global_planes', [])
    if not raw_planes:
        fig = Figure(figsize=(13, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.text(.5, .5, 'No clustered planes', ha='center', va='center', transform=ax.transAxes)
        return fig

    plane_index = int(np.clip(plane_index, 0, len(raw_planes)-1))
    raw = raw_planes[plane_index]
    pid = int(raw['plane_id'])
    fitted = next((x for x in result['planes'] if int(x['plane_id']) == pid), None)
    points = np.asarray(points_mm, dtype=float)
    if fitted is None:
        ids = np.unique(np.asarray(raw['support_indices'], dtype=np.int64))
        ids = ids[(ids >= 0) & (ids < len(points))]
        center = np.mean(points[ids], axis=0) if len(ids) else np.zeros(3)
        try:
            u,v = _plane_basis(raw['representative_normal'])
        except (KeyError, ValueError):
            u,v = np.array([1.,0,0]), np.array([0.,1.,0])
        uv = np.empty((0, 2), dtype=float)
    else:
        ids = np.asarray(fitted.get('projection_indices', fitted['support_indices']), dtype=np.int64)
        center = np.asarray(fitted['center_mm'], dtype=float)
        u, v = np.asarray(fitted['u'], dtype=float), np.asarray(fitted['v'], dtype=float)
        uv = np.asarray(fitted['projection_uv'], dtype=float)
    boundary = {} if fitted is None else fitted['boundary']
    core = np.zeros(len(uv), dtype=bool) if fitted is None else np.asarray(fitted['projection_core_mask'], dtype=bool)
    boundary_mask = np.zeros(len(uv), dtype=bool) if fitted is None else np.asarray(boundary['boundary_mask'], dtype=bool)
    distance = np.full(len(uv), np.nan) if fitted is None else np.asarray(boundary['distance_mm'], dtype=float)
    valid = np.zeros(len(uv), dtype=bool) if fitted is None else np.asarray(boundary['valid_direction'], dtype=bool)
    inward = np.zeros_like(uv) if fitted is None else np.asarray(boundary['inward_normal_uv'], dtype=float)
    vote = {} if fitted is None else fitted['line_vote']
    vote_mask = np.zeros(len(uv), dtype=bool) if fitted is None else np.asarray(vote['vote_mask'], dtype=bool)
    weights = np.zeros(len(uv), dtype=float) if fitted is None else np.asarray(vote['boundary_weights'], dtype=float)
    peaks = [] if fitted is None else list(vote.get('hough_peaks_after_nms', ()))
    peak_index = int(np.clip(peak_index, 0, max(0,len(peaks)-1)))
    selected = peaks[peak_index] if peaks else None

    fig = Figure(figsize=(24.0, 11.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4)
    (ax_proj, ax_distance, ax_weight, ax_select,
     ax_accum, ax_hough, ax_refine, ax_final) = axes.ravel()
    data_axes = (
        ax_proj, ax_distance, ax_weight, ax_select,
        ax_hough, ax_refine, ax_final,
    )
    for ax in data_axes:
        ax.set_xlabel('u [mm]');ax.set_ylabel('v [mm]')
        ax.set_aspect('equal', adjustable='box');ax.grid(alpha=.12)
    if len(uv):
        ax_proj.scatter(uv[~core,0], uv[~core,1], s=5, c='#cbd5e1', alpha=.5, label='halo')
        ax_proj.scatter(uv[core,0], uv[core,1], s=8, c='#2563eb', alpha=.65, label='core')
        center_uv = np.asarray(fitted['projection_center_uv']) if fitted is not None else np.zeros(2)
        ax_proj.scatter(center_uv[0], center_uv[1], marker='X', s=80, c='#dc2626', label='query')
        ax_proj.legend(fontsize=7)
        good = valid & np.isfinite(distance)
        ax_distance.scatter(uv[~good,0], uv[~good,1], s=4, c='#d1d5db')
        if np.any(good):
            im = ax_distance.scatter(uv[good,0], uv[good,1], c=distance[good], s=12,
                                     cmap='turbo_r', vmin=0, vmax=edge_config.boundary_radius_mm)
            fig.colorbar(im, ax=ax_distance, shrink=.7, label='distance [mm]')
        ax_weight.scatter(uv[~vote_mask,0], uv[~vote_mask,1], s=4, c='#d1d5db')
        if np.any(vote_mask):
            im = ax_weight.scatter(uv[vote_mask,0], uv[vote_mask,1], c=weights[vote_mask],
                                   s=18, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax_weight, shrink=.7, label='continuous weight')
        ax_select.scatter(uv[:,0], uv[:,1], s=4, c='#d1d5db')
        ax_select.scatter(uv[boundary_mask,0], uv[boundary_mask,1], s=18, c='#f97316', label='boundary')
        ax_select.scatter(uv[vote_mask,0], uv[vote_mask,1], s=28, facecolors='none', edgecolors='#eab308', label='Hough core voter')
        ax_select.legend(fontsize=7)
        ax_hough.scatter(uv[:,0], uv[:,1], s=4, c='#d1d5db')
        ax_hough.scatter(uv[vote_mask,0], uv[vote_mask,1], s=9, c='#b91c1c')
    ax_proj.set_title(f'3a. P{pid}: core + real observed halo')
    ax_distance.set_title('3b. Continuous boundary-distance estimator')
    ax_weight.set_title('3c. Continuous boundary weights (core voters)')
    ax_select.set_title('3d. Boundary → weighted Hough selection')

    accumulator = np.asarray(vote.get('accumulator', np.empty((0,0))), dtype=float)
    rho_axis = np.asarray(vote.get('rho_mm', ()), dtype=float)
    theta_axis = np.rad2deg(np.asarray(vote.get('theta_rad', ()), dtype=float))
    if accumulator.size and len(rho_axis) and len(theta_axis):
        im = ax_accum.imshow(accumulator, origin='lower', aspect='auto', cmap='magma',
                             extent=(rho_axis[0], rho_axis[-1], theta_axis[0], theta_axis[-1]))
        fig.colorbar(im, ax=ax_accum, shrink=.7, label='weighted vote')
        ax_accum.scatter([p['rho_mm'] for p in peaks],
                         [np.rad2deg(p['theta_rad']) for p in peaks], c='cyan', marker='x', s=18)
    ax_accum.set(title=f'3e. Stored accumulator | peaks={len(peaks)}', xlabel='rho [mm]', ylabel='theta [deg]')

    # Draw every detected peak faintly, and selected Hough stripe brightly.
    if len(uv):
        width = max(float(np.ptp(uv[:,0])),float(np.ptp(uv[:,1])),1.)*1.1
    else:
        width=1.
    for i,peak in enumerate(peaks):
        normal=np.asarray(peak['normal'],dtype=float)
        d=np.array([-normal[1], normal[0]])
        c=normal*float(peak['rho'])
        endpoints=np.stack([c-width*d,c+width*d])
        ax_hough.plot(endpoints[:,0],endpoints[:,1],color='#94a3b8',lw=.7,alpha=.5)
    if selected is not None:
        n=np.asarray(selected['normal'],dtype=float)
        d=np.array([-n[1],n[0]])
        center2=n*float(selected['rho'])
        endpoints=np.stack([center2-width*d, center2+width*d])
        ax_hough.plot(endpoints[:,0],endpoints[:,1],c='#eab308',lw=2.1)
        matched=uv[np.asarray(selected['indices'],dtype=int)]
        if len(matched):ax_hough.scatter(matched[:,0],matched[:,1],s=12,c='#eab308',rasterized=True)
    ax_hough.set_title(f'3f. Hough peak geometry: selected {peak_index+1 if peaks else 0}/{len(peaks)}')

    refinement = vote.get('line_refinement', {})
    before_nms = list(refinement.get('refined_segments_before_nms', ()))
    after_nms = list(refinement.get('refined_segments_after_nms', ()))
    palette = matplotlib.colormaps['tab10']
    if len(uv):
        ax_refine.scatter(uv[:,0], uv[:,1], s=4, c='#d1d5db', alpha=.65)
        ax_final.scatter(uv[:,0], uv[:,1], s=4, c='#d1d5db', alpha=.65)
    # 3g is deliberately pre-score-filter/pre-final-NMS: every seed remains
    # inspectable, including failed optimizations.
    for candidate_index, candidate in enumerate(before_nms):
        color = palette(candidate_index % 10)
        theta0 = float(candidate['initial_theta_rad'])
        normal0 = np.array([np.cos(theta0), np.sin(theta0)])
        direction0 = np.array([-normal0[1], normal0[0]])
        point0 = normal0 * float(candidate['initial_rho_mm'])
        initial_line = np.stack((point0-width*direction0, point0+width*direction0))
        ax_refine.plot(
            initial_line[:,0], initial_line[:,1], ls='--', lw=1.0,
            color=color, alpha=.7,
        )
        if 'start_uv' in candidate:
            segment = np.stack((candidate['start_uv'], candidate['end_uv']))
            ax_refine.plot(
                segment[:,0], segment[:,1], lw=2.0, color=color,
                label=(f"S{candidate['seed_index']} "
                       f"{'OK' if candidate['success'] else 'FAIL'}"),
            )
        if not candidate.get('success', False):
            theta = float(candidate['final_theta_rad'])
            normal = np.array([np.cos(theta), np.sin(theta)])
            point = normal * float(candidate['final_rho_mm'])
            ax_refine.scatter(
                point[0], point[1], marker='x', s=55, linewidths=2,
                color=color,
            )
    if before_nms:
        ax_refine.legend(loc='upper right', fontsize=6)
        selected_candidate = before_nms[int(np.clip(
            peak_index, 0, len(before_nms)-1,
        ))]
        inset = ax_refine.inset_axes([.04, .04, .48, .30])
        support_s = np.asarray(selected_candidate.get('s_mm', ()), dtype=float)
        if len(support_s):
            params = refinement.get('parameters', {})
            c_b = max(float(params.get('c_B', 1.0)), 1e-12)
            c_o = max(float(params.get('c_O', 1.0)), 1e-12)
            inset.plot(support_s, np.asarray(selected_candidate['B'])/c_b,
                       color='#2563eb', lw=.8, label='B/cB')
            inset.plot(support_s, np.asarray(selected_candidate['O'])/c_o,
                       color='#dc2626', lw=.8, label='O/cO')
            inset.plot(support_s, selected_candidate['q'],
                       color='#16a34a', lw=1.0, label='q')
            inset.axhline(
                float(selected_candidate.get('support_threshold', 0.0)),
                color='#111827', ls=':', lw=.7,
            )
            inset.legend(fontsize=5, ncol=3, loc='upper right')
            inset.tick_params(labelsize=5)
    ax_refine.set_title(
        f'3g. Continuous refine before final NMS: {len(before_nms)} '
        '(dashed=seed, solid=finite)',
    )

    for final_index, segment_data in enumerate(after_nms):
        color = palette(int(segment_data['seed_index']) % 10)
        segment = np.stack((segment_data['start_uv'], segment_data['end_uv']))
        ax_final.plot(segment[:,0], segment[:,1], lw=2.8, color=color)
        midpoint = np.mean(segment, axis=0)
        ax_final.text(
            midpoint[0], midpoint[1],
            f"L={segment_data['length_mm']:.1f} mm\n"
            f"S={segment_data['edge_support_score']:.2f}",
            fontsize=6, color=color,
        )
        support_ids = np.asarray(
            segment_data['boundary_support_uv_indices'], dtype=np.int64,
        )
        if len(support_ids):
            ax_final.scatter(
                uv[support_ids,0], uv[support_ids,1], s=9,
                color=color, alpha=.7, rasterized=True,
            )
    ax_final.set_title(
        f'3h. Validity filter + finite-segment NMS final: {len(after_nms)} '
        f"(suppressed={refinement.get('final_nms_suppressed', 0)})",
    )
    if len(uv):
        pad = max(float(np.ptp(uv[:,0])), float(np.ptp(uv[:,1])), 1.) * .06
        for ax in data_axes:
            ax.set_xlim(float(uv[:,0].min())-pad, float(uv[:,0].max())+pad)
            ax.set_ylim(float(uv[:,1].min())-pad, float(uv[:,1].max())+pad)
    if fitted is None:
        title = f'P{pid} ({plane_index+1}/{len(raw_planes)}): no accepted plane fit'
    else:
        counts=fitted['stage_counts']
        title = (f'P{pid} ({plane_index+1}/{len(raw_planes)}) | padded continuous boundary | '
                 f'weighted Hough={counts["hough_peaks"]} | '
                 f'refined={counts.get("refined_before_nms", 0)}→'
                 f'{counts.get("refined_after_nms", 0)} | '
                 f'3-D PCA edges={counts["3d_accepted"]}')
    fig.suptitle(title,fontsize=12,weight='bold')
    fig.tight_layout(rect=(0,0,1,.95))
    return fig

def figure_to_rgb(figure):
    """Render into an owned RGB image independent of figure lifetime."""
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


__all__ = [
    "make_q10_validator_figure",
    "make_extremity_summary_figure",
    "figure_to_rgb",
    "make_plane_edge_debug_figure",
]




def make_projection_validator_figure(result):
    """Read cached UV projections; do not construct hulls, edges or new support."""
    patches = result["plane_patches"]
    evidence = result.get("plane_q10", [])
    figure = Figure(figsize=(max(1, len(patches)) * 4.2, 4.4), dpi=100)
    FigureCanvasAgg(figure)
    for i in range(max(1, len(patches))):
        axis = figure.add_subplot(1, max(1, len(patches)), i + 1)
        if patches:
            patch = patches[i]
            uv = patch["projection_uv"]
            if len(uv):
                axis.scatter(uv[:, 0], uv[:, 1], s=8, color=("crimson", "seagreen", "royalblue")[i % 3], alpha=.8)
            title = f"Plane {patch['plane_id']} | N={len(uv)}"
            if evidence:
                score = evidence[i]
                centroid = score["centroid_uv_mm"]
                axis.scatter(*centroid, color="teal", marker="x", s=40, label="centroid")
                axis.annotate("", xy=centroid, xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "teal", "lw": 2})
                value = "undefined" if score["q10"] is None else f"{score['q10']:.4f}"
                decision = "UNKNOWN" if score["passed"] is None else "PASS" if score["passed"] else "REJECT"
                title += f"\nQ10={value} | min={result['q10_threshold']:g} | {decision}"
                if score["reason"]:
                    axis.text(.02, .02, score["reason"], transform=axis.transAxes, fontsize=7)
            else:
                title += " | " + patch["status"]
            axis.set_title(title, fontsize=10)
        else:
            axis.set_title("No fitted planes")
        axis.scatter(0, 0, marker="+", color="darkorange", s=75, linewidths=2, label="query projected")
        axis.set(xlabel="U (mm)", ylabel="V (mm)", aspect="equal")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    status = result.get("validation_status", "projection only")
    count = f" | {result['passed_plane_count']}/{len(patches)} planes pass" if evidence else ""
    figure.suptitle(f"New validator | query {result['query_index']} | {status}{count}",
                   color={"ACCEPT": "seagreen", "REJECT": "crimson"}.get(status, "darkorange"), y=.97)
    figure.subplots_adjust(left=.07, right=.985, bottom=.16, top=.79, wspace=.28)
    return figure


def q10_step_directions(record, step):
    """Display geometry from stored support/axis/Q10; never optimize or change results."""
    points = record["step_q10_points_mm"][step]
    origin = record["trace"]["positions_mm"][step]
    offsets = points - origin
    lengths = np.linalg.norm(offsets, axis=1)
    units = offsets[lengths > 1e-7] / lengths[lengths > 1e-7, None]
    return units, np.clip(units @ record["trace"]["mean_axes"][step], -1., 1.)


def draw_q10_validator_figure(figure, record, step):
    from ..plane_projection import make_tangent_basis
    from .view3d import q10_debug_voxel_primitives, q10_direction_primitives, q10_spherical_plane_primitives
    figure.clear()
    figure.set_facecolor('#f4f7fb')
    trace = record['trace']
    positions = trace['positions_mm']
    original, final = record['position'], record['refined_position_mm']
    step = int(np.clip(step, 0, len(positions)-1))
    origin = positions[step]
    direction_mode = record.get('direction_mode', 'unit_mean')
    q10 = trace['q10_values'][step]
    final_q10 = record['q10_value']
    refine_enabled = record.get('refine_enabled', True)
    q10_cc = 'ON' if record['refine_cc_enabled'] else 'OFF'
    planes = q10_spherical_plane_primitives(record)
    analysis = record.get('spherical_planes')
    validation = record.get('plane_antipodal')
    status_color = {'ACCEPT': '#16815d', 'REJECT': '#c7364f'}.get(record['status'], '#b77912')
    checks = ['UNKNOWN' if failed is None else 'REJECT' if failed else 'PASS'
              for failed in (record['q10_reject'], record['antipodal_reject'])]
    if record['q10_reject'] is True:
        checks[1] = 'SKIPPED (Q10)'
    if not refine_enabled:
        checks[0] = 'OFF'
    causes = ' + '.join('Q10' if reason == 'q10_below_threshold' else 'ANGULAR_SPAN'
                        for reason in record['rejection_reasons'])
    status = record['status'] + (f' [{causes}]' if causes else '')
    figure.suptitle(f"Q10 + spherical MaxGap  |  NMS {record['point_index']}  |  {status}",
                    x=.035, y=.975, ha='left', fontsize=16, weight='bold', color=status_color)
    quality = f'{final_q10:.4f}' if final_q10 is not None and np.isfinite(final_q10) else 'unavailable'
    quality_text = (f"Final Q10 {quality} / min {record['min_q10']:g}: {checks[0]}" if refine_enabled else
                    'Q10 refine OFF / min gate OFF')
    figure.text(.035, .928,
                f"{quality_text}     "
                f"MaxGap/span: {checks[1]}     Q10 CC {q10_cc}     Final: angular only",
                fontsize=10, color='#34445c')
    figure.text(.035, .895,
                f"Step {step}/{len(positions)-1}  |  radius {record['radius_mm']:g} mm  |  "
                f"voxel {record['voxel_mm']:g} mm  |  axis {direction_mode}  |  "
                f"NMS displacement {np.linalg.norm(final-original):.2f} mm",
                fontsize=9, color='#65738a')
    grid = figure.add_gridspec(2, 3, left=.035, right=.755, top=.84, bottom=.105, wspace=.27, hspace=.40)

    def physical_axes(axis, points, title):
        center = (points.min(axis=0) + points.max(axis=0))/2
        half = max(float(np.ptp(points, axis=0).max())*.56, 1.)
        axis.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
                 zlim=(center[2]-half, center[2]+half), xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(title, loc='left', fontsize=10, pad=9)
        axis.tick_params(labelsize=7)

    def sphere_plot(axis, directions, colors, title):
        # Display thinning only; cached histograms/support counts keep every sample.
        take = np.linspace(0, len(directions)-1, min(len(directions), 4000), dtype=int)
        if len(take):
            axis.scatter(*directions[take].T, c=np.asarray(colors)[take], s=5, alpha=.45)
        phi = np.linspace(0., 2*np.pi, 100)
        zero = np.zeros_like(phi)
        for ring in ((np.cos(phi), np.sin(phi), zero), (np.cos(phi), zero, np.sin(phi)),
                     (zero, np.cos(phi), np.sin(phi))):
            axis.plot(*ring, color='#bac6d6', linewidth=.6, alpha=.5)
        axis.set(xlim=(-1.32, 1.32), ylim=(-1.32, 1.32), zlim=(-1.32, 1.32))
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(title, loc='left', fontsize=10, pad=9)
        axis.set_xticks([-1, 0, 1]); axis.set_yticks([-1, 0, 1]); axis.set_zticks([-1, 0, 1])
        axis.tick_params(labelsize=7)

    path = np.vstack((original, positions)) - original
    trajectory = figure.add_subplot(grid[0, 0], projection='3d')
    points = record['step_points_mm'][step] - original
    trajectory.scatter(*points.T, s=3, color='#a7b2c1', alpha=.25)
    trajectory.plot(*path.T, 'o-', color='#5265aa', markersize=3, linewidth=2, label='Q10 path')
    trajectory.scatter(0, 0, 0, color='#e04848', s=35, label='NMS')
    trajectory.scatter(*(origin-original), color='#f1992d', s=45, label=f'step {step}')
    trajectory.scatter(*(final-original), color=status_color, s=65, marker='*', label='final')
    extent = np.vstack((points, path))
    direction = q10_direction_primitives(record, step)
    if direction is not None:
        delta = direction['tip_mm']-direction['origin_mm']
        trajectory.quiver(*(origin-original), *delta, color='#008e99', linewidth=2, arrow_length_ratio=.14)
        extent = np.vstack((extent, direction['tip_mm']-original))
    physical_axes(trajectory, extent, '1  Position / from original NMS')
    trajectory.legend(fontsize=7, loc='upper left', framealpha=.85)
    units, cosines = q10_step_directions(record, step)
    sphere = figure.add_subplot(grid[0, 1], projection='3d')
    colors = np.tile([1., .28, .62], (len(units), 1))
    colors[cosines >= q10-1e-12] = [.96, .75, .19]
    sphere_plot(sphere, units, colors, f'2  Q10 sphere / {direction_mode}' if refine_enabled else '2  Q10 refinement OFF')
    axis = trace['mean_axes'][step]
    if np.isfinite(axis).all() and np.isfinite(q10):
        u, v, d = make_tangent_basis(axis)
        phi = np.linspace(0., 2*np.pi, 100)
        ring = q10*d + np.sqrt(max(0., 1-q10*q10))*(np.cos(phi)[:, None]*u + np.sin(phi)[:, None]*v)
        sphere.plot(*ring.T, color='#008e99', linewidth=2)
        sphere.quiver(0, 0, 0, *axis, color='#008e99', length=1.2, linewidth=2, arrow_length_ratio=.13)
    histogram = figure.add_subplot(grid[0, 2])
    histogram.hist(cosines, bins=np.linspace(-1, 1, 41), color='#6792c4', alpha=.85)
    if np.isfinite(q10):
        histogram.axvline(q10, color='#008e99', linewidth=2, label=f'Q10 {q10:.4f}')
    if refine_enabled:
        histogram.axvline(record['min_q10'], color='#c7364f', linestyle='--', label=f"min {record['min_q10']:g}")
    histogram.set(xlim=(-1, 1), xlabel=f'cosine to Q10 {direction_mode} axis', ylabel='Count')
    histogram.set_title((f"3  Q10 support {trace['neighbor_counts'][step]} / ROI {trace['roi_counts'][step]}"
                         if refine_enabled else '3  Q10 minimum gate OFF'),
                        loc='left', fontsize=10, pad=9)
    if refine_enabled:
        histogram.legend(fontsize=7)
    histogram.grid(axis='y', alpha=.15)
    histogram.tick_params(labelsize=8)
    spacing = (f"NN {trace['cc_median_nn_mm'][step]:.2f} mm / epsilon {trace['cc_radii_mm'][step]:.2f} mm"
               if record['refine_cc_enabled'] else 'Q10 CC OFF / full R support')
    if not refine_enabled:
        spacing = 'No Q10 calculation / no relocation / no min gate'
    histogram.text(.02, .03, spacing, transform=histogram.transAxes, fontsize=7, color='#34445c')

    cloud = figure.add_subplot(grid[1, 0], projection='3d')
    data = q10_debug_voxel_primitives(record, len(positions)-1)
    points = data['points_mm'] - final
    cloud.scatter(*points.T, c=data['point_colors'], s=8, alpha=.7)
    cloud.scatter(0, 0, 0, color=status_color, s=65, marker='*')
    extent = np.vstack((points, np.zeros(3)))
    final_direction = q10_direction_primitives(record, len(positions)-1)
    if final_direction is not None:
        delta = final_direction['tip_mm'] - final
        cloud.quiver(0, 0, 0, *delta, color='#008e99', linewidth=2, arrow_length_ratio=.14)
        extent = np.vstack((extent, delta))
    for plane in planes:
        arc = .8*record['radius_mm']*plane['kept_arc']
        if len(arc):
            cloud.plot(*arc.T, color=plane['color'], linewidth=1.6)
            extent = np.vstack((extent, arc))
    physical_axes(cloud, extent, '4  Final voxel support / mm')

    anti_axis = figure.add_subplot(grid[1, 1], projection='3d')
    directions = analysis['directions'] if analysis is not None else np.empty((0, 3))
    sphere_plot(anti_axis, directions, np.tile([.68, .72, .78], (len(directions), 1)),
                '5  Great circles / projected inliers')
    for plane in planes:
        anti_axis.plot(*plane['circle'].T, color=plane['color'], linewidth=1.8)
        dots = plane['projected_points']
        take = np.linspace(0, len(dots)-1, min(len(dots), 4000), dtype=int)
        if len(take):
            anti_axis.scatter(*dots[take].T, c=plane['projected_colors'][take], s=9, alpha=.9)
        arc = 1.26*plane['kept_arc']
        if len(arc):
            anti_axis.plot(*arc.T, color=plane['color'], linewidth=1.8)
    history = figure.add_subplot(grid[1, 2])
    history.plot(trace['q10_values'], 'o-', color='#008e99', linewidth=2, markersize=5)
    history.axvline(step, color='#f1992d', linestyle='--', label=f'step {step}')
    if refine_enabled:
        history.axhline(record['min_q10'], color='#c7364f', linestyle=':', label='Q10 min')
    history.set(xlabel='Refinement step', ylabel='Q10 cosine')
    history.set_title('6  Cached Q10 history' if refine_enabled else '6  Q10 refinement OFF',
                      loc='left', fontsize=10, pad=9)
    history.set_xticks(np.arange(len(positions)))
    history.grid(alpha=.15)
    history.legend(fontsize=7)
    history.tick_params(labelsize=8)

    figure.text(.785, .89, 'PER-PLANE CLEANED SPAN', fontsize=10, weight='bold', color='#34445c')
    if planes:
        rose_grid = figure.add_gridspec(3, 2, left=.785, right=.985, top=.81, bottom=.105,
                                       width_ratios=(1.15, 1.), hspace=.65, wspace=.18)
        for row, plane in enumerate(planes):
            rose = figure.add_subplot(rose_grid[row, 0], projection='polar')
            rose.hist(np.radians(plane['angles_deg']), bins=36, range=(0., 2*np.pi),
                      color=plane['color'], edgecolor='white', linewidth=.3, alpha=.9, zorder=2)
            rose.set_theta_zero_location('E')
            rose.set_theta_direction(1)
            rose.set_xticks(np.radians([0, 90, 180, 270]))
            rose.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=7)
            rose.tick_params(axis='y', labelsize=6)
            rose.grid(alpha=.25)
            rose.set_title(f"P{plane['plane_index']+1}", color=plane['color'], weight='bold', fontsize=10, pad=16)
            detail = figure.add_subplot(rose_grid[row, 1])
            detail.axis('off')
            test = plane['test']
            if test is not None:
                height = rose.get_ylim()[1]
                if test['arc_start_deg'] is not None:
                    angles = np.radians(np.linspace(test['arc_start_deg'],
                        test['arc_start_deg'] + test['observed_span_deg'], 145))
                    rose.plot(angles, np.full(len(angles), .90*height), color=plane['color'], linewidth=2)
                rejected_angles = plane['angles_deg'][~test['kept_mask']]
                rose.scatter(np.radians(rejected_angles), np.full(len(rejected_angles), .97*height),
                             color=.3*plane['color'], marker='x', s=12, zorder=4)
                verdict = 'UNKNOWN' if not test['reliable'] else 'REJECT' if test['edge_like'] else 'PASS'
                color = {'REJECT': '#c7364f', 'PASS': '#16815d'}.get(verdict, '#b77912')
                score_text = (f"SPAN {test['observed_span_deg']:.1f}°\ngap {test['max_gap_deg']:.1f}°\n"
                              f"keep {test['kept_count']}/{test['raw_count']}\n"
                              f"out {test['discarded_count']} / tail {test['tail_removed_count']}")
            else:
                verdict, color = 'UNKNOWN', '#b77912'
                score_text = f"N {len(plane['angles_deg'])}\nInsufficient\nplane support"
            detail.text(0, .94, verdict, color=color, weight='bold', fontsize=10, va='top')
            detail.text(0, .72, score_text, fontsize=9, color='#34445c', va='top', linespacing=1.5)
            detail.text(0, .12, f"RMS {plane['residual_rms_deg']:.2f}°\nnew {plane['newly_explained_count']}",
                        fontsize=7, color='#65738a', va='top')
    else:
        empty = figure.add_axes([.79, .20, .19, .60])
        empty.axis('off')
        skipped = record['q10_reject'] is True
        empty.text(.05, .75, 'RANSAC SKIPPED' if skipped else 'ANGULAR CHECK UNKNOWN',
                   fontsize=10, weight='bold', color=status_color if skipped else '#b77912')
        empty.text(.05, .62, ('Q10 below threshold\nRejected before plane fitting' if skipped else
                             'No supported great circles\n\n' + record['reason'].replace('_', ' ')),
                   fontsize=9, color='#65738a', va='top', wrap=True)
    if validation is not None:
        membership = analysis['membership_count']
        summary = (f"K {len(planes)}/{analysis['requested_k']}  |  tolerance {analysis['tolerance_deg']:g}°  |  "
                   f"link ≤ {validation['max_link_gap_deg']:g}°  |  REJECT span ≥ {validation['edge_span_deg']:g}°  |  "
                   f"tail {'ON' if record['gap_tail_enabled'] else 'OFF'}  |  "
                   f"unassigned {np.count_nonzero(membership == 0)} / overlap {np.count_nonzero(membership > 1)}")
    else:
        summary = ('RANSAC skipped: converged Q10 below threshold' if record['q10_reject'] is True else
                   'Angular evidence unavailable: ' + record['reason'].replace('_', ' '))
    figure.text(.035, .045, summary, fontsize=9, color='#34445c')
    figure.text(.035, .018,
                'P1 orange / P2 green / P3 purple   ·   White voxels: overlapping planes   ·   '
                'Dim: discarded by all member planes   ·   Cyan arrow: Q10 direction', fontsize=8, color='#65738a')


def make_q10_validator_figure(record):
    figure = Figure(figsize=(18, 9.5))
    FigureCanvasAgg(figure)
    step = len(record["trace"]["positions_mm"]) - 1

    def set_step(value):
        draw_q10_validator_figure(figure, record, int(value))

    set_step(step)
    return {"figure": figure, "record": record, "set_step": set_step}
