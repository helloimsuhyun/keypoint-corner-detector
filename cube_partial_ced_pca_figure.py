#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive, publication-ready 3D cube / centroid / PCA / CED-weighted PCA.

python cube_partial_ced_pca_figure.py \
  --side 120 \
  --spacing 3 \
  --pca-radius 30 \
  --ced-radius 30 \
  --view-dir 1,1,1

This intentionally does NOT apply bilateral filtering or any separate detector's
fusion score. For a fair comparison, normal and weighted PCA use the same
R_pca neighborhood; CED is measured from the ENTIRE observed cloud, not a
neighborhood cropped around the clicked query point.

Usage:
    python cube_partial_ced_pca_figure.py
    python cube_partial_ced_pca_figure.py --view-dir 1,1,0
    python cube_partial_ced_pca_figure.py --view-dir 1,0,0 --ced-scale fixed
    python cube_partial_ced_pca_figure.py --no-show --output figures/cube_comparison

Click a point in panel (a) to update panels (b)-(d). Press S to save PNG and SVG.ㅉ
Positions in mm; covariance eigenvalues in mm^2.
"""ㅉ
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Settings:
    side_mm: float = 120.0
    spacing_mm: float = 3.0
    pca_radius_mm: float = 30.0
    ced_radius_mm: float = 12.0
    ced_cutoff: float = 0.10
    min_neighbors: int = 12
    noise_std_mm: float = 0.0
    view_direction: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ced_scale: str = "relative"
    dpi: int = 300
    output_prefix: str = "cube_ced_pca_comparison"


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    size = float(np.linalg.norm(v))
    if not np.isfinite(size) or size <= 1e-12:
        raise ValueError("The camera view direction must be nonzero")
    return v / size


def visible_cube_points(cfg: Settings) -> tuple[np.ndarray, list[str], float]:
    """Sample only the outward-facing planes of a convex solid cube.

    Face visibility assumes an orthographic camera looking toward the origin.
    Thus the hidden back/bottom/left faces are never added to the point cloud.
    Duplicated samples on a shared visible edge are deduplicated.
    """
    intervals = max(2, int(np.ceil(cfg.side_mm / cfg.spacing_mm)))
    t = np.linspace(-cfg.side_mm / 2, cfg.side_mm / 2, intervals + 1)
    a, b = np.meshgrid(t, t, indexing="ij")
    direction = normalize(np.array(cfg.view_direction))
    faces = []
    chunks = []
    for axis, name in enumerate(("X", "Y", "Z")):
        if abs(direction[axis]) < 1e-10:
            continue  # grazing/side-on plane is not an observed face
        sign = 1.0 if direction[axis] > 0 else -1.0
        xyz = np.empty((a.size, 3), dtype=float)
        others = [j for j in range(3) if j != axis]
        xyz[:, axis] = sign * cfg.side_mm / 2
        xyz[:, others[0]] = a.ravel()
        xyz[:, others[1]] = b.ravel()
        chunks.append(xyz)
        faces.append(f"{name}{'+' if sign > 0 else '-'}")
    if not chunks:
        raise ValueError("No visible cube faces")
    # Unique uses exact equal grid coordinates generated from the same t vector.
    points = np.unique(np.concatenate(chunks, axis=0), axis=0)
    if cfg.noise_std_mm > 0:
        points = points + np.random.default_rng(20261113).normal(
            0.0, cfg.noise_std_mm, size=points.shape
        )
    return points, faces, cfg.side_mm / intervals


def compute_raw_ced_all(points: np.ndarray, tree: cKDTree, radius: float) -> np.ndarray:
    """Raw centroid offset computed for every globally observed sample."""
    result = np.zeros(len(points), dtype=float)
    neighborhoods = tree.query_ball_point(points, radius + 1e-8, workers=-1)
    for idx, ids in enumerate(neighborhoods):
        if len(ids) >= 3:
            offset = points[ids].mean(axis=0) - points[idx]
            result[idx] = np.linalg.norm(offset) / radius
    return result


@dataclass
class PCAResult:
    centroid: np.ndarray
    eigenvalues: np.ndarray  # descending: lambda_1 >= lambda_2 >= lambda_3
    eigenvectors: np.ndarray  # columns correspond to descending eigenvalues
    effective_count: float


def calculate_pca(points: np.ndarray, weights: np.ndarray | None,
                  minimum_count: int) -> PCAResult | None:
    if weights is None:
        if len(points) < minimum_count:
            return None
        centroid = points.mean(axis=0)
        delta = points - centroid
        covariance = delta.T @ delta / len(points)
        effective = float(len(points))
    else:
        w = np.asarray(weights, dtype=float)
        active = w > 0
        if np.count_nonzero(active) < minimum_count:
            return None
        q, w = points[active], w[active]
        total = float(w.sum())
        effective = total**2 / float(np.dot(w, w))
        if total <= 0 or effective < minimum_count:
            return None
        centroid = np.average(q, axis=0, weights=w)
        delta = q - centroid
        covariance = (delta * w[:, None]).T @ delta / total
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return PCAResult(centroid, np.maximum(0.0, values[order]),
                     vectors[:, order], effective)


class FigureApp:
    def __init__(self, cfg: Settings, show: bool = True):
        # Set backend before importing pyplot in script/main, to support headless rendering.
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        self.plt = plt
        self.cfg = cfg
        self.points, self.faces, self.actual_spacing = visible_cube_points(cfg)
        self.tree = cKDTree(self.points)
        self.ced = compute_raw_ced_all(self.points, self.tree, cfg.ced_radius_mm)
        v = normalize(np.array(cfg.view_direction))
        self.elevation = float(np.degrees(np.arcsin(v[2])))
        self.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
        self.selected = self.initial_index(v)

        self.fig = plt.figure(figsize=(21.4, 6.9), facecolor="white")
        self.axes = [self.fig.add_subplot(1, 4, j + 1, projection="3d")
                     for j in range(4)]
        self.fig.subplots_adjust(left=.016, right=.934, bottom=.14,
                                 top=.82, wspace=.035)
        self.sm = ScalarMappable(norm=Normalize(0, 1), cmap="turbo")
        self.colorbar_ax = self.fig.add_axes([.941, .270, .010, .38])
        self.cb = self.fig.colorbar(self.sm, cax=self.colorbar_ax)
        self.cb.set_label("Raw CED", fontsize=10, rotation=90, labelpad=10)
        self.cb.ax.tick_params(labelsize=8)
        self.fig.suptitle("Partial cube observation  |  Centroid displacement & PCA",
                          fontsize=17, fontweight="semibold", y=.973)
        self.info = self.fig.text(
            .5, .076, "", ha="center", va="center", fontsize=10,
            color="#26364a")
        self.footer = self.fig.text(
            .5, .036,
            "Click a visible sample in (a) to update  (b)–(d)   •   Press S to export PNG + SVG",
            ha="center", va="center", fontsize=9, color="#66758b")
        self.draw_all()
        if show:
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def initial_index(self, view: np.ndarray) -> int:
        # Choose a point on an actually observed face, close to a vertex but not
        # at the perfectly symmetric 3-plane intersection (ambiguous eigenbasis).
        h = self.cfg.side_mm / 2
        target = np.array([h - 9, h - 15, h - 6], dtype=float)
        nonzero = np.flatnonzero(np.abs(view) > 1e-10)
        for i in range(3):
            if view[i] < 0:
                target[i] *= -1
        target[nonzero[0]] = np.sign(view[nonzero[0]]) * h
        return int(self.tree.query(target)[1])

    def local_ids(self, radius_mm: float) -> np.ndarray:
        p = self.points[self.selected]
        return np.asarray(self.tree.query_ball_point(p, radius_mm + 1e-8), dtype=int)

    def scene_axes(self, ax, center, half_width, show_coords=True):
        limits = [(center[j] - half_width, center[j] + half_width) for j in range(3)]
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
        ax.set_box_aspect((1, 1, 1), zoom=.87)
        ax.view_init(elev=self.elevation, azim=self.azimuth, vertical_axis='z')
        ax.set_proj_type("ortho")
        ax.tick_params(labelsize=7, pad=-2)
        if show_coords:
            ax.set_xlabel("x (mm)", fontsize=8, labelpad=-2)
            ax.set_ylabel("y (mm)", fontsize=8, labelpad=-2)
            ax.set_zlabel("z (mm)", fontsize=8, labelpad=-2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((.75, .8, .85, .3))
            axis.line.set_color("#98a9bd")
        ax.grid(False)

    @staticmethod
    def text_box(ax, content: str, y: float = 0.93, color: str = "#203047"):
        ax.text2D(.04, y, content, transform=ax.transAxes, fontsize=9.2,
                  va="top", ha="left", linespacing=1.45, color=color,
                  bbox=dict(boxstyle="round,pad=.36", fc="white", ec="#dce4ed",
                            alpha=.93, lw=.7))

    def title(self, ax, label: str, heading: str):
        ax.set_title(f"{label}  {heading}", loc="center", fontsize=12,
                     fontweight="semibold", pad=13)

    @staticmethod
    def dot(ax, location, color, size=80, marker='o', edge='#ffffff'):
        ax.scatter(*np.asarray(location), c=[color], s=size, marker=marker,
                   edgecolor=edge, linewidth=.9, depthshade=False, zorder=30)

    @staticmethod
    def draw_vector(ax, origin, target, color, label=None):
        d = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
        if np.linalg.norm(d) < 1e-8:
            return
        ax.plot([origin[0], target[0]], [origin[1], target[1]],
                [origin[2], target[2]], color="white", lw=4.2, alpha=.95)
        ax.quiver(*origin, *d, color=color, arrow_length_ratio=.17,
                  linewidth=2.6, normalize=False)
        if label is not None:
            t = origin + .95 * d
            ax.text(*t, label, color=color, fontsize=11, fontweight='bold')

    def draw_pca(self, ax, pca: PCAResult | None, query: np.ndarray, radius: float):
        self.dot(ax, np.zeros(3), "#ed6a33", 125, "*")
        if pca is None:
            self.text_box(ax, "PCA undefined:\ninsufficient effective samples", color="#b54a46")
            return
        c = pca.centroid - query
        self.dot(ax, c, "#22212c", 84, "X")
        for j, color in enumerate(("#c12675", "#008c99")):
            # Sigma (not lambda) is a spatial length. Display exactly +/- 1 sigma.
            scale = float(np.sqrt(max(0., pca.eigenvalues[j])))
            direction = pca.eigenvectors[:, j]
            end = c + direction * scale
            opposite = c - direction * scale
            ax.plot([opposite[0], c[0]], [opposite[1], c[1]],
                    [opposite[2], c[2]], color=color, lw=1.8, alpha=.8, ls='--')
            self.draw_vector(ax, c, end, color, f"$v_{j+1}$")
        ratio = pca.eigenvalues[1] / pca.eigenvalues[0] if pca.eigenvalues[0] > 1e-12 else np.nan
        warning = "\n(axis orientation unstable)" if ratio > .98 else ""
        self.text_box(ax,
            f"$\\lambda_1$ = {pca.eigenvalues[0]:.2f} mm$^2$\n"
            f"$\\lambda_2$ = {pca.eigenvalues[1]:.2f} mm$^2$\n"
            f"$\\lambda_2/\\lambda_1$ = {ratio:.3f}" + warning)

    def draw_all(self):
        p = self.points[self.selected]
        ced_ids = self.local_ids(self.cfg.ced_radius_mm)
        pca_ids = self.local_ids(self.cfg.pca_radius_mm)
        support = self.points[pca_ids]
        weights = self.ced[pca_ids].copy()
        weights[weights < self.cfg.ced_cutoff] = 0.0
        normal = calculate_pca(support, None, self.cfg.min_neighbors)
        weighted = calculate_pca(support, weights, self.cfg.min_neighbors)
        centroid = self.points[ced_ids].mean(axis=0)
        displacement = centroid - p
        r = float(np.linalg.norm(displacement))

        for ax in self.axes:
            ax.cla()
        a, b, c, d = self.axes
        self.title(a, "(a)", "Observed cube")
        a.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                  s=3.2, c="#8ca6b3", alpha=.76, linewidth=0,
                  depthshade=False, rasterized=True)
        self.dot(a, p, "#ed6a33", 155, "*")
        a.scatter(*self.points[pca_ids].T, s=6, c="#29a5b0", alpha=.38,
                  linewidth=0, depthshade=False, rasterized=True)
        self.dot(a, p, "#ed6a33", 155, "*")
        self.scene_axes(a, [0, 0, 0], self.cfg.side_mm / 2 * 1.15)
        self.text_box(a, f"Visible faces: {', '.join(self.faces)}\n"
                         f"Points: {len(self.points):,}\n"
                         f"Selected: ({p[0]:g}, {p[1]:g}, {p[2]:g}) mm", y=.94)

        # (b) raw CED neighborhood and vector.
        self.title(b, "(b)", "Centroid displacement")
        x_ced = self.points[ced_ids] - p
        b.scatter(*x_ced.T, s=15, c="#68abb5", alpha=.78,
                  depthshade=False, edgecolors='none')
        self.dot(b, [0, 0, 0], "#ed6a33", 150, "*")
        self.dot(b, displacement, "#9a45bb", 95, "X")
        self.draw_vector(b, np.zeros(3), displacement, "#a146ba", "$\\Delta$")
        self.scene_axes(b, [0, 0, 0], self.cfg.ced_radius_mm * 1.07)
        self.text_box(b, f"$\\|\\mu_{{CED}}-p\\|$ = {r:.2f} mm\n"
                         f"CED(p) = {r/self.cfg.ced_radius_mm:.3f}\n"
                         f"$R_{{CED}}$ = {self.cfg.ced_radius_mm:g} mm   N = {len(ced_ids)}")

        # (c) standard PCA, unweighted.
        self.title(c, "(c)", "Standard PCA")
        relative = support - p
        c.scatter(*relative.T, s=14, c="#73afbf", alpha=.74,
                  depthshade=False, edgecolors='none')
        self.scene_axes(c, [0, 0, 0], self.cfg.pca_radius_mm * 1.07)
        self.draw_pca(c, normal, p, self.cfg.pca_radius_mm)

        # (d) CED-weighted PCA; raw-CED heatmap from same support as (c).
        self.title(d, "(d)", "CED-weighted PCA")
        observed = self.ced[pca_ids]
        if self.cfg.ced_scale == 'fixed':
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.percentile(observed, 2))
            hi = float(np.percentile(observed, 98))
            if hi <= lo + 1e-8:
                lo, hi = max(0.0, lo - .01), hi + .01
        from matplotlib.colors import Normalize
        self.sm.set_norm(Normalize(vmin=lo, vmax=hi))
        self.cb.update_normal(self.sm)
        self.cb.set_label("Raw CED (" + ("fixed" if self.cfg.ced_scale == "fixed" else "local") + ")", fontsize=9)
        self.colorbar_ax.set_title("CED", fontsize=9, pad=7)
        d.scatter(*relative.T, s=18, c=observed, cmap="turbo",
                  vmin=lo, vmax=hi, alpha=.93, edgecolors='none',
                  depthshade=False, rasterized=True)
        self.scene_axes(d, [0, 0, 0], self.cfg.pca_radius_mm * 1.07)
        self.draw_pca(d, weighted, p, self.cfg.pca_radius_mm)
        if weighted is not None:
            self.text_box(d,
                f"$N_{{eff}}$ = {weighted.effective_count:.1f}   "
                f"cutoff = {self.cfg.ced_cutoff:.2f}\n"
                f"Nonzero weights: {np.count_nonzero(weights)}/{len(weights)}",
                y=.19, color="#38495f")

        self.info.set_text(
            f"Camera direction = {self.cfg.view_direction}     |     "
            f"PCA radius = {self.cfg.pca_radius_mm:g} mm     |     "
            f"CED radius = {self.cfg.ced_radius_mm:g} mm     |     "
            "Axes are ±1σ (σ = √λ)     |     "
            "Pink: v₁     Teal: v₂"
        )
        self.fig.canvas.draw_idle()
        if normal is not None and weighted is not None:
            print(f"Point={p.round(2)}  CED={r/self.cfg.ced_radius_mm:.4f}  "
                  f"PCA λ1/λ2={normal.eigenvalues[:2].round(3)}  "
                  f"Weighted λ1/λ2={weighted.eigenvalues[:2].round(3)}  "
                  f"Neff={weighted.effective_count:.1f}")
        else:
            print(f"Point={p.round(2)}  CED={r/self.cfg.ced_radius_mm:.4f}  "
                  f"Normal PCA valid={normal is not None}  "
                  f"Weighted PCA valid={weighted is not None}")

    def on_click(self, event):
        if event.inaxes is not self.axes[0] or event.button != 1:
            return
        # Project all 3D samples using CURRENT camera matrix, so clicking still
        # works after rotating the overview with Matplotlib's 3D mouse controls.
        from mpl_toolkits.mplot3d import proj3d
        a = self.axes[0]
        x, y, z = proj3d.proj_transform(self.points[:, 0], self.points[:, 1],
                                        self.points[:, 2], a.get_proj())
        pixels = a.transData.transform(np.column_stack([x, y]))
        pixel_dist = np.linalg.norm(pixels - [event.x, event.y], axis=1)
        nearby = np.flatnonzero(pixel_dist <= 13.0)
        if len(nearby) == 0:
            return
        # Within ~4 pixels of the nearest screen position, use nearest camera depth.
        min_error = float(pixel_dist[nearby].min())
        candidates = nearby[pixel_dist[nearby] < min_error + 3.0]
        chosen = int(candidates[np.argmin(z[candidates])])
        if chosen == self.selected:
            return
        self.selected = chosen
        # Match the current overview angle for all detail panels.
        self.elevation, self.azimuth = float(a.elev), float(a.azim)
        self.draw_all()

    def on_key(self, event):
        if event.key and event.key.lower() == "s":
            self.save()

    def save(self):
        path = Path(self.cfg.output_prefix).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            result = path.with_suffix("." + ext)
            self.fig.savefig(result, dpi=self.cfg.dpi, bbox_inches="tight",
                             facecolor="white")
            print("Saved:", result.resolve())


def read_args() -> tuple[Settings, bool]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", type=float, default=120, help="Cube side (mm)")
    parser.add_argument("--spacing", type=float, default=3, help="Surface sample pitch (mm)")
    parser.add_argument("--view-dir", default="1,1,1",
                        help="Camera position direction: 1,1,1 / 1,1,0 / 1,0,0")
    parser.add_argument("--pca-radius", type=float, default=30)
    parser.add_argument("--ced-radius", type=float, default=30)
    parser.add_argument("--ced-cutoff", type=float, default=.10)
    parser.add_argument("--min-neighbors", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0)
    parser.add_argument("--ced-scale", choices=("relative", "fixed"), default="relative")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", default="cube_ced_pca_comparison")
    parser.add_argument("--no-show", action="store_true", help="Render PNG/SVG without interactive window")
    args = parser.parse_args()
    try:
        direction = tuple(float(v.strip()) for v in args.view_dir.split(','))
        if len(direction) != 3:
            raise ValueError()
        normalize(np.array(direction))
    except ValueError:
        parser.error("--view-dir must contain three values and be nonzero (e.g. 1,1,1)")
    if (min(args.side, args.spacing, args.pca_radius, args.ced_radius) <= 0
            or args.ced_cutoff < 0 or args.noise_std < 0 or args.min_neighbors < 3):
        parser.error("Invalid nonpositive radius/side/pitch or negative cutoff/noise")
    settings = Settings(
        side_mm=args.side, spacing_mm=args.spacing,
        pca_radius_mm=args.pca_radius, ced_radius_mm=args.ced_radius,
        ced_cutoff=args.ced_cutoff, min_neighbors=args.min_neighbors,
        noise_std_mm=args.noise_std, view_direction=direction,
        ced_scale=args.ced_scale, dpi=args.dpi,
        output_prefix=args.output)
    return settings, not args.no_show


def main():
    cfg, show = read_args()
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    app = FigureApp(cfg, show=show)
    print(f"Observed faces = {app.faces};  N={len(app.points):,}; "
          f"actual surface spacing = {app.actual_spacing:.3f} mm")
    app.save()
    if show:
        app.plt.show()


if __name__ == "__main__":
    main()