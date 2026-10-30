"""Open3D views for active detector and plane/edge diagnostics."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from pointcloud_inputs.visualization import require_open3d
from ..plane_projection import make_tangent_basis, normalize


_PLANE_COLORS = np.asarray([
    [0.20, 0.65, 0.95],
    [1.00, 0.55, 0.10],
    [0.35, 0.85, 0.35],
], dtype=np.float64)

_REFINER_EDGE_COLORS = {
    1: np.asarray([1.00, 0.62, 0.08], dtype=np.float64),
    2: np.asarray([0.05, 0.92, 1.00], dtype=np.float64),
    3: np.asarray([0.86, 0.22, 1.00], dtype=np.float64),
}

class _LazyOpen3D:
    def __getattr__(self, name):
        return getattr(require_open3d(), name)


o3d = _LazyOpen3D()


def guided_validator_primitives(result):
    """Resolve display-only colors and plane/sphere-intersection disks."""
    colors = np.full(result["roi_points_mm"].shape, 0.55)
    palette = np.eye(3)
    planes = result["planes"]
    roi = result["roi_points_mm"]
    disks = []
    if planes:
        residuals = np.column_stack([
            np.where(p["inlier_mask"], np.abs(roi @ p["normal"] + p["offset_mm"]), np.inf)
            for p in planes
        ])
        supported = np.isfinite(residuals).any(axis=1)
        colors[supported] = palette[
            np.argmin(residuals[supported], axis=1)]
        theta = np.linspace(0, 2 * np.pi, 97)
        for plane in planes:
            u, v, _ = make_tangent_basis(plane["normal"])
            signed_distance = plane["normal"] @ result["query_position_mm"] + plane["offset_mm"]
            center = result["query_position_mm"] - signed_distance * plane["normal"]
            disk_radius = np.sqrt(max(0., result["radius_mm"] ** 2 - signed_distance ** 2))
            disks.append(center + disk_radius * (
                np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v))
    return {"point_colors": colors, "disk_vertices_mm": disks}


def open_guided_validator_view(scene_points_mm, query_indices, scene_colors):
    """Upload Coarse once and keep Fine heatmap separate from plane fitting."""
    gui, rendering = o3d.visualization.gui, o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("VALIDATOR | 1. Coarse -> 2. Fine -> 3. Plane fitting", 1800, 650)
    overview, fine_view, local = gui.SceneWidget(), gui.SceneWidget(), gui.SceneWidget()
    panels = (overview, fine_view, local)
    labels = [gui.Label(text) for text in (
        "1. Coarse Fusion heatmap + NMS | click orange query",
        "2. Fine Fusion heatmap + NMS | select Coarse first",
        "3. Plane fitting | select Fine NMS first",
    )]
    for widget, label in zip(panels, labels):
        widget.scene = rendering.Open3DScene(window.renderer)
        widget.scene.set_background([0.08, 0.08, 0.08, 1.0])
        window.add_child(label)
        window.add_child(widget)

    def layout(_):
        rect = window.content_rect
        for index, (widget, label) in enumerate(zip(panels, labels)):
            left = rect.x + index * rect.width // 3
            right = rect.x + (index + 1) * rect.width // 3
            label.frame = gui.Rect(left, rect.y, right - left, 28)
            widget.frame = gui.Rect(left, rect.y + 28, right - left, max(1, rect.height - 28))

    window.set_on_layout(layout)
    scene_points = np.asarray(scene_points_mm)
    queries = scene_points[query_indices]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(scene_points)
    cloud.colors = o3d.utility.Vector3dVector(scene_colors)
    material = rendering.MaterialRecord()
    material.shader, material.point_size = "defaultUnlit", 2.0
    overview.scene.add_geometry("scene", cloud, material)
    if len(queries):
        markers = o3d.geometry.PointCloud()
        markers.points = o3d.utility.Vector3dVector(queries)
        markers.paint_uniform_color([1.0, 0.5, 0.0])
        query_material = rendering.MaterialRecord()
        query_material.shader, query_material.point_size = "defaultUnlit", 10.0
        overview.scene.add_geometry("queries", markers, query_material)
    bounds = cloud.get_axis_aligned_bounding_box()
    overview.setup_camera(60.0, bounds, bounds.get_center())
    display = {"app": app, "gui": gui, "window": window, "overview": overview,
               "fine_view": fine_view, "local": local, "labels": labels, "query_xyz": queries.copy(),
               "query_indices": np.asarray(query_indices).copy(), "closed": False}

    def closed():
        display["closed"] = True
        return True

    window.set_on_close(closed)
    return display


def _pick_validator_points(widget, points, indices, pixel_x, pixel_y):
    frame = widget.frame
    if not len(points) or frame.width <= 0 or frame.height <= 0:
        return None
    camera = widget.scene.camera
    homogeneous = np.column_stack((points, np.ones(len(points))))
    clip = homogeneous @ (camera.get_projection_matrix() @ camera.get_view_matrix()).T
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
    ndc = np.full((len(points), 3), np.nan)
    ndc[valid] = clip[valid, :3] / clip[valid, 3, None]
    valid &= (np.abs(ndc) <= 1).all(axis=1)
    screen = np.column_stack((frame.x + (ndc[:, 0] + 1) * frame.width / 2,
                              frame.y + (1 - ndc[:, 1]) * frame.height / 2))
    distance2 = np.sum((screen - [pixel_x, pixel_y]) ** 2, axis=1)
    hits = np.flatnonzero(valid & (distance2 <= 8 ** 2))
    if not len(hits):
        return None
    order = np.lexsort((indices[hits], ndc[hits, 2], distance2[hits]))
    return int(indices[hits[order[0]]])


def pick_guided_validator_query(display, pixel_x, pixel_y):
    return _pick_validator_points(display["overview"], display["query_xyz"],
                                  display["query_indices"], pixel_x, pixel_y)


def pick_validator_fine_query(display, pixel_x, pixel_y):
    fine = display.get("fine_result")
    if fine is None:
        return None
    return _pick_validator_points(display["fine_view"], fine["candidate_points_mm"],
                                  fine["candidate_fine_indices"], pixel_x, pixel_y)


def update_validator_fine_heatmap(display, result):
    """Read cached Fine Fusion scores; fitting changes only the selected marker."""
    rendering = o3d.visualization.rendering
    fine = result["fine_result"]
    widget = display["fine_view"]
    scene = widget.scene
    coarse = result["coarse_query_position_mm"]
    radius = fine["fine_search_radius_mm"]
    previous = display.get("fine_result")
    refresh = (previous is None or previous["coarse_query_index"] != fine["coarse_query_index"]
               or not result.get("planes_computed", True))
    if refresh:
        scene.clear_geometry()
        display["fine_selected_present"] = False
        scores = fine["scores"]
        vmax = dense_heatmap_vmax(scores)["fusion_score"]
        points = scores["query_position_mm"]
        if len(points):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(dense_heatmap_colors(
                scores["fusion_score"], scores["fusion_valid_mask"], vmax=vmax))
            material = rendering.MaterialRecord()
            material.shader, material.point_size = "defaultUnlit", 4.0
            scene.add_geometry("fine_heatmap", cloud, material)
        if len(fine["candidate_points_mm"]):
            markers = o3d.geometry.PointCloud()
            markers.points = o3d.utility.Vector3dVector(fine["candidate_points_mm"])
            markers.paint_uniform_color([1.0, 1.0, 0.0])
            material = rendering.MaterialRecord()
            material.shader, material.point_size = "defaultUnlit", 8.0
            scene.add_geometry("fine_nms", markers, material)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.4, radius * 0.025))
        marker.translate(coarse)
        material = rendering.MaterialRecord()
        material.shader, material.base_color = "defaultUnlit", [1.0, 0.5, 0.0, 1.0]
        scene.add_geometry("coarse_query", marker, material)
        bounds = o3d.geometry.AxisAlignedBoundingBox(coarse - radius, coarse + radius)
        widget.setup_camera(60.0, bounds, coarse)
        scene.camera.look_at(coarse, coarse + radius * np.array([1.7, 1.3, 1.5]), [0., 0., 1.])
        display["labels"][1].text = (f"2. Fine Fusion heatmap (Q99={vmax:.4g}) | "
                                    f"{len(fine['candidate_fine_indices'])} NMS candidates")
    display["fine_result"] = fine
    if display.get("fine_selected_present"):
        scene.remove_geometry("selected_fine")
    selected = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.5, radius * 0.04))
    selected.translate(result["query_position_mm"])
    material = rendering.MaterialRecord()
    material.shader, material.base_color = "defaultUnlit", [1.0, 0.0, 0.0, 1.0]
    scene.add_geometry("selected_fine", selected, material)
    display["fine_selected_present"] = True
    widget.force_redraw()


def update_guided_validator_view(display, result):
    """Replace only small ROI geometry; keep the overview and its camera intact."""
    rendering = o3d.visualization.rendering
    update_validator_fine_heatmap(display, result)
    widget = display["local"]
    scene = widget.scene
    scene.clear_geometry()
    if not result.get("planes_computed", True):
        display["labels"][2].text = "3. Plane fitting | click a Fine NMS candidate"
        widget.force_redraw()
        display["window"].post_redraw()
        return
    display["labels"][2].text = f"3. Plane fitting | {len(result['planes'])} planes"
    data = guided_validator_primitives(result)
    query, radius = result["query_position_mm"], result["radius_mm"]
    points = result["roi_points_mm"]
    if len(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(data["point_colors"])
        material = rendering.MaterialRecord()
        material.shader, material.point_size = "defaultUnlit", 4.0
        scene.add_geometry("roi", cloud, material)
    vertices, lines = _wire_sphere(query, radius, 64)
    wire = o3d.geometry.LineSet()
    wire.points = o3d.utility.Vector3dVector(vertices)
    wire.lines = o3d.utility.Vector2iVector(lines)
    wire.paint_uniform_color([0.85, 0.65, 0.0])
    material = rendering.MaterialRecord()
    material.shader, material.line_width = "unlitLine", 1.0
    scene.add_geometry("radius", wire, material)
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.5, radius * 0.035))
    marker.translate(query)
    marker.compute_vertex_normals()
    material = rendering.MaterialRecord()
    material.shader, material.base_color = "defaultUnlit", [1.0, 0.0, 0.0, 1.0]
    scene.add_geometry("fine_query", marker, material)
    coarse = result["coarse_query_position_mm"]
    coarse_marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.4, radius * 0.025))
    coarse_marker.translate(coarse)
    material = rendering.MaterialRecord()
    material.shader, material.base_color = "defaultUnlit", [1.0, 0.5, 0.0, 1.0]
    scene.add_geometry("coarse_query", coarse_marker, material)
    for i, (plane, disk) in enumerate(zip(result["planes"], data["disk_vertices_mm"])):
        color = np.eye(3)[i].tolist()
        mesh = o3d.geometry.TriangleMesh()
        center = query - (plane["normal"] @ query + plane["offset_mm"]) * plane["normal"]
        mesh.vertices = o3d.utility.Vector3dVector(np.vstack((center, disk)))
        mesh.triangles = o3d.utility.Vector3iVector(
            [[0, j, j + 1] for j in range(1, len(disk))])
        mesh.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color, material.has_alpha = color + [0.60], True
        material.emissive_color = (np.asarray(color) * 0.25).tolist() + [1.0]
        scene.add_geometry(f"plane_{i}", mesh, material)
        outline = o3d.geometry.LineSet()
        outline.points = o3d.utility.Vector3dVector(disk)
        segments = len(disk) - 1
        outline.lines = o3d.utility.Vector2iVector(
            [[j, j + 1] for j in range(segments)]
            + [[0, segments // 2], [segments // 4, 3 * segments // 4]])
        outline.paint_uniform_color(color)
        material = rendering.MaterialRecord()
        material.shader, material.line_width = "unlitLine", 3.0
        scene.add_geometry(f"outline_{i}", outline, material)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius * 0.012, cone_radius=radius * 0.025,
            cylinder_height=radius * 0.55, cone_height=radius * 0.15)
        u, v, n = make_tangent_basis(plane["normal"])
        arrow.rotate(np.column_stack((u, v, n)), center=np.zeros(3))
        arrow.translate(center)
        arrow.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader, material.base_color = "defaultUnlit", color + [1.0]
        scene.add_geometry(f"normal_{i}", arrow, material)
    bounds_points = np.vstack((query - radius, query + radius, coarse))
    bounds = o3d.geometry.AxisAlignedBoundingBox(bounds_points.min(axis=0), bounds_points.max(axis=0))
    widget.setup_camera(60.0, bounds, query)
    scene.camera.look_at(query, query + radius * np.array([1.7, 1.3, 1.5]), [0.0, 0.0, 1.0])
    widget.force_redraw()
    display["window"].post_redraw()


def raw_vote_heatmap(votes):
    """Zero is gray; positive raw mass uses a linear 0..maximum color range."""
    values = np.asarray(votes, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raw votes must be finite and nonnegative")
    positive = values > 0
    colors = np.tile([0.28, 0.28, 0.30], (len(values), 1))
    low = float(values[positive].min()) if positive.any() else 0.0
    high = float(values.max(initial=0))
    if positive.any():
        level = values[positive] / high
        colors[positive] = centroid_saliency_colors(level)
    return colors, {"min_positive": low, "max": high,
                    "zero_fraction": float(np.mean(~positive)) if len(values) else 0.0}


def extremity_primitives(points_mm, result, *, color_mode="heatmap"):
    """Read-only copies for live views and saved experiments."""
    points = np.asarray(points_mm, dtype=float)
    ids = np.asarray(result["roi_raw_indices"])
    labels = np.asarray(result["cluster_labels"])
    colors = np.tile([0.16, 0.17, 0.19], (len(points), 1))
    raw_display = None
    if color_mode == "clusters":
        colors[ids] = matplotlib.colormaps["tab20"](labels % 20 / 19.0)[:, :3]
    elif color_mode == "raw_heatmap":
        colors[ids], raw_display = raw_vote_heatmap(result["raw_votes"])
    elif color_mode == "nms":
        colors[ids] = [0.42, 0.43, 0.46]
    elif color_mode == "heatmap":
        scores = np.asarray(result["normalized_score"])
        colors[ids] = dense_heatmap_colors(scores, scores > 0, vmax=max(float(scores.max(initial=0)), 0.01))
    else:
        raise ValueError(f"Unknown extremity color mode: {color_mode}")
    return {"points_mm": points.copy(), "point_colors": colors.copy(),
            "query_points_mm": np.asarray(result["query_position_mm"]).reshape(1, 3).copy(),
            "candidate_points_mm": points[result["candidate_raw_indices"]].copy(),
            "raw_vote_display": raw_display}


def open_extremity_view(points_mm, result, previous=None, *, color_mode="heatmap", left=10):
    if previous is not None:
        previous.destroy_window()
    data = extremity_primitives(points_mm, result, color_mode=color_mode)
    view = o3d.visualization.Visualizer()
    titles = {
        "clusters": f"BALL 1/3 | object components ({result['cluster_count']})",
        "raw_heatmap": "BALL 2/3 | RAW votes",
        "nms": f"BALL 3/3 | NMS hypotheses ({result['candidate_count']}) | magenta: unverified",
        "heatmap": "BALL | normalized votes",
    }
    title = titles[color_mode] + " | red: original query"
    if color_mode == "raw_heatmap":
        display = data["raw_vote_display"]
        title = (f"BALL 2/3 | RAW votes LINEAR [0, {display['max']:.3g}] "
                 f"| zero(gray)={display['zero_fraction']:.1%}")
    if not view.create_window(window_name=title, width=500, height=720, left=left, top=45):
        view.destroy_window()
        raise RuntimeError("Failed to create ball 3-D view")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data["points_mm"])
    cloud.colors = o3d.utility.Vector3dVector(data["point_colors"])
    view.add_geometry(cloud)
    for positions, radius, color in (
        (data["candidate_points_mm"] if color_mode == "nms" else (), 3.0, [1.0, 0.0, 1.0]),
        (np.asarray(result.get("gt_corners_mm", [])).reshape(-1, 3)
         if color_mode == "nms" else (), 1.1, [0.0, 1.0, 0.0]),
    ):
        for position in positions:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            marker.translate(position)
            marker.paint_uniform_color(color)
            marker.compute_vertex_normals()
            view.add_geometry(marker)
    if "query_position_mm" in result:
        center = result["query_position_mm"]
        for radius, color in ((max(result["config"].radii_mm), [1.0, 1.0, 0.0]),
                              (max(result["config"].radii_mm) + result["config"].padding_mm,
                               [0.6, 0.6, 0.6])):
            xyz, lines = _wire_sphere(center, radius, 96)
            sphere = o3d.geometry.LineSet()
            sphere.points = o3d.utility.Vector3dVector(xyz)
            sphere.lines = o3d.utility.Vector2iVector(lines)
            sphere.paint_uniform_color(color)
            view.add_geometry(sphere)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=2.4)
        marker.translate(center)
        marker.paint_uniform_color([1.0, 0.0, 0.0])
        marker.compute_vertex_normals()
        view.add_geometry(marker)
    view.get_render_option().point_size = 3.0
    view.get_render_option().background_color = np.asarray([0.06, 0.07, 0.09])
    return view


def open_extremity_views(points_mm, result, previous=()):
    """Open component, pre-NMS raw-vote and final peak views side by side."""
    for view in previous:
        if view is not None:
            view.destroy_window()
    views = []
    try:
        for index, mode in enumerate(("clusters", "raw_heatmap", "nms")):
            views.append(open_extremity_view(points_mm, result, color_mode=mode, left=10 + index * 510))
    except Exception:
        for view in views:
            view.destroy_window()
        raise
    return tuple(views)


def open_extremity_overview(raw_points_mm, scene_points_mm, batch):
    """Overlay cached Ball hypotheses on the original detector NMS heatmap."""
    raw = np.asarray(raw_points_mm)
    scene = np.asarray(scene_points_mm)
    records = batch["records"]
    # Only identical raw IDs share a display marker; nearby vertices stay separate.
    candidate_ids = (np.unique(np.concatenate([r["candidate_raw_indices"] for r in records]))
                     if records else np.empty(0, dtype=np.int64))
    queries = scene[batch["query_indices"]]
    scores = batch["ced_score_result"]
    vmax = dense_heatmap_vmax(scores)["fusion_score"]
    view = o3d.visualization.Visualizer()
    title = (f"BALL ALL | CED -> PCA -> NMS -> Ball | queries={len(queries)} "
             f"raw peaks={len(candidate_ids)} | Fusion Q99={vmax:.4g} "
             "| red: CED query | magenta: unverified hypothesis")
    if not view.create_window(window_name=title, width=900, height=650, left=10, top=90):
        view.destroy_window()
        raise RuntimeError("Failed to create ball overview")
    try:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(scene.copy())
        cloud.colors = o3d.utility.Vector3dVector(dense_heatmap_colors(
            scores["fusion_score"], scores["fusion_valid_mask"], vmax=vmax))
        view.add_geometry(cloud)
        for positions, radius, color in (
            (queries, 2.4, [1.0, 0.0, 0.0]),
            (raw[candidate_ids], 3.0, [1.0, 0.0, 1.0]),
        ):
            if not len(positions):
                continue
            markers = o3d.geometry.TriangleMesh()
            for position in positions:
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
                marker.translate(position)
                markers += marker
            markers.paint_uniform_color(color)
            markers.compute_vertex_normals()
            view.add_geometry(markers)
        view.get_render_option().background_color = np.asarray([0.06, 0.07, 0.09])
        return view
    except Exception:
        view.destroy_window()
        raise


def make_extremity_direction_figure(result, directions):
    """Rotatable raw-ROI picker and selected-point unit-sphere diagnostic."""
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = Figure(figsize=(12, 6), constrained_layout=True)
    roi_ax = fig.add_subplot(121, projection="3d")
    direction_ax = fig.add_subplot(122, projection="3d")
    xyz = result["roi_points_mm"] - result["query_position_mm"]
    colors, _ = raw_vote_heatmap(result["raw_votes"])
    roi_ax.scatter(*xyz.T, c=colors, s=12, depthshade=False)
    roi_ax.scatter(0, 0, 0, c="red", marker="+", s=100)
    candidates = xyz[result["candidate_local_indices"]]
    if len(candidates):
        roi_ax.scatter(*candidates.T, c="magenta", s=45, depthshade=False)
    limit = max(result["config"].radii_mm) + result["config"].padding_mm
    # Some Matplotlib versions cannot autoscale an empty 3-D line collection.
    initial_segment = np.zeros((1, 2, 3))
    core_wire = Line3DCollection(initial_segment, colors="#d5a500", linewidths=0.6)
    search_wire = Line3DCollection(initial_segment, colors="gray", linewidths=0.4)
    for wire in (core_wire, search_wire):
        roi_ax.add_collection3d(wire)
    roi_ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
               xlabel="X - query (mm)", ylabel="Y - query (mm)", zlabel="Z - query (mm)",
               title=f"Query {result['query_index']} | click any raw ROI point")
    roi_ax.set_box_aspect((1, 1, 1))
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    direction_ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                                color="gray", linewidth=0.4, alpha=0.25)
    background = direction_ax.scatter(*directions.T, c="gray", s=12, alpha=0.4, depthshade=False)
    rays = Line3DCollection(initial_segment, colors="#e49b00", linewidths=0.7, alpha=0.6)
    direction_ax.add_collection3d(rays)
    rays.set_segments([])
    direction_ax.scatter(0, 0, 0, c="black", s=18)
    direction_ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1),
                     xlabel="direction X", ylabel="direction Y", zlabel="direction Z",
                     title=f"{len(directions)} directions | select a point on the left")
    direction_ax.set_box_aspect((1, 1, 1))
    mapper = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="turbo")
    colorbar = fig.colorbar(mapper, ax=direction_ax, shrink=0.6, pad=0.12)
    colorbar.set_label("Raw vote / direction\nlinear, per-point range")
    colorbar.ax.set_visible(False)
    display = {"figure": fig, "roi_axes": roi_ax, "direction_axes": direction_ax,
               "roi_xyz": xyz, "raw_indices": result["roi_raw_indices"].copy(),
               "core_wire": core_wire, "search_wire": search_wire, "rays": rays,
               "selected_marker": None, "direction_markers": None, "diagnostic": None,
               "direction_background": background, "color_mapper": mapper, "colorbar": colorbar,
               "sphere_mode": "contribution", "direction_display_range": (0., 0.)}
    update_extremity_direction_scale(display, max(result["config"].radii_mm), result["config"].padding_mm)
    return display


def pick_extremity_roi_point(display, pixel_x, pixel_y):
    """Map the current 3-D view's nearest screen sample back to its raw-ROI index."""
    from mpl_toolkits.mplot3d import proj3d

    xyz = display["roi_xyz"]
    if not len(xyz):
        return None
    ax = display["roi_axes"]
    x, y, z = proj3d.proj_transform(*xyz.T, ax.get_proj())
    screen = ax.transData.transform(np.column_stack((x, y)))
    distance2 = np.sum((screen - [pixel_x, pixel_y]) ** 2, axis=1)
    local = np.flatnonzero(np.isfinite(distance2) & (distance2 <= 8 ** 2))
    if not len(local):
        return None
    order = np.lexsort((display["raw_indices"][local], z[local], distance2[local]))
    return int(local[order[0]])


def update_extremity_direction_scale(display, radius_mm, padding_mm):
    for radius, key in ((radius_mm, "core_wire"), (radius_mm + padding_mm, "search_wire")):
        vertices, lines = _wire_sphere(np.zeros(3), radius, 48)
        display[key].set_segments(vertices[lines])


def update_extremity_direction_figure(display, diagnostic, result):
    """Update markers without clearing either interactive camera orientation."""
    point = diagnostic["position_mm"] - result["query_position_mm"]
    if display["selected_marker"] is not None:
        display["selected_marker"].remove()
    display["selected_marker"] = display["roi_axes"].scatter(
        *point, s=140, facecolors="none", edgecolors="#00bcd4", linewidths=2, depthshade=False)
    if display["direction_markers"] is not None:
        display["direction_markers"].remove()
    heatmap = display["sphere_mode"] == "contribution"
    display["direction_background"].set_visible(not heatmap)
    if heatmap:
        colors, stats = raw_vote_heatmap(diagnostic["direction_votes"])
        display["direction_markers"] = display["direction_axes"].scatter(
            *diagnostic["directions"].T, c=colors, s=38, edgecolors="black", linewidths=0.2, depthshade=False)
        high = stats["max"]
        display["direction_display_range"] = (0., high)
        display["color_mapper"].set_norm(matplotlib.colors.Normalize(0, high if high > 0 else 1))
        display["colorbar"].update_normal(display["color_mapper"])
        display["colorbar"].ax.set_visible(high > 0)
        display["rays"].set_segments([])
        summary = f"Vote mass: sum={diagnostic['direction_vote_sum']:.4g} | max={high:.3g}"
    else:
        selected = diagnostic["directions"][diagnostic["top1_mask"]]
        display["direction_markers"] = display["direction_axes"].scatter(
            *selected.T, c="#e49b00", s=42, edgecolors="black", linewidths=0.3, depthshade=False)
        display["rays"].set_segments(np.stack((np.zeros_like(selected), selected), axis=1))
        display["colorbar"].ax.set_visible(False)
        summary = f"Top 1%: {diagnostic['top1_count']}/{len(diagnostic['directions'])} directions"
    update_extremity_direction_scale(display, diagnostic["radius_mm"], diagnostic["padding_mm"])
    region = ("candidate core" if diagnostic["inside_core"] else
              "padding: no output vote" if diagnostic["inside_search"] else "outside selected search ROI")
    display["direction_axes"].set_title(
        f"Raw {diagnostic['raw_point_index']} | component {diagnostic['cluster_id']}\n"
        f"{summary} "
        f"| r={diagnostic['radius_mm']:g} mm\n{region}")
    display["diagnostic"] = diagnostic


def save_extremity_figure(points_mm, result, path, *, title="Extremity voting"):
    """Export four actual XYZ views; no projection enters the detector."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    points = np.asarray(points_mm)
    fig = Figure(figsize=(14, 10), constrained_layout=True)
    FigureCanvasAgg(fig)
    gt = np.asarray(result.get("gt_corners_mm", [])).reshape(-1, 3)
    candidates = points[result["candidate_raw_indices"]]
    queries = np.asarray(result["query_position_mm"]).reshape(1, 3)
    for panel, (label, mode) in enumerate((("Observed XYZ + GT", None),
                                          ("Local object components + unchanged query", "clusters"),
                                          ("Normalized voting heatmap", "heatmap"),
                                          ("NMS candidates + GT", None))):
        ax = fig.add_subplot(2, 2, panel + 1, projection="3d")
        colors = ("#a9b3bd" if mode is None else
                  extremity_primitives(points, result, color_mode=mode)["point_colors"])
        ax.scatter(*points.T, c=colors, s=2, alpha=0.6)
        if panel == 1 and len(queries):
            ax.scatter(*queries.T, c="red", s=50, marker="+", label="Original query")
        if panel in (0, 3) and len(gt):
            ax.scatter(*gt.T, c="#00a43b", s=65, marker="x", label="GT corner")
        if panel == 3 and len(candidates):
            ax.scatter(*candidates.T, facecolors="none", edgecolors="#df00ba", s=70,
                       label="Unverified candidate")
        ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=label)
        span = max(np.ptp(points, axis=0).max(), 1.0)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
        if panel in (0, 1, 3):
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    fig.clear()


def save_extremity_html(points_mm, result, path, *, title="Extremity voting"):
    """Self-contained rotatable artifact; Plotly is only needed for this export."""
    import plotly.graph_objects as go
    points = np.asarray(points_mm)
    data = extremity_primitives(points, result)
    rgb = (data["point_colors"] * 255).astype(int)
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    hover = np.zeros((len(points), 5))
    hover[:, 0] = np.arange(len(points))
    hover[:, 3] = -1
    ids = result["roi_raw_indices"]
    hover[ids, 1:] = np.column_stack((result["raw_votes"], result["normalized_score"],
                                     result["cluster_labels"], result["scale_support_count"]))
    fig = go.Figure(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
        name="XYZ / normalized votes", marker=dict(size=2, color=colors), customdata=hover,
        hovertemplate="index=%{customdata[0]} votes=%{customdata[1]}<br>fraction=%{customdata[2]:.4f}"
                      " component=%{customdata[3]} scale-support=%{customdata[4]}<extra></extra>"))
    for name, xyz, color, symbol, size in (
        ("Original query", data["query_points_mm"], "red", "cross", 6),
        ("NMS candidates (unverified)", data["candidate_points_mm"], "magenta", "diamond", 6),
        ("GT corners", np.asarray(result.get("gt_corners_mm", [])).reshape(-1, 3), "limegreen", "x", 6),
    ):
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name,
                                  marker=dict(size=size, color=color, symbol=symbol)))
    cluster_colors = ["rgb(%d,%d,%d)" % tuple(c) for c in
                      (extremity_primitives(points, result, color_mode="clusters")["point_colors"] * 255).astype(int)]
    fig.update_layout(title=title, scene=dict(aspectmode="data", xaxis_title="X (mm)",
                                             yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        updatemenus=[dict(buttons=[dict(label="Vote heatmap", method="restyle", args=[{"marker.color": [colors]}, [0]]),
                                  dict(label="Components", method="restyle", args=[{"marker.color": [cluster_colors]}, [0]])])])
    fig.write_html(str(path), include_plotlyjs=True)


def save_soft_vote_comparison(points_mm, results, gt_mm, path, *, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm

    points = np.asarray(points_mm)
    gt = np.asarray(gt_mm).reshape(-1, 3)
    figure = Figure(figsize=(15, 11), constrained_layout=True)
    FigureCanvasAgg(figure)
    center = next(iter(results.values()))["query_position_mm"]
    for index, (label, result) in enumerate(results.items()):
        ax = figure.add_subplot(2, 2, index + 1, projection="3d")
        ids = result["roi_raw_indices"]
        roi = points[ids]
        votes = result["raw_votes"]
        positive = votes > 0
        ax.scatter(*points.T, c="#c2c4c8", s=2, alpha=0.3)
        if positive.any():
            low, high = float(votes[positive].min()), float(votes.max())
            cloud = ax.scatter(*roi[positive].T, c=votes[positive], s=12, cmap="turbo",
                               norm=LogNorm(vmin=low, vmax=high if high > low else low * 1.01))
            figure.colorbar(cloud, ax=ax, shrink=0.55, pad=0.08, label="Raw vote mass (log)")
        candidates = points[result["candidate_raw_indices"]]
        if len(candidates):
            ax.scatter(*candidates.T, facecolors="none", edgecolors="magenta", s=100, label="NMS hypothesis")
        local_gt = gt[np.linalg.norm(gt - center, axis=1) <= 30]
        if len(local_gt):
            ax.scatter(*local_gt.T, c="green", marker="x", s=80, label="GT")
        ax.scatter(*center, c="red", marker="+", s=60, label="Unchanged query")
        ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
               title=f"{label} | NMS={len(candidates)}, zero votes={np.mean(~positive):.1%}")
        ax.set_xlim(center[0] - 42, center[0] + 42)
        ax.set_ylim(center[1] - 42, center[1] + 42)
        ax.set_zlim(center[2] - 42, center[2] + 42)
        ax.legend(fontsize=8, loc="upper left")
    figure.suptitle(title + "\nGray: zero/outside support; positive votes use per-panel log scales")
    figure.savefig(path, dpi=140)
    figure.clear()


def centroid_saliency_colors(scores):
    """Map dimensionless CED scores to a fixed 0..1 turbo heatmap."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("CED scores must be finite")
    return matplotlib.colormaps["turbo"](np.clip(values, 0.0, 1.0))[:, :3]


def dense_heatmap_colors(scores, valid_mask, *, vmax=1.0):
    """Map ``0..vmax`` to Turbo and reserve gray for invalid voxel points."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    upper = float(vmax)
    if len(values) != len(valid):
        raise ValueError("scores and valid_mask must have the same length")
    if not np.isfinite(values).all():
        raise ValueError("Dense heatmap scores must be finite")
    if not np.isfinite(upper) or upper < 0.0:
        raise ValueError("vmax must be finite and non-negative")
    normalized = (
        np.zeros_like(values)
        if upper == 0.0
        else np.clip(values / upper, 0.0, 1.0)
    )
    colors = centroid_saliency_colors(normalized)
    colors[~valid] = [0.28, 0.28, 0.30]
    return colors


def pca_ratio_heatmap_colors(score_result, *, vmax=1.0):
    """Color the PCA ratio actually consumed by fusion.

    ``pca_ratio_saturated`` is identical to the raw ratio when tau is None,
    and contains the tanh-transformed ratio when saturation is configured.
    """
    ratio = np.asarray(
        score_result["pca_ratio_saturated"], dtype=np.float64,
    ).reshape(-1)
    valid = np.asarray(score_result["pca_valid_mask"], dtype=bool).reshape(-1)
    queried = np.asarray(score_result["pca_query_mask"], dtype=bool).reshape(-1)
    invalid = np.asarray(
        score_result["invalid_support_mask"], dtype=bool,
    ).reshape(-1)
    for name, mask in (
        ("pca_valid_mask", valid),
        ("pca_query_mask", queried),
        ("invalid_support_mask", invalid),
    ):
        if len(mask) != len(ratio):
            raise ValueError(f"{name} must align with pca_ratio_saturated")
    colors = dense_heatmap_colors(ratio, valid, vmax=vmax)
    colors[~queried] = [0.16, 0.16, 0.18]
    colors[invalid] = [0.58, 0.58, 0.62]
    return colors


def dense_heatmap_vmax(score_result, quantile=0.99):
    """Return Q-based display maxima without modifying numerical scores.

    Original CED defines the shared range for point-wise and Bilateral CED. The
    dimensionless PCA ratio and Fusion score each receive their own range.
    Invalid samples are excluded from quantiles.
    """
    q = float(quantile)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be finite and in [0, 1]")

    def valid_quantile(value_key, mask_key):
        values = np.asarray(score_result[value_key], dtype=np.float64).reshape(-1)
        valid = np.asarray(score_result[mask_key], dtype=bool).reshape(-1)
        if len(values) != len(valid):
            raise ValueError(f"{value_key} and {mask_key} must have the same length")
        if not np.isfinite(values).all():
            raise ValueError(f"{value_key} must be finite")
        selected = values[valid]
        return 0.0 if len(selected) == 0 else float(np.quantile(selected, q))

    ced_shared = valid_quantile("ced_score", "ced_valid_mask")
    pca = valid_quantile("pca_ratio_saturated", "pca_valid_mask")
    fusion = valid_quantile("fusion_score", "fusion_valid_mask")
    return {
        "ced_score": ced_shared,
        "bilateral_ced_score": ced_shared,
        "pca_ratio_saturated": pca,
        "fusion_score": fusion,
    }


def corner_candidate_primitives(points_mm, score_result):
    """Return immutable marker/debug values for extracted 3-D candidates."""
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    extraction = score_result.get("candidate_extraction")
    if extraction is None:
        return {
            "indices": np.empty(0, dtype=np.int64),
            "positions_mm": np.empty((0, 3), dtype=np.float64),
            "ced_score": np.empty(0, dtype=np.float64),
            "bilateral_ced_score": np.empty(0, dtype=np.float64),
            "pca_ratio_raw": np.empty(0, dtype=np.float64),
            "pca_ratio_saturated": np.empty(0, dtype=np.float64),
            "pca_ratio_saturation_tau": score_result.get(
                "pca_ratio_saturation_tau"
            ),
            "fusion_score": np.empty(0, dtype=np.float64),
            "nms_radius_mm": 0.0,
            "candidate_count": 0,
        }
    indices = np.asarray(extraction["candidate_indices"], dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise IndexError("candidate_indices contains an out-of-range point index")
    ced = np.asarray(score_result["ced_score"], dtype=np.float64)
    bilateral = np.asarray(
        score_result.get("bilateral_ced_score", ced), dtype=np.float64,
    )
    pca_raw = np.asarray(score_result["pca_ratio_raw"], dtype=np.float64)
    pca_saturated = np.asarray(
        score_result["pca_ratio_saturated"], dtype=np.float64,
    )
    fusion_score = np.asarray(score_result["fusion_score"], dtype=np.float64)
    return {
        "indices": indices.copy(),
        "positions_mm": points[indices].copy(),
        "ced_score": ced[indices].copy(),
        "bilateral_ced_score": bilateral[indices].copy(),
        "pca_ratio_raw": pca_raw[indices].copy(),
        "pca_ratio_saturated": pca_saturated[indices].copy(),
        "pca_ratio_saturation_tau": score_result.get(
            "pca_ratio_saturation_tau"
        ),
        "fusion_score": fusion_score[indices].copy(),
        "nms_radius_mm": float(extraction["nms_radius_mm"]),
        "candidate_count": int(extraction["candidate_count"]),
    }


def _wire_sphere(center, radius, samples):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    radius = float(radius)
    samples = int(samples)
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Wire-sphere center/radius must be finite and radius positive")
    if samples < 8:
        raise ValueError("Wire-sphere samples must be at least 8")
    angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    cosine, sine = radius * np.cos(angle), radius * np.sin(angle)
    circles = (
        center + np.column_stack((cosine, sine, np.zeros(samples))),
        center + np.column_stack((cosine, np.zeros(samples), sine)),
        center + np.column_stack((np.zeros(samples), cosine, sine)),
    )
    sphere_points = np.vstack(circles)
    sphere_lines = []
    for circle_index in range(3):
        offset = circle_index * samples
        sphere_lines.extend(
            (offset + index, offset + (index + 1) % samples)
            for index in range(samples)
        )
    return sphere_points, np.asarray(sphere_lines, dtype=np.int32)


def cornervins_ahc_primitives(
    organized_points_mm,
    analysis,
    *,
    stage="coarse",
    circle_samples=96,
):
    """Build one organized AHC/RANSAC stage and the exact query ROI."""

    organized = np.asarray(organized_points_mm, dtype=np.float64)
    if organized.ndim != 3 or organized.shape[2] != 3:
        raise ValueError("organized_points_mm must have shape (H,W,3)")
    query = np.asarray(analysis["query_position_mm"], dtype=np.float64).reshape(3)
    radius = float(
        analysis.get(
            "organized_extraction_radius_mm",
            analysis["query_radius_mm"],
        )
    )
    if not np.isfinite(query).all() or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("CornerVINS query and radius must be finite and positive")

    valid = np.isfinite(organized).all(axis=2).reshape(-1)
    raw_pixels = np.flatnonzero(valid)
    raw = organized.reshape(-1, 3)[raw_pixels]
    lookup = np.full(valid.shape, -1, dtype=np.int64)
    lookup[raw_pixels] = np.arange(len(raw_pixels), dtype=np.int64)
    inside_roi = np.linalg.norm(raw - query, axis=1) <= radius + 1e-9
    colors = np.tile([0.14, 0.15, 0.17], (len(raw), 1))
    colors[inside_roi] = [0.42, 0.43, 0.46]
    palette = matplotlib.colormaps["tab20"]
    result = analysis["cornervins_result"]
    if stage == "coarse":
        groups = tuple(result.ahc_regions)
        patches = ()
    elif stage == "refined":
        membership = np.asarray(result.refined_membership_image, dtype=np.int32)
        if membership.shape != organized.shape[:2]:
            raise ValueError("refined AHC membership must match organized image")
        labels = np.unique(membership[membership >= 0])
        groups = tuple(
            np.flatnonzero(membership.reshape(-1) == label).astype(np.int64)
            for label in labels
        )
        patches = ()
    elif stage == "ransac":
        patches = tuple(getattr(result, "ransac_patches", ()))
        groups = tuple(patch.organized_inlier_indices for patch in patches)
    else:
        raise ValueError("stage must be coarse, refined, or ransac")

    for region_id, pixels in enumerate(groups):
        pixel_ids = np.asarray(pixels, dtype=np.int64).reshape(-1)
        if np.any(pixel_ids < 0) or np.any(pixel_ids >= valid.size):
            raise IndexError("organized stage contains an invalid pixel index")
        local = lookup[pixel_ids]
        local = local[local >= 0]
        colors[local] = palette((region_id % 20) / 19.0)[:3]
    sphere_points, sphere_lines = _wire_sphere(
        query, radius, int(circle_samples)
    )
    return {
        "points_mm": raw.copy(),
        "point_colors": colors,
        "inside_roi_mask": inside_roi,
        "query_position_mm": query.copy(),
        "query_radius_mm": radius,
        "sphere_points_mm": sphere_points,
        "sphere_lines": sphere_lines,
        "region_count": len(groups),
        "stage": stage,
        "plane_centers_mm": np.asarray(
            [patch.center for patch in patches], dtype=np.float64
        ).reshape(-1, 3),
        "plane_normals": np.asarray(
            [patch.normal for patch in patches], dtype=np.float64
        ).reshape(-1, 3),
    }


def cornervins_scene_roi_primitives(
    points_mm,
    analysis,
    *,
    circle_samples=96,
):
    """Build full-scene context colors and the exact CornerVINS query ROI."""

    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N,3)")
    roi_ids = np.asarray(analysis["roi_point_indices"], dtype=np.int64).reshape(-1)
    if np.any(roi_ids < 0) or np.any(roi_ids >= len(points)):
        raise IndexError("CornerVINS ROI contains an invalid scene index")
    query = np.asarray(analysis["query_position_mm"], dtype=np.float64).reshape(3)
    radius = float(analysis["query_radius_mm"])
    if not np.isfinite(query).all() or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("CornerVINS query and radius must be finite and positive")
    colors = np.tile([0.14, 0.15, 0.17], (len(points), 1))
    colors[roi_ids] = [0.42, 0.43, 0.46]
    sphere_points, sphere_lines = _wire_sphere(
        query, radius, int(circle_samples)
    )
    return {
        "points_mm": points.copy(),
        "point_colors": colors,
        "roi_point_indices": roi_ids.copy(),
        "query_position_mm": query.copy(),
        "query_radius_mm": radius,
        "sphere_points_mm": sphere_points,
        "sphere_lines": sphere_lines,
    }


def _camera_signature(parameters):
    """Return the numeric camera state used to detect the manipulated view."""
    intrinsic = np.asarray(
        parameters.intrinsic.intrinsic_matrix,
        dtype=np.float64,
    ).reshape(-1)
    extrinsic = np.asarray(parameters.extrinsic, dtype=np.float64).reshape(-1)
    return np.concatenate((intrinsic, extrinsic))


def synchronize_dense_camera_views(windows, active, previous_signatures):
    """Propagate the camera that changed since the previous render cycle.

    Returning ``master_index=None`` means no user-visible camera change was
    detected. This lets any dense-score or RGB window become the master,
    rather than continuously forcing the first window onto the others.
    """
    if len(windows) != len(active):
        raise ValueError("windows and active must have the same length")
    if previous_signatures is not None and len(previous_signatures) != len(windows):
        raise ValueError("previous_signatures must align with windows")

    parameters = [None] * len(windows)
    signatures = [None] * len(windows)
    for index, vis in enumerate(windows):
        if not active[index]:
            continue
        parameters[index] = (
            vis.get_view_control().convert_to_pinhole_camera_parameters()
        )
        signatures[index] = _camera_signature(parameters[index])

    if previous_signatures is None:
        return signatures, None
    changed = [
        index
        for index, signature in enumerate(signatures)
        if signature is not None
        and previous_signatures[index] is not None
        and not np.allclose(
            signature,
            previous_signatures[index],
            rtol=1e-8,
            atol=1e-10,
        )
    ]
    if not changed:
        return signatures, None

    # Normal interaction changes one window per event cycle. If multiple
    # windows changed, prefer the later panel deterministically.
    master_index = changed[-1]
    master_parameters = parameters[master_index]
    master_signature = signatures[master_index]
    for index, vis in enumerate(windows):
        if not active[index] or index == master_index:
            continue
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            master_parameters,
            allow_arbitrary=True,
        )
        signatures[index] = master_signature.copy()
    return signatures, master_index


def dense_heatmap_panels():
    """Return the seven synchronized whole-cloud panel specifications."""
    return (
        ("Original Point Cloud", None, None, "geometry"),
        ("Point-wise CED", "ced_score", "ced_valid_mask", "CED range"),
        (
            "Bilateral CED",
            "bilateral_ced_score",
            "bilateral_valid_mask",
            "CED range",
        ),
        (
            "PCA Ratio Used for Fusion",
            "pca_ratio_saturated",
            "pca_valid_mask",
            "PCA range",
        ),
        (
            "Final Fusion Score",
            "fusion_score",
            "fusion_valid_mask",
            "Fusion Q99",
        ),
        (
            "NMS Corner Candidates",
            "fusion_score",
            "fusion_valid_mask",
            "Fusion Q99",
        ),
    )


def show_dense_score_heatmaps(
    points_mm,
    score_result,
    *,
    marker_radius_mm=1.0,
    rgb_points_mm=None,
    rgb_colors_rgb=None,
    only_nms=False,
):
    """Show score fields and optional registered RGB in synchronized views."""
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if only_nms:
        rgb_points_mm = rgb_colors_rgb = None
    if (rgb_points_mm is None) != (rgb_colors_rgb is None):
        raise ValueError("rgb_points_mm and rgb_colors_rgb must be provided together")
    rgb_points = None
    rgb_colors = None
    if rgb_points_mm is not None:
        rgb_points = np.asarray(rgb_points_mm, dtype=np.float64)
        raw_rgb_colors = np.asarray(rgb_colors_rgb)
        if (
            rgb_points.ndim != 2
            or rgb_points.shape[1] != 3
            or not np.isfinite(rgb_points).all()
        ):
            raise ValueError("rgb_points_mm must be finite with shape (N, 3)")
        if raw_rgb_colors.shape != (len(rgb_points), 3):
            raise ValueError("rgb_colors_rgb must align with rgb_points_mm")
        integer_rgb = np.issubdtype(raw_rgb_colors.dtype, np.integer)
        rgb_colors = np.asarray(raw_rgb_colors, dtype=np.float64)
        if not np.isfinite(rgb_colors).all():
            raise ValueError("rgb_colors_rgb must be finite")
        if integer_rgb or (rgb_colors.size and np.max(rgb_colors) > 1.0):
            rgb_colors = rgb_colors / 255.0
        if np.any(rgb_colors < 0.0) or np.any(rgb_colors > 1.0):
            raise ValueError("rgb_colors_rgb values must be in [0, 1] or [0, 255]")
    display_vmax = dense_heatmap_vmax(score_result, quantile=0.99)
    candidate_debug = corner_candidate_primitives(points, score_result)
    panels = dense_heatmap_panels()
    panel_indices = (5,) if only_nms else range(len(panels))
    windows = []
    panel_count = len(panel_indices) + int(rgb_points is not None)

    def window_layout(index):
        if panel_count == 4:
            return {
                "width": 820,
                "height": 470,
                "left": 15 + 840 * (index % 2),
                "top": 35 + 490 * (index // 2),
            }
        if 7 <= panel_count <= 8:
            return {
                "width": 460,
                "height": 450,
                "left": 15 + 475 * (index % 4),
                "top": 35 + 470 * (index // 4),
            }
        if 5 <= panel_count <= 6:
            return {
                "width": 600,
                "height": 450,
                "left": 15 + 620 * (index % 3),
                "top": 35 + 470 * (index // 3),
            }
        return {
            "width": 600,
            "height": 760,
            "left": 15 + 620 * index,
            "top": 45,
        }

    def build_candidate_markers():
        if not len(candidate_debug["indices"]):
            return None
        markers = o3d.geometry.TriangleMesh()
        for position in candidate_debug["positions_mm"]:
            marker = o3d.geometry.TriangleMesh.create_sphere(
                radius=max(0.4, float(marker_radius_mm)),
                resolution=6,
            )
            marker.translate(position)
            marker.paint_uniform_color([1.0, 0.05, 0.03])
            markers += marker
        markers.compute_vertex_normals()
        return markers

    for window_index, panel_index in enumerate(panel_indices):
        label, key, valid_key, range_label = panels[panel_index]
        values = (
            np.zeros(len(points), dtype=np.float64)
            if key is None
            else np.asarray(score_result[key], dtype=np.float64).reshape(-1)
        )
        if len(values) != len(points):
            raise ValueError(f"{key} must align with points_mm")
        valid = (
            np.ones(len(points), dtype=bool)
            if valid_key is None
            else np.asarray(score_result[valid_key], dtype=bool).reshape(-1)
        )
        if len(valid) != len(points):
            raise ValueError(f"{valid_key} must align with points_mm")
        selected = values[valid]
        data_range = (
            "none"
            if len(selected) == 0
            else f"{np.min(selected):.3f}..{np.max(selected):.3f}"
        )
        scale_text = (
            f"Q99={display_vmax[key]:.4g}" if key is not None else None
        )
        title = (
            f"{panel_index + 1} {label} | "
            + ("neutral geometry | " if key is None else
               f"Turbo 0..{scale_text} ({range_label}) | ")
            + f"valid={np.count_nonzero(valid)}/{len(points)} | data={data_range}"
        )
        if panel_index == 3:
            tau = score_result["pca_ratio_saturation_tau"]
            ratio_mode = "raw identity" if tau is None else f"saturated tau={tau}"
            title += (
                f" | queried={np.count_nonzero(score_result['pca_query_mask'])} "
                f"invalid={np.count_nonzero(score_result['invalid_support_mask'])} "
                f"mode={ratio_mode} "
            )
        if panel_index == 5:
            extraction = score_result["candidate_extraction"]
            title += (
                f" | {extraction.get('fusion_threshold_mode', 'fixed')}:"
                f"score>{extraction['fusion_score_threshold']:.4g} "
                f"| NMS={candidate_debug['candidate_count']} | M: markers ON/OFF"
            )
        vis = (
            o3d.visualization.VisualizerWithKeyCallback()
            if panel_index == 5
            else o3d.visualization.Visualizer()
        )
        layout = window_layout(window_index)
        if not vis.create_window(
            window_name=title,
            **layout,
        ):
            for existing in windows:
                existing.destroy_window()
            raise RuntimeError("Failed to create dense heatmap windows")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        ratio_colors = None
        if key is None:
            cloud.colors = o3d.utility.Vector3dVector(
                np.full((len(points), 3), 0.72, dtype=np.float64)
            )
        elif panel_index == 3:
            ratio_colors = pca_ratio_heatmap_colors(
                score_result,
                vmax=display_vmax[key],
            )
            cloud.colors = o3d.utility.Vector3dVector(ratio_colors)
        else:
            panel_colors = dense_heatmap_colors(
                values, valid, vmax=display_vmax[key],
            )
            cloud.colors = o3d.utility.Vector3dVector(
                panel_colors
            )
        vis.add_geometry(cloud)
        if panel_index == 5:
            markers = build_candidate_markers()
            if markers is not None:
                vis.add_geometry(markers)
                marker_state = {"visible": True}
                marker_label = "candidate"

                def toggle_markers(
                    callback_vis,
                    marker_geometry=markers,
                    state=marker_state,
                    label=marker_label,
                ):
                    if state["visible"]:
                        callback_vis.remove_geometry(
                            marker_geometry, reset_bounding_box=False,
                        )
                    else:
                        callback_vis.add_geometry(
                            marker_geometry, reset_bounding_box=False,
                        )
                    state["visible"] = not state["visible"]
                    print(
                        f"[HEATMAP] {label} markers "
                        + ("ON" if state["visible"] else "OFF"),
                        flush=True,
                    )
                    callback_vis.update_renderer()
                    return False

                vis.register_key_callback(ord("M"), toggle_markers)
        option = vis.get_render_option()
        if option is not None:
            option.background_color = np.array([0.025, 0.025, 0.025])
            option.point_size = 4.0
        windows.append(vis)

    if rgb_points is not None:
        rgb_panel_index = len(panels)
        title = (
            "RealSense RGB + corner candidates (auxiliary, not a heatmap) | "
            f"points={len(rgb_points)} | candidates={len(candidate_debug['indices'])}"
        )
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(
            window_name=title,
            **window_layout(rgb_panel_index),
        ):
            for existing in windows:
                existing.destroy_window()
            raise RuntimeError("Failed to create RealSense RGB candidate window")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(rgb_points)
        cloud.colors = o3d.utility.Vector3dVector(rgb_colors)
        vis.add_geometry(cloud)
        markers = build_candidate_markers()
        if markers is not None:
            vis.add_geometry(markers)
        option = vis.get_render_option()
        if option is not None:
            option.background_color = np.array([0.025, 0.025, 0.025])
            option.point_size = 3.0
        windows.append(vis)

    active = [True] * len(windows)
    camera_signatures, _master = synchronize_dense_camera_views(
        windows,
        active,
        None,
    )
    while any(active):
        for index, vis in enumerate(windows):
            if active[index]:
                active[index] = bool(vis.poll_events())
        if not any(active):
            break
        camera_signatures, _master = synchronize_dense_camera_views(
            windows,
            active,
            camera_signatures,
        )
        for index, vis in enumerate(windows):
            if active[index]:
                vis.update_renderer()
    for vis in windows:
        vis.destroy_window()


def open_cornervins_plane_views(
    points_mm,
    organized_points_mm,
    analysis,
    previous=(),
    *,
    show_ahc=True,
    show_patches=True,
    show_hulls=True,
):
    """Open coarse AHC, refined AHC, RANSAC, and final hull views."""

    for view in previous or ():
        if view is not None:
            view.destroy_window()
    points = np.asarray(points_mm, dtype=np.float64)
    organized = np.asarray(organized_points_mm, dtype=np.float64)
    result = analysis["cornervins_result"]
    query = np.asarray(analysis["query_position_mm"], dtype=np.float64)
    palette = matplotlib.colormaps["tab20"]
    views = []

    def finish(view, *, point_size=3.5):
        option = view.get_render_option()
        option.background_color = np.asarray([0.025, 0.028, 0.035])
        option.point_size = point_size
        option.line_width = 4.0
        control = view.get_view_control()
        control.set_lookat(query)
        control.set_front(normalize([0.75, 0.55, 0.65]))
        control.set_up(normalize([0.0, -1.0, 0.0]))
        control.set_zoom(0.72)
        return view

    def add_query_roi(view, primitive):
        roi = o3d.geometry.LineSet()
        roi.points = o3d.utility.Vector3dVector(primitive["sphere_points_mm"])
        roi.lines = o3d.utility.Vector2iVector(primitive["sphere_lines"])
        roi.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.82, 0.08], (len(primitive["sphere_lines"]), 1))
        )
        view.add_geometry(roi)

        query_marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(1.0, 0.025 * primitive["query_radius_mm"]),
            resolution=8,
        )
        query_marker.translate(primitive["query_position_mm"])
        query_marker.paint_uniform_color([1.0, 0.08, 0.05])
        query_marker.compute_vertex_normals()
        view.add_geometry(query_marker)

    def open_organized_stage(*, stage, title, left, show_normals=False):
        primitive = cornervins_ahc_primitives(organized, analysis, stage=stage)
        view = o3d.visualization.Visualizer()
        if view.create_window(window_name=title, width=500, height=720, left=left, top=45):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(primitive["points_mm"])
            cloud.colors = o3d.utility.Vector3dVector(primitive["point_colors"])
            view.add_geometry(cloud)
            if show_normals:
                for plane_id, (center, normal) in enumerate(
                    zip(primitive["plane_centers_mm"], primitive["plane_normals"])
                ):
                    color = np.asarray(palette((plane_id % 20) / 19.0)[:3])
                    normal_line = o3d.geometry.LineSet()
                    normal_line.points = o3d.utility.Vector3dVector(
                        np.vstack(
                            (
                                center,
                                center
                                + 0.18
                                * analysis["query_radius_mm"]
                                * normalize(normal),
                            )
                        )
                    )
                    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    normal_line.colors = o3d.utility.Vector3dVector([color])
                    view.add_geometry(normal_line)
            add_query_roi(view, primitive)
            return finish(view, point_size=2.5)
        return None

    if show_ahc:
        for stage, title, left in (
            (
                "coarse",
                "CornerVINS 1/4 | coarse AHC (Feng Algorithm 3) | "
                f"R={analysis['query_radius_mm']:.1f}mm",
                10,
            ),
            (
                "refined",
                "CornerVINS 2/4 | refined AHC (Feng Algorithm 4)",
                490,
            ),
        ):
            view = open_organized_stage(
                stage=stage, title=title, left=left, show_normals=False
            )
            if view is not None:
                views.append(view)

    def open_patch_view(*, title, left, include_hulls):
        primitive = cornervins_scene_roi_primitives(points, analysis)
        colors = primitive["point_colors"].copy()
        view = o3d.visualization.Visualizer()
        if view.create_window(
            window_name=title, width=500, height=720, left=left, top=45,
        ):
            for plane in analysis["global_planes"]:
                plane_id = int(plane["plane_id"])
                color = np.asarray(palette((plane_id % 20) / 19.0)[:3])
                support = np.asarray(plane["support_indices"], dtype=np.int64)
                if np.any(support < 0) or np.any(support >= len(points)):
                    raise IndexError("CornerVINS plane support contains an invalid scene index")
                colors[support] = color
                center = np.asarray(plane["representative_center"], dtype=np.float64)
                normal = normalize(plane["representative_normal"])
                normal_line = o3d.geometry.LineSet()
                normal_line.points = o3d.utility.Vector3dVector(
                    np.vstack((center, center + 0.18 * analysis["query_radius_mm"] * normal))
                )
                normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                normal_line.colors = o3d.utility.Vector3dVector([color])
                view.add_geometry(normal_line)
                hull = np.asarray(plane.get("hull_vertices_3d", ()), dtype=np.float64)
                if include_hulls and len(hull) >= 2:
                    boundary = o3d.geometry.LineSet()
                    boundary.points = o3d.utility.Vector3dVector(hull)
                    boundary.lines = o3d.utility.Vector2iVector(
                        np.asarray(
                            [[index, (index + 1) % len(hull)] for index in range(len(hull))],
                            dtype=np.int32,
                        )
                    )
                    boundary.colors = o3d.utility.Vector3dVector(
                        np.tile(np.clip(0.65 * color + 0.35, 0.0, 1.0), (len(hull), 1))
                    )
                    view.add_geometry(boundary)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(primitive["points_mm"])
            cloud.colors = o3d.utility.Vector3dVector(colors)
            view.add_geometry(cloud)
            add_query_roi(view, primitive)
            return finish(view)
        return None

    if show_patches:
        view = open_organized_stage(
            stage="ransac",
            title=(
                "CornerVINS 3/4 | RANSAC accepted + LS refit | "
                f"planes={len(result.ransac_patches)}"
            ),
            left=970,
            show_normals=True,
        )
        if view is not None:
            views.append(view)

    if show_hulls:
        view = open_patch_view(
            title=(
                "CornerVINS 4/4 | merged final planes + convex hulls | "
                f"planes={len(analysis['global_planes'])} yellow=ROI red=query"
            ),
            left=1450,
            include_hulls=True,
        )
        if view is not None:
            views.append(view)

    return tuple(views)


def plane_edge_refiner_primitives(analysis):
    """Prepare refined segments with a stable 1/2/3-plane color category.

    A three-plane detector edge is expanded into plane-pair edges by the
    refiner. The category therefore comes from the immutable source edge,
    rather than the pair stored on each refined edge.
    """
    refinement = analysis.get("refinement")
    if not refinement:
        return []
    source_by_id = {
        int(edge["edge_id"]): edge for edge in analysis.get("edges", ())
    }
    primitives = []
    for edge in refinement.get("edges", ()):
        refined = edge.get("refined", {})
        start = np.asarray(refined.get("start_mm"), dtype=np.float64)
        end = np.asarray(refined.get("end_mm"), dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            continue
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        source = source_by_id.get(int(edge.get("source_edge_id", -1)), edge)
        plane_count = len(set(int(value) for value in source.get("plane_ids", ())))
        category = min(3, max(1, plane_count))
        primitives.append({
            "refined_edge_id": int(edge.get("refined_edge_id", -1)),
            "source_edge_id": int(edge.get("source_edge_id", -1)),
            "plane_count": int(plane_count),
            "category": int(category),
            "start_mm": start.copy(),
            "end_mm": end.copy(),
            "color": _REFINER_EDGE_COLORS[category].copy(),
        })
    return primitives


def _segment_cylinder(start, end, radius, color):
    """Return a cylinder aligned to one finite 3-D segment."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 1e-8:
        return None
    direction = delta / length
    z_axis = np.asarray([0.0, 0.0, 1.0])
    cosine = float(np.clip(z_axis @ direction, -1.0, 1.0))
    if cosine < -1.0 + 1e-10:
        rotation = np.diag([1.0, -1.0, -1.0])
    else:
        cross = np.cross(z_axis, direction)
        skew = np.asarray([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        rotation = np.eye(3) + skew + (skew @ skew) / max(1.0 + cosine, 1e-12)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=length, resolution=12, split=1,
    )
    mesh.rotate(rotation, center=np.zeros(3))
    mesh.translate(0.5 * (start + end))
    mesh.paint_uniform_color(np.asarray(color, dtype=float))
    mesh.compute_vertex_normals()
    return mesh


def show_detector_validator_results(points_mm, validation, *, voxel_3d_mm):
    """Show only final validator edges and corners over neutral scene context."""
    points = np.asarray(points_mm, dtype=np.float64)
    edges = validation.get("final_edges", ())
    corners = validation.get("final_corners", ())
    diagnostics = validation.get("diagnostics", {})
    vis = o3d.visualization.Visualizer()
    title = (
        "Detector final | dense NMS -> shared validator -> refiner | "
        f"edges={len(edges)} corners={len(corners)} "
        f"queries={diagnostics.get('completed_count', 0)}/"
        f"{diagnostics.get('candidate_count', 0)} | "
        "orange=1-plane cyan=2-plane violet=3-plane red=corner"
    )
    if not vis.create_window(window_name=title, width=1100, height=820):
        print("[3D] Could not open detector final edge/corner view.")
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0.16, 0.17, 0.20])
    vis.add_geometry(cloud)

    edge_radius = max(1.5, min(3.0, 0.18 * float(voxel_3d_mm)))
    for edge in edges:
        category = min(3, max(1, int(edge.get("plane_count", 1))))
        mesh = _segment_cylinder(
            edge["start_mm"], edge["end_mm"], edge_radius,
            _REFINER_EDGE_COLORS[category],
        )
        if mesh is not None:
            vis.add_geometry(mesh)
    corner_radius = max(4.0, 2.4 * edge_radius)
    for corner in corners:
        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=corner_radius, resolution=12,
        )
        marker.translate(np.asarray(corner["position_mm"], dtype=np.float64))
        marker.paint_uniform_color([1.0, 0.04, 0.04])
        marker.compute_vertex_normals()
        vis.add_geometry(marker)

    option = vis.get_render_option()
    option.background_color = np.asarray([0.018, 0.020, 0.027])
    option.point_size = 2.5
    if len(points):
        control = vis.get_view_control()
        control.set_lookat(np.mean(points, axis=0))
        control.set_front(normalize([0.75, 0.55, 0.65]))
        control.set_up(normalize([0.0, -1.0, 0.0]))
        control.set_zoom(0.72)
    vis.run()
    vis.destroy_window()


def open_plane_edge_graph_view(points_mm, analysis, previous=None):
    """Display only the immutable detector plane/edge graph."""
    if previous is not None:
        previous.destroy_window()
    points = np.asarray(points_mm, dtype=np.float64)
    vis = o3d.visualization.Visualizer()
    d = analysis['diagnostics']
    title = (f"Plane -> padded 2D boundary -> 3D Edge Graph | {analysis.get('search_scope', 'query')} "
             f"ROI={len(analysis['roi_point_indices'])} R={analysis.get('query_radius_mm') or 0:.1f}mm "
             f"planes={d['plane_count']} hough={d['hough_peaks']} "
             f"edges={d['merged_edge_count']} shared={d['shared_edge_count']} "
             f"corners={d.get('single_plane_corner_count',0)} "
             '| blue=Hough/PCA edge orange=detector corner')
    if not vis.create_window(window_name=title, width=1050, height=800):
        print('[3D] Could not open projected-Hough edge graph view.')
        return None
    base = o3d.geometry.PointCloud()
    local_ids = np.asarray(analysis["roi_point_indices"], dtype=np.int64)
    base.points = o3d.utility.Vector3dVector(points)
    base.paint_uniform_color([0.12, 0.13, 0.15])
    vis.add_geometry(base)
    core_cloud = o3d.geometry.PointCloud()
    core_cloud.points = o3d.utility.Vector3dVector(points[local_ids])
    core_cloud.paint_uniform_color([0.28, 0.30, 0.34])
    vis.add_geometry(core_cloud)
    # Explicitly show the spherical query neighborhood used by ALL stages.
    if analysis.get('query_position_mm') is not None and analysis.get('query_radius_mm'):
        q = np.asarray(analysis['query_position_mm'], dtype=float)
        r = float(analysis['query_radius_mm'])
        angles = np.linspace(0, 2*np.pi, 73)
        ring_points, ring_lines = [], []
        for a, b in ((0,1), (0,2), (1,2)):
            ring = np.tile(q, (len(angles),1))
            ring[:,a] += r * np.cos(angles)
            ring[:,b] += r * np.sin(angles)
            offset = len(ring_points)
            ring_points.extend(ring)
            ring_lines.extend([(offset+i,offset+i+1) for i in range(len(angles)-1)])
        radius_lines = o3d.geometry.LineSet()
        radius_lines.points = o3d.utility.Vector3dVector(np.asarray(ring_points))
        radius_lines.lines = o3d.utility.Vector2iVector(np.asarray(ring_lines))
        radius_lines.paint_uniform_color([0.95,0.35,0.20])
        vis.add_geometry(radius_lines)
    for i, plane in enumerate(analysis['planes']):
        # Show full plane support, not just a small boundary stripe in a gray scene.
        support_ids = np.asarray(plane['support_indices'], dtype=np.int64)
        if len(support_ids):
            plane_cloud = o3d.geometry.PointCloud()
            plane_cloud.points = o3d.utility.Vector3dVector(points[support_ids])
            base_rgb = np.asarray(_PLANE_COLORS[i % len(_PLANE_COLORS)], dtype=float)
            plane_cloud.paint_uniform_color((0.46 * base_rgb + 0.16).tolist())
            vis.add_geometry(plane_cloud)
            # Fitted plane wireframe in the original 3-D coordinate system.
            # Reuse fitted-plane coordinate frame and its observed 2-D range.
            plane_points = points[support_ids]
            origin = np.asarray(plane['center_mm'], dtype=float)
            u, v = np.asarray(plane['u'], dtype=float), np.asarray(plane['v'], dtype=float)
            planar = np.column_stack(((plane_points-origin)@u, (plane_points-origin)@v))
            lo, hi = np.min(planar, axis=0), np.max(planar, axis=0)
            coords = np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],
                               [lo[0],hi[1]]], dtype=float)
            if np.linalg.norm(hi-lo) > 1e-6:
                wire = o3d.geometry.LineSet()
                wire.points = o3d.utility.Vector3dVector(
                    origin + coords[:,0,None]*u + coords[:,1,None]*v)
                wire.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]])
                wire.colors = o3d.utility.Vector3dVector(np.tile(base_rgb, (4,1)))
                vis.add_geometry(wire)
        ids = np.asarray(plane['boundary_indices'], dtype=np.int64)
        if len(ids) == 0:
            continue
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[ids])
        cloud.paint_uniform_color(_PLANE_COLORS[i % len(_PLANE_COLORS)])
        vis.add_geometry(cloud)
    # Detector output is retained as the immutable Hough/PCA initialization.
    segments = analysis['edges']
    if segments:
        xyz = np.vstack([[e['start_mm'], e['end_mm']] for e in segments])
        line_ids = np.asarray([[2*i,2*i+1] for i in range(len(segments))], dtype=np.int32)
        colors = np.asarray([[0.12, 0.48, 1.00] for _edge in segments], dtype=np.float64)
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(xyz)
        lines.lines = o3d.utility.Vector2iVector(line_ids)
        lines.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(lines)
    # Single-plane boundary junctions are valid corner candidates even when
    # no other plane or shared edge has been observed.
    for corner in analysis.get('corners', []):
        # Marker radius is visualization-only, and does not affect localization.
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 0.34, 0.08])
        sphere.translate(np.asarray(corner['position_mm'], dtype=float))
        vis.add_geometry(sphere)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.025,0.028,0.035])
    opt.point_size = 3.0
    opt.line_width = 5.0
    if analysis.get('query_position_mm') is not None:
        control = vis.get_view_control()
        control.set_lookat(np.asarray(analysis['query_position_mm'], dtype=float))
        control.set_front(normalize([0.75, 0.55, 0.65]))
        control.set_up(normalize([0.0, -1.0, 0.0]))
        control.set_zoom(0.72)
    return vis


def open_plane_edge_refiner_view(points_mm, analysis, previous=None):
    """Display refined planes, thick categorized edges, and refined corners."""
    if previous is not None:
        previous.destroy_window()
    refinement = analysis.get("refinement")
    if not refinement:
        return None
    points = np.asarray(points_mm, dtype=np.float64)
    primitives = plane_edge_refiner_primitives(analysis)
    counts = {
        category: sum(item["category"] == category for item in primitives)
        for category in (1, 2, 3)
    }
    diagnostics = refinement.get("diagnostics", {})
    title = (
        "Hough Edge Refiner | "
        f"graphs={diagnostics.get('graph_count', 0)} "
        f"edges={len(primitives)} (1P={counts[1]} 2P={counts[2]} 3P={counts[3]}) "
        f"corners={diagnostics.get('corner_count', 0)} | "
        "orange=1-plane cyan=2-plane violet=3-plane "
        "magenta=plane intersection red=corner"
    )
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(
        window_name=title, width=850, height=800, left=1060, top=45,
    ):
        print("[3D] Could not open Hough edge refiner view.")
        return None

    base = o3d.geometry.PointCloud()
    base.points = o3d.utility.Vector3dVector(points)
    base.paint_uniform_color([0.09, 0.10, 0.12])
    vis.add_geometry(base)
    local_ids = np.asarray(analysis.get("roi_point_indices", ()), dtype=np.int64)
    if len(local_ids):
        core = o3d.geometry.PointCloud()
        core.points = o3d.utility.Vector3dVector(points[local_ids])
        core.paint_uniform_color([0.24, 0.26, 0.30])
        vis.add_geometry(core)

    # The refined plane context is deliberately muted so the thick edge
    # cylinders remain legible even on dense point clouds.
    for index, plane in enumerate(refinement.get("planes", ())):
        support_ids = np.asarray(plane.get("support_indices", ()), dtype=np.int64)
        if not len(support_ids):
            continue
        base_color = np.asarray(_PLANE_COLORS[index % len(_PLANE_COLORS)])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[support_ids])
        cloud.paint_uniform_color((0.32 * base_color + 0.12).tolist())
        vis.add_geometry(cloud)

        refined = plane["refined"]
        center = np.asarray(refined["center_mm"], dtype=np.float64)
        u = np.asarray(refined["u"], dtype=np.float64)
        v = np.asarray(refined["v"], dtype=np.float64)
        planar = np.column_stack(((points[support_ids] - center) @ u,
                                  (points[support_ids] - center) @ v))
        lo, hi = np.min(planar, axis=0), np.max(planar, axis=0)
        if np.linalg.norm(hi - lo) > 1e-6:
            uv = np.asarray([
                [lo[0], lo[1]], [hi[0], lo[1]],
                [hi[0], hi[1]], [lo[0], hi[1]],
            ])
            wire = o3d.geometry.LineSet()
            wire.points = o3d.utility.Vector3dVector(
                center + uv[:, :1] * u + uv[:, 1:] * v
            )
            wire.lines = o3d.utility.Vector2iVector(
                np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
            )
            wire.paint_uniform_color(base_color)
            vis.add_geometry(wire)

    query_radius = float(analysis.get("query_radius_mm") or 0.0)
    edge_radius = max(1.5, min(3.0, query_radius * 0.018))
    intersection_radius = max(0.65, edge_radius * 0.45)
    for item in refinement.get("plane_intersections", ()):
        mesh = _segment_cylinder(
            item["start_mm"], item["end_mm"], intersection_radius,
            [1.00, 0.12, 0.82],
        )
        if mesh is not None:
            vis.add_geometry(mesh)
    for item in primitives:
        mesh = _segment_cylinder(
            item["start_mm"], item["end_mm"], edge_radius, item["color"],
        )
        if mesh is not None:
            vis.add_geometry(mesh)
    for corner in refinement.get("corners", ()):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(4.0, 2.4 * edge_radius), resolution=12,
        )
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 0.04, 0.04])
        sphere.translate(np.asarray(corner["position_mm"], dtype=np.float64))
        vis.add_geometry(sphere)

    options = vis.get_render_option()
    options.background_color = np.asarray([0.018, 0.020, 0.027])
    options.point_size = 2.5
    options.line_width = 2.0
    if analysis.get("query_position_mm") is not None:
        control = vis.get_view_control()
        control.set_lookat(np.asarray(analysis["query_position_mm"], dtype=float))
        control.set_front(normalize([0.75, 0.55, 0.65]))
        control.set_up(normalize([0.0, -1.0, 0.0]))
        control.set_zoom(0.72)
    return vis


__all__ = [
    "centroid_saliency_colors",
    "dense_heatmap_colors",
    "pca_ratio_heatmap_colors",
    "dense_heatmap_vmax",
    "corner_candidate_primitives",
    "cornervins_ahc_primitives",
    "cornervins_scene_roi_primitives",
    "synchronize_dense_camera_views",
    "dense_heatmap_panels",
    "show_dense_score_heatmaps",
    "open_cornervins_plane_views",
    "plane_edge_refiner_primitives",
    "show_detector_validator_results",
    "open_plane_edge_graph_view",
    "open_plane_edge_refiner_view",
]
