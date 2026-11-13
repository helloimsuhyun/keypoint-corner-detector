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

_Q10_PLANE_COLORS = np.array([(255, 130, 62), (81, 218, 129), (159, 130, 255)]) / 255.

_REFINER_EDGE_COLORS = {
    1: np.asarray([1.00, 0.62, 0.08], dtype=np.float64),
    2: np.asarray([0.05, 0.92, 1.00], dtype=np.float64),
    3: np.asarray([0.86, 0.22, 1.00], dtype=np.float64),
}

class _LazyOpen3D:
    def __getattr__(self, name):
        return getattr(require_open3d(), name)


o3d = _LazyOpen3D()


def open_nms_peak_view(rgb, *, display=None):
    """Reuse a native image window in the caller's already initialized GUI app."""
    import open3d.visualization.gui as gui
    if display is None or display['closed']:
        window = gui.Application.instance.create_window('NMS | score relative to neighbors', 1210, 572)
        image = gui.ImageWidget()
        window.add_child(image)
        display = dict(window=window, image=image, closed=False)
        def layout(_):
            image.frame = window.content_rect
        def closed():
            display['closed'] = True
            return True
        window.set_on_layout(layout)
        window.set_on_close(closed)
    display['image'].update_image(o3d.geometry.Image(np.ascontiguousarray(rgb)))
    display['window'].set_needs_layout()
    return display


def synchronize_baseline_cameras(widgets, previous=None):
    """Copy rotation, pan and zoom from whichever comparison panel changed."""
    def signature(widget):
        camera = widget.scene.camera
        return np.concatenate((np.asarray(camera.get_view_matrix()).ravel(),
                               np.asarray(camera.get_projection_matrix()).ravel()))
    current = [signature(widget) for widget in widgets]
    changed = ([0] if previous is None else
               [i for i, (now, before) in enumerate(zip(current, previous))
                if not np.allclose(now, before, rtol=0, atol=1e-7)])
    if changed:
        master = changed[0]
        for i, widget in enumerate(widgets):
            if i != master:
                widget.scene.camera.copy_from(widgets[master].scene.camera)
            widget.force_redraw()
        current = [signature(widget) for widget in widgets]
    return current, bool(changed)


def open_baseline_comparison_view(points_mm, rgb, gt_mm, results, *, title):
    """One row: GT followed by each method on the identical saved input cloud."""
    points = np.asarray(points_mm)
    colors = np.asarray(rgb)
    panels = [{"method": "GT", "status": "ok", "xyz_mm": np.asarray(gt_mm)}, *results]
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or colors.shape != points.shape or not np.isfinite(points).all()):
        raise ValueError("Comparison requires finite nonempty XYZ and matching RGB")
    for panel in panels:
        xyz = np.asarray(panel["xyz_mm"])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("Comparison corners must be finite (K, 3) coordinates in mm")
    gui, rendering = o3d.visualization.gui, o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(f"BASELINES | {title}", min(1920, 520 * len(panels)), 760)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(.25 + .35 * colors.astype(float) / 255.)
    cloud_material = rendering.MaterialRecord()
    cloud_material.shader, cloud_material.point_size = "defaultUnlit", 2.0
    marker_material = rendering.MaterialRecord()
    marker_material.shader, marker_material.point_size = "defaultUnlit", 11.0
    bounds = cloud.get_axis_aligned_bounding_box()
    widgets, labels = [], []
    names = {"harris3d": "PCL Harris3D", "orthogonal": "Sommer", "ahmed": "Ahmed"}
    for index, panel in enumerate(panels):
        widget = gui.SceneWidget()
        widget.scene = rendering.Open3DScene(window.renderer)
        widget.scene.set_background([.07, .07, .09, 1.])
        widget.scene.add_geometry("input_cloud", cloud, cloud_material, False)
        xyz = panel["xyz_mm"]
        if len(xyz):
            markers = o3d.geometry.PointCloud()
            markers.points = o3d.utility.Vector3dVector(xyz)
            markers.paint_uniform_color([.1, 1., .25] if index == 0 else [1., .15, .1])
            widget.scene.add_geometry("corners", markers, marker_material, False)
        name = names.get(panel["method"], panel["method"])
        if panel.get("implementation") == "python_port":
            name += " (Python port)"
        status = f"{len(xyz)} corners" if panel["status"] == "ok" else panel["status"].upper()
        label = gui.Label(f"{name} | {status}")
        label.tooltip = panel.get("error", "") or "Drag: rotate | wheel: zoom | camera synchronized across all panels"
        window.add_child(label)
        window.add_child(widget)
        widget.setup_camera(60., bounds, bounds.get_center())
        center = bounds.get_center()
        distance = max(float(np.linalg.norm(bounds.get_extent())), 1.) * 1.5
        widget.scene.camera.look_at(center, center + [0., 0., -distance], [0., -1., 0.])
        widgets.append(widget)
        labels.append(label)

    def layout(_):
        rect = window.content_rect
        for index, (widget, label) in enumerate(zip(widgets, labels)):
            left = rect.x + index * rect.width // len(widgets)
            right = rect.x + (index + 1) * rect.width // len(widgets)
            label.frame = gui.Rect(left + 8, rect.y, right - left - 8, 32)
            widget.frame = gui.Rect(left, rect.y + 32, right - left, max(1, rect.height - 32))

    signatures, _ = synchronize_baseline_cameras(widgets)
    closed = False

    def close():
        nonlocal closed
        closed = True
        return True

    def tick():
        nonlocal signatures
        if closed:
            return False
        signatures, changed = synchronize_baseline_cameras(widgets, signatures)
        return changed

    window.set_on_layout(layout)
    window.set_on_tick_event(tick)
    window.set_on_close(close)
    return {"app": app, "window": window, "widgets": widgets, "labels": labels}


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


def open_guided_validator_view(scene_points_mm, query_indices, scene_colors, *, fusion_vmax=1.0,
                               raw_points_mm=None, query_positions_mm=None, projection_only=False):
    """Upload Coarse once; display query-local voxel input and fitted planes separately."""
    fusion_vmax = float(fusion_vmax)
    if not np.isfinite(fusion_vmax) or fusion_vmax < 0:
        raise ValueError("Fusion display maximum must be finite and nonnegative")
    scene_points = np.asarray(scene_points_mm)
    queries = scene_points[query_indices] if query_positions_mm is None else np.asarray(query_positions_mm, dtype=float)
    if queries.shape != (len(query_indices), 3) or not np.isfinite(queries).all():
        raise ValueError("Query marker coordinates must align with scene query IDs")
    gui, rendering = o3d.visualization.gui, o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(("NEW VALIDATOR" if projection_only else "VALIDATOR") +
                               " | 1. Coarse -> 2. Query voxel -> 3. Plane fitting", 1800, 650)
    overview, roi_view, local = gui.SceneWidget(), gui.SceneWidget(), gui.SceneWidget()
    panels = (overview, roi_view, local)
    labels = [gui.Label(text) for text in (
        f"1. Coarse Fusion 0..{fusion_vmax:.4g} (frame Q99) | click orange query",
        "2. Query radius -> voxel input | select Coarse first",
        "3. Plane fitting | select Coarse first",
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
            label_height = 52 if index in (1, 2) else 28
            label.frame = gui.Rect(left, rect.y, right - left, label_height)
            widget.frame = gui.Rect(left, rect.y + label_height, right - left, max(1, rect.height - label_height))

    window.set_on_layout(layout)
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
               "roi_view": roi_view, "local": local, "labels": labels, "query_xyz": queries.copy(),
               "query_indices": np.asarray(query_indices).copy(), "closed": False,
               "fusion_vmax": fusion_vmax}
    display["geometry_windows"] = []
    if projection_only:
        projection_window = app.create_window("NEW VALIDATOR | 4. 2D plane projections", 1260, 480)
        label, image = gui.Label("4. 2D projection | select a query"), gui.ImageWidget()
        projection_window.add_child(label)
        projection_window.add_child(image)
        stage = {"window": projection_window, "number": 4, "edge_type": None, "panels": [], "closed": False}
        def projection_layout(_):
            rect = projection_window.content_rect
            label.frame = gui.Rect(rect.x, rect.y, rect.width, 28)
            image.frame = gui.Rect(rect.x, rect.y + 28, rect.width, max(1, rect.height - 28))
        def projection_closed():
            stage["closed"] = True
            return True
        projection_window.set_on_layout(projection_layout)
        projection_window.set_on_close(projection_closed)
        display.update(projection_image=image, projection_label=label, projection_stage=stage)
        display["geometry_windows"].append(stage)
    for number, title, edge_type in (() if projection_only else (
        (4, "Padded hull / ROI clipping / TLS / Before-After | S: save PNG", None),
        (5, "Plane-pair intersection edges", "plane_intersection"),
        (6, "Edges -> overlap removal -> endpoint Corners -> all candidates on raw cloud", None),
    )):
        stage_window = app.create_window(f"VALIDATOR | {number}. {title}",
                                         1800 if number in (4, 6) else 1000,
                                         1000 if number in (4, 6) else 800)
        stage = {"window": stage_window, "number": number, "edge_type": edge_type,
                 "panels": [], "closed": False}
        for index in range(5 if number == 4 else 4 if number == 6 else 1):
            widget = gui.SceneWidget()
            widget.scene = rendering.Open3DScene(stage_window.renderer)
            widget.scene.set_background([0.08, 0.08, 0.08, 1.0])
            raw_count = 0
            if number == 6 and index == 3 and raw_points_mm is not None:
                raw_points = np.asarray(raw_points_mm)
                raw_count = len(raw_points)
                if raw_count:
                    raw_cloud = o3d.geometry.PointCloud()
                    raw_cloud.points = o3d.utility.Vector3dVector(raw_points)
                    raw_cloud.paint_uniform_color([.38, .38, .38])
                    raw_material = rendering.MaterialRecord()
                    raw_material.shader, raw_material.point_size = "defaultUnlit", 1.5
                    widget.scene.add_geometry("original_raw_cloud", raw_cloud, raw_material)
            label = gui.Label(f"{number}. NOT_COMPUTED | select Coarse first")
            stage_window.add_child(label)
            stage_window.add_child(widget)
            stage["panels"].append({"widget": widget, "label": label,
                                    "plane_id": None,
                                    "graph_step": index + 1 if number == 6 else None,
                                    "overlay_geometry_names": [], "raw_point_count": raw_count,
                                    "roi_debug_view": ("A", "B", "C", "D-before", "D-after")[index]
                                    if number == 4 else None})

        def stage_layout(_, stage=stage):
            rect = stage["window"].content_rect
            count = len(stage["panels"])
            for index, panel in enumerate(stage["panels"]):
                columns = (3 if index < 3 else 2) if stage["number"] == 4 else count
                column = (index if index < 3 else index - 3) if stage["number"] == 4 else index
                top = rect.y + (rect.height // 2 if stage["number"] == 4 and index >= 3 else 0)
                if stage["number"] == 6:
                    columns, column = 2, index % 2
                    top = rect.y + (index // 2) * (rect.height // 2)
                available_height = rect.height // 2 if stage["number"] in (4, 6) else rect.height
                left = rect.x + column * rect.width // columns
                right = rect.x + (column + 1) * rect.width // columns
                height = 92 if stage["number"] == 4 else 72 if stage["number"] == 6 else 52
                panel["label"].frame = gui.Rect(left, top, right - left, height)
                panel["widget"].frame = gui.Rect(left, top + height, right - left, max(1, available_height - height))

        def stage_closed(stage=stage):
            stage["closed"] = True
            return True

        stage_window.set_on_layout(stage_layout)
        stage_window.set_on_close(stage_closed)
        display["geometry_windows"].append(stage)

    def closed():
        display["closed"] = True
        for stage in display["geometry_windows"]:
            if not stage["closed"]:
                stage["window"].close()
                stage["closed"] = True
        return True

    window.set_on_close(closed)
    return display


def update_validator_projection_view(display, rgb):
    stage = display["projection_stage"]
    if not stage["closed"]:
        display["projection_image"].update_image(o3d.geometry.Image(np.ascontiguousarray(rgb)))
        display["projection_label"].text = "4. 2D UV projection | hull / edge / corner not computed"
        stage["window"].post_redraw()


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


def update_validator_roi_view(display, result):
    """Show the stored core input after raw-radius cropping and voxelization."""
    rendering = o3d.visualization.rendering
    widget = display["roi_view"]
    scene = widget.scene
    scene.clear_geometry()
    query, radius = result["query_position_mm"], result["radius_mm"]
    points = result["roi_points_mm"]
    material = rendering.MaterialRecord()
    material.shader, material.point_size = "defaultUnlit", 4.0
    if len(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.paint_uniform_color([.7, .7, .7])
        scene.add_geometry("voxel_input", cloud, material)
    vertices, indices = _wire_sphere(query, radius, 64)
    wire = o3d.geometry.LineSet()
    wire.points = o3d.utility.Vector3dVector(vertices)
    wire.lines = o3d.utility.Vector2iVector(indices)
    wire.paint_uniform_color([.85, .65, 0.])
    material = rendering.MaterialRecord()
    material.shader, material.line_width = "unlitLine", 1.0
    scene.add_geometry("query_radius", wire, material)
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.5, radius * .035))
    marker.translate(query)
    material = rendering.MaterialRecord()
    material.shader, material.base_color = "defaultUnlit", [1., .5, 0., 1.]
    scene.add_geometry("coarse_query", marker, material)
    display["labels"][1].text = (
        f"2. Query radius {radius:g} mm -> voxel {result['ransac_voxel_mm']:g} mm\n"
        f"Raw core: {len(result['roi_raw_point_indices'])} -> {len(points)} RANSAC input points")
    if display.get("roi_query_index") != result["query_index"]:
        bounds = o3d.geometry.AxisAlignedBoundingBox(query - radius, query + radius)
        widget.setup_camera(60.0, bounds, query)
        scene.camera.look_at(query, query + radius * np.array([1.7, 1.3, 1.5]), [0., 0., 1.])
        display["roi_query_index"] = result["query_index"]
    widget.force_redraw()


def _add_validator_corner_geometry(scene, result, *, edge_type=None, plane_id=None):
    """Display stored support, hulls, finite edges and vertices without fitting anything."""
    rendering = o3d.visualization.rendering

    def lines(name, points, indices, color, width):
        wire = o3d.geometry.LineSet()
        wire.points = o3d.utility.Vector3dVector(points)
        wire.lines = o3d.utility.Vector2iVector(indices)
        wire.paint_uniform_color(color)
        material = rendering.MaterialRecord()
        material.shader, material.line_width = "unlitLine", width
        scene.add_geometry(name, wire, material)

    points = result["support_roi_points_mm"]
    patches = [patch for patch in result["plane_patches"]
               if plane_id is None or patch["plane_id"] == plane_id]
    ids = result["support_roi_ransac_indices"]
    if plane_id is not None:
        positions = np.searchsorted(ids, patches[0]["support_ransac_indices"]) if patches else np.empty(0, dtype=int)
        points, ids = points[positions], ids[positions]
    if len(points):
        colors = np.full(points.shape, .32)
        for patch in patches:
            positions = np.searchsorted(ids, patch["support_ransac_indices"])
            colors[positions] = .25 + .55 * np.eye(3)[patch["plane_id"] % 3]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        material = rendering.MaterialRecord()
        material.shader, material.point_size = "defaultUnlit", 2.5
        scene.add_geometry("support_context", cloud, material)
    if edge_type is None:
        vertices, indices = _wire_sphere(result["query_position_mm"], result["support_radius_mm"], 64)
        lines("support_radius", vertices, indices, [.4, .4, .4], 1.)
        vertices, indices = _wire_sphere(result["query_position_mm"], result["radius_mm"], 64)
        lines("core_radius", vertices, indices, [.85, .65, 0.], 1.)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=.8)
        marker.translate(result["query_position_mm"])
        material = rendering.MaterialRecord()
        material.shader, material.base_color = "defaultUnlit", [1., .5, 0., 1.]
        scene.add_geometry("coarse_query", marker, material)
    for patch in patches:
        vertices = patch["hull_vertices_mm"]
        if not len(vertices):
            continue
        color = np.eye(3)[patch["plane_id"] % 3].tolist() if patch["valid"] else [.7, .7, .7]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector([[0, j, j + 1] for j in range(1, len(vertices) - 1)])
        mesh.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader, material.base_color = "defaultLitTransparency", color + [.18]
        material.has_alpha = True
        scene.add_geometry(f"hull_fill_{patch['plane_id']}", mesh, material)
        lines(f"hull_rim_{patch['plane_id']}", vertices,
              [[j, (j + 1) % len(vertices)] for j in range(len(vertices))],
              [.35, .35, .35] if edge_type is not None else color, 1.5)
    if edge_type == "plane_intersection":
        by_id = {patch["plane_id"]: patch for patch in patches}
        radius = result["support_radius_mm"]
        for pair in result.get("plane_pair_diagnostics", ()):
            origin = pair.get("intersection_line_origin_mm")
            if origin is None:
                continue
            direction = pair["intersection_line_direction"]
            start, end = origin - radius * direction, origin + radius * direction
            name = "intersection_" + "_".join(str(index) for index in pair["plane_ids"])
            lines(name + "_infinite_line", [start, end], [[0, 1]], [1., .85, .1], 2.)
            width = pair["strip_half_width_mm"]
            if width <= 0:
                continue
            for index in pair["plane_ids"]:
                patch = by_id[index]
                across = np.cross(patch["normal"], direction)
                across /= np.linalg.norm(across)
                vertices = np.array([start - width * across, end - width * across,
                                     end + width * across, start + width * across])
                # Display-only offset prevents flicker against the coplanar hull fill.
                vertices += .001 * max(1., radius) * patch["normal"]
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
                mesh.compute_vertex_normals()
                color = np.eye(3)[index % 3].tolist()
                material = rendering.MaterialRecord()
                material.shader, material.base_color = "defaultLitTransparency", color + [.28]
                material.has_alpha = True
                scene.add_geometry(f"{name}_strip_plane_{index}", mesh, material)
                lines(f"{name}_strip_bounds_plane_{index}", vertices,
                      [[0, 1], [3, 2]], color, 1.)
    for edge in result["edge_candidates"]:
        if edge_type is not None and edge["edge_type"] != edge_type:
            continue
        if plane_id is not None and plane_id not in edge["plane_ids"]:
            continue
        if not edge["valid"]:
            lines(f"unknown_edge_{edge['source_edge_id']}", [edge["start_mm"], edge["end_mm"]],
                  [[0, 1]], [.45, .45, .45], 1.)
    selected = {edge_id for corner in result["corners"] for edge_id in corner["edge_ids"]} if edge_type is None else set()
    for edge in result["edges"]:
        if edge_type is not None and edge["edge_type"] != edge_type:
            continue
        if plane_id is not None and plane_id not in edge["plane_ids"]:
            continue
        color = ([.1, 1., .2] if edge["edge_id"] in selected else
                 [0., .9, 1.] if edge["edge_type"] == "plane_intersection" else [1., .6, .1])
        segments = ([edge["start_mm"] + time * edge["direction"]
                     for interval in edge["active_intervals_mm"] for time in interval]
                    if "active_intervals_mm" in edge else [edge["start_mm"], edge["end_mm"]])
        lines(f"finite_edge_{edge['edge_id']}", segments,
              [[i, i + 1] for i in range(0, len(segments), 2)],
              color, 5. if edge["edge_id"] in selected else 3.)
    if edge_type is not None:
        return
    for index, corner in enumerate(result["corners"]):
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=1.6)
        marker.translate(corner["position"])
        material = rendering.MaterialRecord()
        material.shader, material.base_color = "defaultUnlit", [.1, 1., .2, 1.]
        scene.add_geometry(f"corner_candidate_{index}", marker, material)


def _add_validator_graph_step(scene, result, step):
    """Read one query's original edges, masks and endpoint Corners into a process panel."""
    rendering = o3d.visualization.rendering
    geometry_names = []

    def lines(name, points, color, width):
        if not len(points):
            return
        wire = o3d.geometry.LineSet()
        wire.points = o3d.utility.Vector3dVector(points)
        wire.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, len(points), 2)])
        wire.paint_uniform_color(color)
        material = rendering.MaterialRecord()
        material.shader, material.line_width = "unlitLine", width
        scene.add_geometry(name, wire, material)
        geometry_names.append(name)

    def intervals(name, edge, ranges, color, width, dashed=False):
        direction = edge["end_mm"] - edge["start_mm"]
        length = np.linalg.norm(direction)
        if length <= 1e-10:
            return
        direction = direction / length
        points = []
        for low, high in ranges:
            if high - low <= 1e-8:
                continue
            dash = min(3., (high - low) / 5.)
            pieces = [(low, high)] if not dashed else [
                (start, min(start + dash, high)) for start in np.arange(low, high, 2 * dash)]
            points.extend(edge["start_mm"] + time * direction for piece in pieces for time in piece)
        lines(name, points, color, width)

    original = result.get("all_edges", result["edges"])
    active = result["edges"]
    overlap = result.get("edge_overlap_settings")
    if step == 2 and overlap is not None:
        radius = overlap["distance_mm"]
        for edge in active:
            if edge["edge_type"] != "plane_intersection":
                continue
            direction = edge["end_mm"] - edge["start_mm"]
            length = np.linalg.norm(direction)
            if length <= 1e-10:
                continue
            u, v, axis = make_tangent_basis(direction / length)
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=float(length), resolution=32)
            mesh.rotate(np.column_stack((u, v, axis)), center=np.zeros(3))
            mesh.translate(.5 * (edge["start_mm"] + edge["end_mm"]))
            mesh.compute_vertex_normals()
            material = rendering.MaterialRecord()
            material.shader, material.base_color = "defaultLitTransparency", [.15, .55, 1., .16]
            material.has_alpha = True
            scene.add_geometry(f"overlap_distance_range_{edge['edge_id']}", mesh, material)
            theta = np.linspace(0., 2 * np.pi, 48, endpoint=False)
            ring_offsets = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
            for endpoint, center in enumerate((edge["start_mm"], edge["end_mm"])):
                ring = center + ring_offsets
                lines(f"overlap_range_end_{edge['edge_id']}_{endpoint}",
                      [point for i in range(len(ring)) for point in (ring[i], ring[(i + 1) % len(ring)])],
                      [.15, .55, 1.], 1.)
    if step == 1:
        for patch in result["plane_patches"]:
            vertices = patch["hull_vertices_mm"]
            lines(f"process_hull_{patch['plane_id']}",
                  [point for i in range(len(vertices))
                   for point in (vertices[i], vertices[(i + 1) % len(vertices)])], [.25, .25, .25], 1.)
    selected = {identifier for corner in result["corners"] for identifier in corner["edge_ids"]}
    displayed_edges = ([edge for edge in active if edge["edge_id"] in selected] if step == 4
                       else original if step == 1 else active)
    for edge in displayed_edges:
        shared = edge["edge_type"] == "plane_intersection"
        color = ([0., .9, 1.] if shared else
                 np.eye(3)[edge["plane_ids"][0] % 3].tolist() if step == 1 else [.5, .5, .5])
        if step in (3, 4) and edge["edge_id"] in selected:
            color = [0., .9, 1.] if step == 4 and shared else [.1, 1., .2]
        length = float(np.linalg.norm(edge["end_mm"] - edge["start_mm"]))
        ranges = [(0., length)] if step == 1 else edge.get("active_intervals_mm", [(0., length)])
        intervals(f"process_edge_{edge['edge_id']}", edge, ranges, color,
                  5. if step in (3, 4) and edge["edge_id"] in selected else 3.)
    if step == 2:
        for edge in original:
            intervals(f"removed_boundary_{edge['edge_id']}", edge,
                      edge.get("masked_intervals_mm", ()), [1., .15, 1.], 4., dashed=True)
    if step not in (3, 4):
        return geometry_names
    by_id = {edge["edge_id"]: edge for edge in active}
    for index, corner in enumerate(result["corners"]):
        position = corner["position"]
        for identifier, side in corner["endpoint_ids"].items():
            endpoint = by_id[identifier]["start_mm" if side == 0 else "end_mm"]
            lines(f"endpoint_gap_c{index}_e{identifier}", [endpoint, position], [1., .8, .1], 2.)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=.3)
        marker.translate(position)
        material = rendering.MaterialRecord()
        material.shader, material.base_color = "defaultUnlit", [.1, 1., .2, 1.]
        name = f"corner_candidate_{index}"
        scene.add_geometry(name, marker, material)
        geometry_names.append(name)
        points, indices = _wire_sphere(position, 1.6, 32)
        lines(f"corner_halo_{index}", [points[i] for pair in indices for i in pair], [.1, 1., .2], 1.)
    return geometry_names


def _add_validator_roi_debug(widget, result, view, label_handles):
    """Draw stored clipping and endpoint provenance; no hull or line is recomputed here."""
    scene, rendering = widget.scene, o3d.visualization.rendering
    snapshot = result["roi_debug_before"] if view == "D-before" else result
    patches = {patch["plane_id"]: patch for patch in snapshot["plane_patches"]}
    debug = snapshot["boundary_debug"]

    def lines(name, points, color, width=2., dashed=False):
        segments = []
        for start, end in points:
            length = np.linalg.norm(end - start)
            if length <= 1e-10:
                continue
            pieces = [(0., length)] if not dashed else [(t, min(t + 2., length)) for t in np.arange(0., length, 4.)]
            segments.extend(start + t / length * (end - start) for piece in pieces for t in piece)
        if not segments:
            return
        wire = o3d.geometry.LineSet()
        wire.points = o3d.utility.Vector3dVector(segments)
        wire.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, len(segments), 2)])
        wire.paint_uniform_color(color)
        material = rendering.MaterialRecord()
        material.shader, material.line_width = "unlitLine", width
        scene.add_geometry(name, wire, material)

    def nodes(name, items, patch, active_color):
        active = [node for node in items if node["corner_enabled"]]
        if active:
            mesh = o3d.geometry.TriangleMesh()
            for node in active:
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.35, result["ransac_voxel_mm"] * .12), resolution=6)
                marker.translate(node["position_3d"])
                mesh += marker
            material = rendering.MaterialRecord()
            material.shader, material.base_color = "defaultUnlit", active_color + [1.]
            scene.add_geometry(name + "_active", mesh, material)
        cross = []
        for node in items:
            if node["corner_enabled"]:
                continue
            for direction in (patch["basis_u"] + patch["basis_v"], patch["basis_u"] - patch["basis_v"]):
                cross.append((node["position_3d"] - direction, node["position_3d"] + direction))
        lines(name + "_inactive", cross, [1., .5, 0.], 3.)

    if view == "A":
        for name, radius, color in (("core", result["radius_mm"], [.1, .5, 1.]),
                                    ("padded", result["support_radius_mm"], [1., .5, 0.])):
            points, indices = _wire_sphere(result["query_position_mm"], radius, 64)
            lines("debug_" + name, [(points[a], points[b]) for a, b in indices[::2]], color, 1.)
        ids = sorted({int(index) for patch in patches.values() for index in patch["support_ransac_indices"]})
        if ids:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(result["ransac_points_mm"][ids])
            cloud.paint_uniform_color([.55, .55, .55])
            material = rendering.MaterialRecord()
            material.shader, material.point_size = "defaultUnlit", 2.
            scene.add_geometry("debug_plane_inliers", cloud, material)
    for patch_debug in debug:
        patch = patches[patch_debug["plane_id"]]
        name = f"roi_p{patch['plane_id']}"
        if view in ("A", "B") or view.startswith("D-"):
            lines(name + "_hull", [(edge["start_mm"], edge["end_mm"]) for edge in patch_debug["hull_edges"]],
                  [.1, .9, .2] if view == "A" else [.45, .45, .45], 2. if view == "A" else 1., dashed=view != "A")
        if view == "A":
            nodes(name, patch_debug["original_nodes"], patch, [.1, .9, .2])
            continue
        if view in ("B", "C"):
            lines(name + "_clipped", [(edge["start_mm"], edge["end_mm"]) for edge in patch_debug["clipped_edges"]],
                  [.1, .5, 1.] if view == "B" else [.5, .5, .5], 3. if view == "B" else 1.)
        if view == "B":
            removed = []
            for edge in patch_debug["hull_edges"]:
                start, end = edge["start_mm"], edge["end_mm"]
                interval = edge["clip_interval_mm"]
                if interval is None:
                    removed.append((start, end))
                else:
                    length = np.linalg.norm(end - start)
                    direction = (end - start) / length
                    removed.extend(((start, start + interval[0] * direction),
                                    (start + interval[1] * direction, end)))
            lines(name + "_removed", removed, [.8, .35, .35], 2., dashed=True)
            nodes(name, list(patch_debug["nodes"].values()), patch, [.1, .5, 1.])
            continue
        final = [line for line in patch_debug["lines"] if line["kept_in_graph"]]
        lines(name + "_tls", [(line["start_mm"], line["end_mm"]) for line in final], [.1, .5, 1.], 4.)
        endpoints = {line[side + "_node_id"]: {**line[side + "_node"], "position_3d": line[side + "_mm"]}
                     for line in final for side in ("start", "end")}
        nodes(name, list(endpoints.values()), patch, [.1, .5, 1.])
        if view == "C":
            absorbed = {identifier for line in final for identifier in line["support_node_ids"][1:-1]} - endpoints.keys()
            if absorbed:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector([patch_debug["nodes"][identifier]["position_3d"] for identifier in sorted(absorbed)])
                cloud.paint_uniform_color([.6, .6, .6])
                material = rendering.MaterialRecord()
                material.shader, material.point_size = "defaultUnlit", 4.
                scene.add_geometry(name + "_absorbed", cloud, material)
            for line in final:
                label_handles.append(widget.add_3d_label(
                    (line["start_mm"] + line["end_mm"]) * .5,
                    f"{line['id']} | TLS RMS error {line['tls_rms_error_mm']:.3f} mm"))
    if view.startswith("D-"):
        for index, candidate in enumerate(snapshot["corners"]):
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=.7, resolution=6)
            marker.translate(candidate["position"])
            material = rendering.MaterialRecord()
            material.shader, material.base_color = "defaultUnlit", [.9, .9, .9, 1.]
            scene.add_geometry(f"debug_corner_candidate_{index}", marker, material)
    enabled = [node["corner_enabled"] for patch in debug for node in patch["nodes"].values()]
    endpoints = {node["id"]: node for patch in debug for line in patch["lines"] if line["kept_in_graph"]
                 for node in (line["start_node"], line["end_node"])}
    return (f"Clipped nodes A/I: {sum(enabled)}/{len(enabled) - sum(enabled)} | "
            f"edges: {sum(len(patch['clipped_edges']) for patch in debug)}\n"
            f"Final lines: {sum(line['kept_in_graph'] for patch in debug for line in patch['lines'])} | "
            f"endpoints A/I: {sum(node['corner_enabled'] for node in endpoints.values())}/"
            f"{sum(not node['corner_enabled'] for node in endpoints.values())} | "
            f"corner candidates: {len(snapshot['corners'])}")


def save_guided_validator_roi_views(display):
    """Capture the five existing scenes into A/B/C and a paired Before/After PNG."""
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo
    from PIL import Image, ImageDraw

    stage = next((stage for stage in display["geometry_windows"] if stage["number"] == 4 and not stage["closed"]), None)
    if stage is None or stage.get("query_index") is None:
        return None
    directory = Path("outputs/validator_roi") / f"query_{stage['query_index']}_{datetime.now(ZoneInfo('Asia/Seoul')):%Y%m%d_%H%M%S_%f}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[validator ROI] Cannot create PNG directory: {exc}", flush=True)
        return None
    captures = {}
    captions = {panel["roi_debug_view"]: panel["label"].text + panel.get("png_line_errors", "") for panel in stage["panels"]}

    def captured(panel, rendered):
        picture = Image.fromarray(np.asarray(rendered)).convert("RGB")
        text = captions[panel["roi_debug_view"]]
        header = 16 * (text.count("\n") + 1) + 12
        image = Image.new("RGB", (picture.width, picture.height + header), (30, 30, 30))
        image.paste(picture, (0, header))
        ImageDraw.Draw(image).multiline_text((6, 5), text, fill="white", spacing=4)
        captures[panel["roi_debug_view"]] = image
        if len(captures) != len(stage["panels"]):
            return
        try:
            for view in ("A", "B", "C"):
                captures[view].save(directory / f"{view}.png")
            before, after = captures["D-before"], captures["D-after"]
            comparison = Image.new("RGB", (before.width + after.width, max(before.height, after.height)), (30, 30, 30))
            comparison.paste(before, (0, 0))
            comparison.paste(after, (before.width, 0))
            comparison.save(directory / "D_before_after.png")
        except OSError as exc:
            print(f"[validator ROI] PNG save failed: {exc}", flush=True)
            return
        print(f"[validator ROI] PNG saved: {directory.resolve()}", flush=True)

    for panel in stage["panels"]:
        panel["widget"].scene.scene.render_to_image(lambda image, panel=panel: captured(panel, image))
    return directory


def update_validator_geometry_view(display, result):
    """Read stored results into pair-edge, per-plane boundary and integrated views."""
    for stage in display["geometry_windows"]:
        if stage["closed"]:
            continue
        number, edge_type = stage["number"], stage["edge_type"]
        if number == 4 and not result.get("geometry_computed"):
            stage["query_index"] = None
        for panel in stage["panels"]:
            widget, label, plane_id = panel["widget"], panel["label"], panel["plane_id"]
            scene = widget.scene
            if number == 6 and panel["graph_step"] == 4:
                for name in panel["overlay_geometry_names"]:
                    scene.remove_geometry(name)
                panel["overlay_geometry_names"] = []
            else:
                scene.clear_geometry()
            if number == 4:
                for handle in panel.get("labels_3d", ()):
                    widget.remove_3d_label(handle)
                panel["labels_3d"] = []
            if not result.get("planes_computed", True) or not result.get("geometry_computed"):
                label.text = (f"{number}. NOT_COMPUTED | select a Coarse query"
                              if not result.get("planes_computed", True) else
                              f"{number}. NOT_COMPUTED | geometry validation disabled")
            else:
                if number == 4:
                    view = panel["roi_debug_view"]
                    counts = _add_validator_roi_debug(widget, result, view, panel["labels_3d"])
                    title, legend = {
                        "A": ("A. Padded Hull", "Blue dashed: ROI | orange dashed: padding | green: hull/nodes"),
                        "B": ("B. ROI Clipping", "Blue: retained/active | orange X: clipped | red dashed: removed"),
                        "C": ("C. TLS Line Clustering", "Blue: final lines/active | orange X: inactive | gray dots: absorbed"),
                        "D-before": ("D. Before: core-only Hull", "Same planes/core voxel samples | white: corner candidates"),
                        "D-after": ("D. After: padded Hull -> clipping -> TLS", "Same planes/core voxel samples | orange X: inactive | white: candidates"),
                    }[view]
                    label.text = f"{title} | query {result['query_index']}\n{counts}\n{legend}"
                    panel["png_line_errors"] = ("\n" + "\n".join(
                        f"{line['id']}: TLS RMS error {line['tls_rms_error_mm']:.3f} mm"
                        for patch in result["boundary_debug"] for line in patch["lines"] if line["kept_in_graph"])) if view == "C" else ""
                    if panel.get("query_index") != result["query_index"]:
                        query, radius = result["query_position_mm"], result["support_radius_mm"]
                        bounds = o3d.geometry.AxisAlignedBoundingBox(query - radius, query + radius)
                        widget.setup_camera(60., bounds, query)
                        scene.camera.look_at(query, query + radius * np.array([1.7, 1.3, 1.5]), [0., 0., 1.])
                        panel["query_index"] = result["query_index"]
                    stage["query_index"] = result["query_index"]
                    widget.force_redraw()
                    continue
                edges = [edge for edge in result["edges"]
                         if (edge_type is None or edge["edge_type"] == edge_type)
                         and (plane_id is None or plane_id in edge["plane_ids"])]
                if edge_type == "plane_intersection":
                    widths = sorted({pair["strip_half_width_mm"]
                                     for pair in result.get("plane_pair_diagnostics", ())})
                    strip_label = "/".join(f"{width:g}" for width in widths) if widths else "N/A"
                    label.text = (f"{number}. Plane-pair intersections | {len(edges)} edges | strip +/-{strip_label} mm\n"
                                  "Yellow: infinite line (local span) | RGB translucent: strips | cyan: finite edge | gray: hull / unsupported")
                else:
                    step = panel["graph_step"]
                    original = result.get("all_edges", result["edges"])
                    shared = sum(edge["edge_type"] == "plane_intersection" for edge in original)
                    masked = sum(bool(edge.get("masked_intervals_mm")) for edge in original)
                    if step == 1:
                        label.text = (f"6-1. Original edges | query {result['query_index']}\n"
                                      f"{len(original) - shared} boundary | {shared} shared edges\n"
                                      "RGB: plane boundaries | cyan: shared edges")
                    elif step == 2:
                        overlap = result.get("edge_overlap_settings")
                        criteria = (f"Blue radius: {overlap['distance_mm']:g} mm | angle <= {overlap['angle_deg']:g} deg | "
                                    f"overlap >= {overlap['min_length_mm']:g} mm" if overlap is not None else
                                    "Overlap distance settings unavailable")
                        label.text = (f"6-2. Overlap removal | {masked} boundaries masked\n"
                                      f"{criteria}\n"
                                      "Magenta dashed: removed | gray: kept | cyan: shared")
                    elif step == 3:
                        label.text = (f"6-3. Endpoint Corner integration | {result['validation_status']}\n"
                                      f"{len(result['corners'])} unverified Corners | endpoint/merge radius {result['corner_radius_mm']:g} mm\n"
                                      "Green: candidates | yellow: selected endpoint gaps | no refinement")
                    else:
                        background = (f"{panel['raw_point_count']} original raw points" if panel["raw_point_count"] else
                                      "original raw cloud unavailable")
                        used = {identifier for corner in result["corners"] for identifier in corner["edge_ids"]}
                        shared_ids = [edge["edge_id"] for edge in result["edges"]
                                      if edge["edge_id"] in used and edge["edge_type"] == "plane_intersection"]
                        boundary_ids = [edge["edge_id"] for edge in result["edges"]
                                        if edge["edge_id"] in used and edge["edge_type"] == "boundary"]
                        label.text = (f"6-4. All Corner candidates on original cloud | query {result['query_index']}\n"
                                      f"{len(result['corners'])} Corners | boundary {boundary_ids} | shared {shared_ids} | {background}\n"
                                      "Gray: raw | green: unverified corners / boundary | cyan: shared | yellow: endpoint gap")
                if number == 6:
                    panel["overlay_geometry_names"] = _add_validator_graph_step(scene, result, panel["graph_step"])
                else:
                    _add_validator_corner_geometry(scene, result, edge_type=edge_type, plane_id=plane_id)
                query, radius = result["query_position_mm"], result["support_radius_mm"]
                if number == 6:
                    radius += result.get("corner_radius_mm", 0.)
                if panel.get("query_index") != result["query_index"]:
                    bounds = o3d.geometry.AxisAlignedBoundingBox(query - radius, query + radius)
                    widget.setup_camera(60.0, bounds, query)
                    scene.camera.look_at(query, query + radius * np.array([1.7, 1.3, 1.5]), [0., 0., 1.])
                    panel["query_index"] = result["query_index"]
            widget.force_redraw()
        stage["window"].post_redraw()


def update_guided_validator_view(display, result):
    """Replace only small ROI geometry; keep the overview and its camera intact."""
    rendering = o3d.visualization.rendering
    update_validator_roi_view(display, result)
    update_validator_geometry_view(display, result)
    widget = display["local"]
    scene = widget.scene
    scene.clear_geometry()
    if not result.get("planes_computed", True):
        display["labels"][2].text = "3. Plane fitting | select a Coarse query"
        widget.force_redraw()
        display["window"].post_redraw()
        return
    plane_counts = " | ".join(
        f"P{i} ({color}): {plane.get('inlier_count', '?')}"
        for i, (plane, color) in enumerate(zip(result["planes"], ("red", "green", "blue")))
    )
    display["labels"][2].text = (
        f"3. Plane fitting | {len(result['planes'])} planes | min inliers: {result.get('min_required', '?')}\n"
        + (plane_counts or "No accepted planes")
    )
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
    material.base_color = [1.0, 0.5, 0.0, 1.0]
    scene.add_geometry("coarse_query", marker, material)
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
        material.base_color = color + [0.60]
        material.has_alpha = True
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
    bounds = o3d.geometry.AxisAlignedBoundingBox(query - radius, query + radius)
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
    result = {
        "ced_score": ced_shared,
        "bilateral_ced_score": ced_shared,
        "pca_ratio_saturated": pca,
        "fusion_score": fusion,
        "fusion_score_raw": fusion,
    }
    if score_result.get("fusion_mode") == "local_minimum":
        result["local_minimum_score"] = valid_quantile("local_minimum_score", "local_minimum_valid_mask")
    return result


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


def fusion_rejection_primitives(points_mm, score_result):
    """Color cached threshold/NMS/post-gate outcomes without rerunning a stage."""
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    extraction = score_result["candidate_extraction"]
    scores = np.asarray(score_result["fusion_score"], dtype=float)
    valid = np.asarray(score_result["fusion_valid_mask"], dtype=bool)
    if scores.shape != (len(points),) or valid.shape != scores.shape:
        raise ValueError("Fusion score/validity must align with points_mm")
    valid = valid & np.isfinite(scores)
    eligible = np.asarray(extraction.get("eligible_score_mask",
        valid & (scores > extraction["fusion_score_threshold"])), dtype=bool)
    if eligible.shape != scores.shape:
        raise ValueError("Fusion eligibility must align with points_mm")
    indices = np.asarray(extraction.get("nms_candidate_indices", extraction["candidate_indices"]), dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("NMS indices must be valid scene indices")
    nms = np.zeros(len(points), dtype=bool)
    nms[indices] = True
    gate = extraction.get("post_nms_lambda1_gate")
    passed, invalid = np.ones(len(indices), dtype=bool), np.zeros(len(indices), dtype=bool)
    if gate is not None and gate["enabled"]:
        passed = np.asarray(gate["passed_mask"], dtype=bool)
        invalid = np.asarray(gate["invalid_mask"], dtype=bool)
        if (passed.shape != indices.shape or invalid.shape != indices.shape
                or not np.array_equal(indices, gate["nms_indices"])):
            raise ValueError("Post-NMS gate masks must align with raw NMS indices")
    ratio_failed, pca_invalid, gate_passed = (np.zeros(len(points), dtype=bool) for _ in range(3))
    ratio_failed[indices] = ~passed & ~invalid
    pca_invalid[indices] = invalid
    gate_passed[indices] = passed
    groups = (
        ("invalid_score", ~valid, (.16, .16, .18)),
        ("fusion_threshold", valid & ~eligible, (.10, .40, 1.)),
        ("nms_suppressed", eligible & ~nms, (.40, .40, .42)),
        ("lambda1_upper", ratio_failed, (1., .05, .75)),
        ("invalid_weighted_pca", pca_invalid, (1., .55, .05)),
        ("gates_passed", gate_passed, (.10, 1., .30)),
    )
    colors = np.empty((len(points), 3))
    counts = {}
    for name, mask, color in groups:
        colors[mask] = color
        counts[name] = int(np.count_nonzero(mask))
    threshold = extraction["fusion_score_threshold"]
    gate_text = (f"weighted lambda1/R_PCA² > {gate['threshold']:g}"
                 if gate is not None and gate["enabled"] else "lambda1 gate OFF")
    legend = (f"BLUE: Fusion <= {threshold:g} ({counts['fusion_threshold']}) | "
              f"MAGENTA: {gate_text} ({counts['lambda1_upper']}) | "
              f"ORANGE: invalid weighted PCA ({counts['invalid_weighted_pca']}) | "
              f"GREEN: both gates passed, pre-Q10 ({counts['gates_passed']}) | "
              f"GRAY: NMS suppressed ({counts['nms_suppressed']}) | "
              f"DARK: invalid Fusion ({counts['invalid_score']})")
    return {"indices": indices.copy(), "positions_mm": points[indices].copy(),
            "marker_colors": colors[indices].copy(), "point_colors": colors,
            "counts": counts, "legend": legend}


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
    intrinsic = np.asarray(parameters.intrinsic.intrinsic_matrix, dtype=np.float64)
    extrinsic = np.asarray(parameters.extrinsic, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or abs(np.linalg.det(extrinsic[:3, :3])) <= 1e-12):
        raise ValueError("Invalid Open3D camera matrices")
    return np.concatenate((intrinsic.ravel(), extrinsic.ravel()))


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
        try:
            parameters[index] = vis.get_view_control().convert_to_pinhole_camera_parameters()
            signatures[index] = _camera_signature(parameters[index])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue

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
        if signatures[index] is None:
            continue
        try:
            restored = vis.get_view_control().convert_from_pinhole_camera_parameters(master_parameters, allow_arbitrary=True)
            if restored is not False:
                signatures[index] = master_signature.copy()
        except (TypeError, ValueError, RuntimeError):
            continue
    return signatures, master_index


def dense_heatmap_panels(bilateral_mode="ced", fusion_mode="pca_ratio"):
    """Return six synchronized panels for the selected bilateral stage."""
    bilateral_panel = ("Bilateral CED", "bilateral_ced_score", "bilateral_valid_mask", "CED range")
    if bilateral_mode == "fusion":
        bilateral_panel = ("Raw Fusion (before bilateral)", "fusion_score_raw", "fusion_valid_mask", "Fusion Q99")
    elif bilateral_mode == "off":
        bilateral_panel = ("Original CED (bilateral OFF)", "bilateral_ced_score", "ced_valid_mask", "CED range")
    factor_panel = ("PCA Ratio Used for Fusion", "pca_ratio_saturated", "pca_valid_mask", "PCA range")
    if fusion_mode == "local_minimum":
        factor_panel = ("Local lambda1 Minimum (Min-Max)", "local_minimum_score",
                        "local_minimum_valid_mask", "Local minimum range")
    return (
        ("Original Point Cloud", None, None, "geometry"),
        ("Point-wise CED", "ced_score", "ced_valid_mask", "CED range"),
        bilateral_panel,
        factor_panel,
        (
            "Bilateral Fusion Score" if bilateral_mode == "fusion" else "Final Fusion Score",
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
    lite_debug_callback=None,
):
    """Show score fields and optional registered RGB in synchronized views."""
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    lite = score_result.get("lite_validator")
    if only_nms and lite is None:
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
    rejection_debug = fusion_rejection_primitives(points, score_result)
    print("[NMS GATE COLORS] " + rejection_debug["legend"], flush=True)
    panels = dense_heatmap_panels(score_result.get("bilateral_mode", "ced"),
                                  score_result.get("fusion_mode", "pca_ratio"))
    panel_indices = (5,) if only_nms else range(len(panels))
    windows = []
    final_rgb = lite is not None and rgb_points is not None
    q10 = lite is not None and lite.get("kind") == "q10_antipodal"
    cone = lite is not None and lite.get("kind") == "q10_antipodal"
    interactive_lite = lite is not None and (lite["debug"] or cone)
    result_view_state = {}
    inspector = None
    pending_record = None

    def inspect_candidate(record):
        nonlocal pending_record
        pending_record = record

    panel_count = len(panel_indices) + int(rgb_points is not None and not final_rgb)

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

    def build_candidate_markers(debug=candidate_debug):
        if not len(debug["indices"]):
            return None
        markers = o3d.geometry.TriangleMesh()
        colors = debug.get("marker_colors", np.tile([1.0, 0.05, 0.03], (len(debug["indices"]), 1)))
        for position, color in zip(debug["positions_mm"], colors):
            marker = o3d.geometry.TriangleMesh.create_sphere(
                radius=max(0.4, float(marker_radius_mm)),
                resolution=6,
            )
            marker.translate(position)
            marker.paint_uniform_color(color)
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
        if key == "pca_ratio_saturated":
            tau = score_result["pca_ratio_saturation_tau"]
            ratio_mode = "raw identity" if tau is None else f"saturated tau={tau}"
            title += (
                f" | queried={np.count_nonzero(score_result['pca_query_mask'])} "
                f"invalid={np.count_nonzero(score_result['invalid_support_mask'])} "
                f"mode={ratio_mode} "
            )
        elif key == "local_minimum_score":
            title += f" | Normal PCA lambda1 | R={score_result['pca_radius_mm']:g} mm | tau={score_result['local_minimum_tau']:g}"
        if panel_index == 5:
            title = "6 NMS Corner Candidates | gate reasons | " + rejection_debug["legend"] + " | M: markers ON/OFF"
            if interactive_lite:
                title += " | Left click green marker: Lite debug"
        vis = (
            o3d.visualization.VisualizerWithKeyCallback()
            if panel_index == 5 or cone
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
        if cone:
            from .query_ui import bind_safe_camera_keys
            bind_safe_camera_keys(vis)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        ratio_colors = None
        if panel_index == 5:
            cloud.colors = o3d.utility.Vector3dVector(rejection_debug["point_colors"])
        elif key is None:
            cloud.colors = o3d.utility.Vector3dVector(
                np.full((len(points), 3), 0.72, dtype=np.float64)
            )
        elif key == "pca_ratio_saturated":
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
            markers = build_candidate_markers(rejection_debug)
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
            if interactive_lite and lite_debug_callback is not None:
                from .query_ui import bind_lite_validator_click
                bind_lite_validator_click(vis, lite["records"], inspect_candidate)
        option = vis.get_render_option()
        if option is not None:
            option.background_color = np.array([0.025, 0.025, 0.025])
            option.point_size = 4.0
        windows.append(vis)

    if rgb_points is not None and not final_rgb:
        rgb_panel_index = len(panels)
        title = (
            "RealSense RGB + corner candidates (auxiliary, not a heatmap) | "
            f"points={len(rgb_points)} | candidates={len(candidate_debug['indices'])}"
        )
        vis = o3d.visualization.VisualizerWithKeyCallback() if cone else o3d.visualization.Visualizer()
        if not vis.create_window(
            window_name=title,
            **window_layout(rgb_panel_index),
        ):
            for existing in windows:
                existing.destroy_window()
            raise RuntimeError("Failed to create RealSense RGB candidate window")
        if cone:
            from .query_ui import bind_safe_camera_keys
            bind_safe_camera_keys(vis)
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

    try:
        if lite is not None:
            if q10:
                converged_view = open_q10_convergence_results(points, score_result, marker_radius_mm)
                windows.append(converged_view)
                if interactive_lite and lite_debug_callback is not None:
                    from .query_ui import bind_lite_validator_click
                    bind_lite_validator_click(converged_view, lite["records"], inspect_candidate,
                                               position_key="refined_position_mm")
            view_options = {"state": result_view_state} if cone else {}
            if final_rgb:
                view_options.update(rgb_points_mm=rgb_points, rgb_colors_rgb=rgb_colors)
            windows.append(open_lite_validator_results(points, score_result, marker_radius_mm, **view_options))
            if interactive_lite and lite_debug_callback is not None:
                from .query_ui import bind_lite_validator_click
                bind_lite_validator_click(windows[-1], lite["records"] if q10 and not final_rgb else
                    [r for r in lite["records"] if r["status"] == "ACCEPT"], inspect_candidate,
                    **({"position_key": "refined_position_mm"} if q10 else {}))
                print("[LITE DEBUG] Ready: left-click an NMS marker in the NMS or Lite Validator window.", flush=True)
        active = [True] * len(windows)
        camera_signatures, _master = synchronize_dense_camera_views(windows, active, None)
        while any(active):
            for index, vis in enumerate(windows):
                if active[index]:
                    active[index] = bool(vis.poll_events())
            # Construct/poll Tk after GLFW has finished dispatching mouse callbacks.
            if pending_record is not None:
                if inspector is not None:
                    inspector["close"]()
                inspector = lite_debug_callback(pending_record)
                if cone and active[-1]:
                    refresh_lite_markers(result_view_state, lite, marker_radius_mm)
                pending_record = None
            if inspector is not None and not inspector["poll"]():
                inspector = None
            if not any(active):
                break
            camera_signatures, _master = synchronize_dense_camera_views(windows, active, camera_signatures)
            for index, vis in enumerate(windows):
                if active[index]:
                    vis.update_renderer()
    finally:
        if inspector is not None:
            inspector["close"]()
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


def open_total_result_views(points_mm, raw_points_mm, result, *, validator_voxel_mm):
    """Create exactly two views; batch final meshes by colour to reduce draw calls."""
    points, raw = np.asarray(points_mm), np.asarray(raw_points_mm)
    scores, validation = result["scores"], result["validation"]
    upper = dense_heatmap_vmax(scores)["fusion_score"]
    heatmap = o3d.geometry.PointCloud()
    heatmap.points = o3d.utility.Vector3dVector(points)
    heatmap.colors = o3d.utility.Vector3dVector(dense_heatmap_colors(
        scores["fusion_score"], scores["fusion_valid_mask"], vmax=upper))
    seeds = o3d.geometry.TriangleMesh()
    for index in validation["candidate_indices"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=2., resolution=6)
        marker.translate(points[index])
        seeds += marker
    seeds.paint_uniform_color([1., .05, .03])
    if len(seeds.vertices):
        seeds.compute_vertex_normals()
    background = o3d.geometry.PointCloud()
    background.points = o3d.utility.Vector3dVector(raw)
    background.paint_uniform_color([.38, .38, .38])
    meshes = {name: o3d.geometry.TriangleMesh() for name in ("boundary", "plane_intersection", "endpoint_gap")}
    colours = {"boundary": [.1, 1., .2], "plane_intersection": [0., .9, 1.],
               "endpoint_gap": [1., .8, .1]}
    edge_radius = max(.5, .15 * validator_voxel_mm)
    for edge in validation["final_edges"]:
        delta = edge["end_mm"] - edge["start_mm"]
        length = float(np.linalg.norm(delta))
        if length <= 1e-8:
            continue
        direction = delta / length
        for low, high in edge["active_intervals_mm"]:
            bar = _segment_cylinder(edge["start_mm"] + low * direction,
                                    edge["start_mm"] + high * direction,
                                    edge_radius, colours[edge["edge_type"]])
            if bar is not None:
                meshes[edge["edge_type"]] += bar
        endpoint = edge["start_mm" if edge["endpoint_id"] == 0 else "end_mm"]
        bar = _segment_cylinder(endpoint, edge["corner_position_mm"], .6 * edge_radius, colours["endpoint_gap"])
        if bar is not None:
            meshes["endpoint_gap"] += bar
    corners = o3d.geometry.TriangleMesh()
    for corner in validation["final_corners"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(2., .8 * validator_voxel_mm), resolution=10)
        marker.translate(corner["position_mm"])
        corners += marker
    corners.paint_uniform_color([1., .05, .03])
    if len(corners.vertices):
        corners.compute_vertex_normals()
    diagnostics = validation["diagnostics"]
    titles = (
        f"TOTAL | 1. Detector Fusion heatmap 0..{upper:.4g} (Q99) | NMS={diagnostics['candidate_count']} | R: recapture Q/Esc: quit",
        f"TOTAL | 2. Final corners={len(validation['final_corners'])} edges={len(validation['final_edges'])} | "
        "gray=raw red=unverified corner green=boundary cyan=shared yellow=endpoint gap | R: recapture Q/Esc: quit",
    )
    contents = ((heatmap, [seeds]), (background, [*meshes.values(), corners]))
    windows = []
    try:
        for index, (cloud, overlays) in enumerate(contents):
            vis = o3d.visualization.VisualizerWithKeyCallback()
            if not vis.create_window(window_name=titles[index], width=920, height=800, left=15 + index * 940, top=45):
                raise RuntimeError("Failed to create total-mode result window")
            windows.append(vis)
            vis.add_geometry(cloud)
            for mesh in overlays:
                if len(mesh.vertices):
                    vis.add_geometry(mesh, reset_bounding_box=False)
            option = vis.get_render_option()
            option.background_color = np.asarray([.08, .08, .08])
            option.point_size, option.light_on = (4. if index == 0 else 1.5), False
            control = vis.get_view_control()
            control.set_lookat(heatmap.get_axis_aligned_bounding_box().get_center())
            control.set_front(normalize([.75, .55, .65]))
            control.set_up(normalize([0., -1., 0.]))
            control.set_zoom(.72)
        return windows
    except BaseException:
        for vis in windows:
            vis.destroy_window()
        raise


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
    "fusion_rejection_primitives",
    "cornervins_ahc_primitives",
    "cornervins_scene_roi_primitives",
    "synchronize_dense_camera_views",
    "dense_heatmap_panels",
    "show_dense_score_heatmaps",
    "open_lite_validator_results", "pick_lite_validator_candidate",
    "refresh_lite_markers",
    "open_total_result_views",
    "open_cornervins_plane_views",
    "plane_edge_refiner_primitives",
    "show_detector_validator_results",
    "open_plane_edge_graph_view",
    "open_plane_edge_refiner_view",
]


def _lite_record_color(record):
    return {"ACCEPT": [.1, .9, .25], "REJECT": [1., .08, .12]}.get(record["status"], [1., .65, .05])


def _lite_marker_mesh(result, marker_radius_mm, *, accepted_only=False):
    markers = o3d.geometry.TriangleMesh()
    for record in result["records"]:
        if accepted_only and record["status"] != "ACCEPT":
            continue
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.4, marker_radius_mm), resolution=6)
        marker.translate(record["refined_position_mm"])
        marker.paint_uniform_color([.1, .9, .25] if accepted_only else _lite_record_color(record))
        markers += marker
    if not markers.is_empty():
        markers.compute_vertex_normals()
    return markers


def refresh_lite_markers(state, result, marker_radius_mm):
    """Refresh only display geometry after an explicit numerical click evaluation."""
    vis = state["view"]
    vis.remove_geometry(state["markers"], reset_bounding_box=False)
    state["markers"] = _lite_marker_mesh(result, marker_radius_mm, accepted_only=state.get("accepted_only", False))
    if not state["markers"].is_empty():
        vis.add_geometry(state["markers"], reset_bounding_box=False)
    vis.update_renderer()


def open_lite_validator_results(points_mm, score_result, marker_radius_mm, *, state=None,
                                rgb_points_mm=None, rgb_colors_rgb=None):
    """One extra full-scene window; existing NMS panels and positions are untouched."""
    result = score_result["lite_validator"]
    final_rgb = rgb_points_mm is not None
    vis = o3d.visualization.VisualizerWithKeyCallback()
    title = (f"3 Final CC + antipodal | green PASS={result['accept_count']} / red REJECT={result['reject_count']} "
             f"/ amber UNKNOWN={result['indeterminate_count']} | click any marker")
    if final_rgb:
        title = (f"Final RealSense RGB point cloud | green PASS corners: {result['accept_count']} "
                 f"/ {result['candidate_count']} | click: cached debug")
    if not vis.create_window(window_name=title, width=650, height=500, left=640, top=45):
        raise RuntimeError("Failed to create Lite Validator result window")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(rgb_points_mm if final_rgb else points_mm)
    cloud.colors = o3d.utility.Vector3dVector(rgb_colors_rgb if final_rgb else dense_heatmap_colors(
        score_result["fusion_score"], score_result["fusion_valid_mask"],
        vmax=dense_heatmap_vmax(score_result)["fusion_score"]))
    vis.add_geometry(cloud)
    markers = _lite_marker_mesh(result, marker_radius_mm, accepted_only=final_rgb)
    if not markers.is_empty():
        vis.add_geometry(markers, reset_bounding_box=False)
    if state is not None:
        state.update(view=vis, markers=markers, accepted_only=final_rgb)
    option = vis.get_render_option()
    option.background_color, option.point_size = np.array([.025, .025, .025]), 4.
    return vis


def open_q10_convergence_results(points_mm, score_result, marker_radius_mm):
    """Intermediate positions in orange; final acceptance has its own window."""
    result = score_result["lite_validator"]
    vis = o3d.visualization.VisualizerWithKeyCallback()
    title = ("2 Q10 convergence | orange: refined positions" if result.get("refine_enabled", True) else
             "2 Q10 refine OFF | orange: original NMS positions")
    if not vis.create_window(window_name=title + " | click: path + MaxGap",
                             width=650, height=500, left=15, top=560):
        raise RuntimeError("Failed to create Q10 convergence view")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_mm)
    cloud.paint_uniform_color([.45, .48, .52])
    vis.add_geometry(cloud)
    markers = o3d.geometry.TriangleMesh()
    for record in result["records"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.4, marker_radius_mm), resolution=6)
        marker.translate(record["refined_position_mm"])
        marker.paint_uniform_color([1., .65, .05])
        markers += marker
    if not markers.is_empty():
        markers.compute_vertex_normals()
        vis.add_geometry(markers, reset_bounding_box=False)
    option = vis.get_render_option()
    option.background_color, option.point_size = np.array([.025, .025, .025]), 4.
    return vis


def q10_debug_voxel_primitives(record, step):
    """One sample per analysis ID, colored by all cached plane memberships."""
    all_ids = np.r_[record["step_analysis_indices"][step], record["roi_analysis_indices"]]
    all_points = np.vstack((record["step_points_mm"][step], record["roi_analysis_points_mm"]))
    ids, first = np.unique(all_ids, return_index=True)
    points = all_points[first].copy()
    classes = np.zeros(len(ids), dtype=np.int8)
    colors = np.tile([.2, .45, .85], (len(ids), 1))
    analysis = record.get("spherical_planes")
    planes = analysis["planes"] if analysis is not None else ()
    memberships = np.zeros((len(ids), len(planes)), dtype=bool)
    retained = np.zeros_like(memberships)
    vector_ids = np.asarray(record.get("vector_analysis_indices", []), dtype=np.int64)
    if len(vector_ids):
        rows = np.searchsorted(ids, vector_ids)
        if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], vector_ids):
            raise ValueError("Scored voxel IDs must belong to the displayed analysis support")
        classes[rows] = 1
        colors[rows] = [.55, .55, .60]
        validation = record.get("plane_antipodal")
        tests = {test["plane_index"]: test for test in validation["tests"]} if validation else {}
        for index, plane in enumerate(planes):
            mask = np.asarray(plane["inlier_mask"], dtype=bool)
            if mask.shape != vector_ids.shape:
                raise ValueError("Plane membership masks must align with stored analysis IDs")
            members = rows[mask]
            memberships[members, index] = True
            test = tests.get(index)
            keep = np.asarray(test["kept_mask"], dtype=bool) if test is not None else np.ones(len(members), dtype=bool)
            if keep.shape != members.shape:
                raise ValueError("Kept angular masks must align with the plane's stored members")
            retained[members[keep], index] = True
            colors[members] = _Q10_PLANE_COLORS[index]
        member_count = memberships.sum(axis=1)
        colors[member_count > 1] = [1., 1., 1.]
        kept = retained.any(axis=1)
        discarded = (member_count > 0) & ~kept
        colors[discarded] *= .3
        classes[kept], classes[discarded] = 2, 3
    return {"analysis_indices": ids, "points_mm": points, "point_colors": colors,
            "plane_memberships": memberships, "retained_memberships": retained,
            "point_classes": classes, "scored_mask": classes > 0,
            "retained_count": int(np.count_nonzero(classes == 2)),
            "discarded_count": int(np.count_nonzero(classes == 3))}


def q10_direction_primitives(record, step):
    """Read the cached selected axis; display a radius-long arrow in world mm."""
    axis = np.asarray(record["trace"]["mean_axes"][step], dtype=float)
    if not np.isfinite(axis).all():
        return None
    origin = record["trace"]["positions_mm"][step].copy()
    return {"origin_mm": origin, "direction": axis.copy(),
            "tip_mm": origin + record["radius_mm"] * axis,
            "mode": record.get("direction_mode", "unit_mean")}


def q10_spherical_plane_primitives(record):
    """Display-only circles, cached kept arcs and per-plane projected directions."""
    from ..spherical_antipodal import _sphere_plane_basis
    analysis = record.get("spherical_planes")
    if analysis is None:
        return []
    validation = record.get("plane_antipodal")
    tests = {test["plane_index"]: test for test in validation["tests"]} if validation else {}
    phi = np.linspace(0., 2*np.pi, 145)
    planes = []
    for index, plane in enumerate(analysis["planes"]):
        u, v = _sphere_plane_basis(plane["normal"])
        test = tests.get(index)
        arc = np.empty((0, 3))
        colors = np.tile(_Q10_PLANE_COLORS[index], (len(plane["angles_deg"]), 1))
        if test is not None:
            colors[~test["kept_mask"]] *= .3
            if test["arc_start_deg"] is not None:
                angles = np.radians(np.linspace(test["arc_start_deg"],
                    test["arc_start_deg"] + test["observed_span_deg"],
                    max(2, int(np.ceil(test["observed_span_deg"])) + 1)))
                arc = np.cos(angles)[:, None]*u + np.sin(angles)[:, None]*v
        planes.append({"plane_index": index, "color": _Q10_PLANE_COLORS[index].copy(),
                       "circle": 1.08 * (np.cos(phi)[:, None]*u + np.sin(phi)[:, None]*v),
                       "projected_points": (1.11 + .025*index) * plane["projected_directions"],
                       "projected_colors": colors, "kept_arc": arc,
                       "angles_deg": plane["angles_deg"].copy(),
                       "residual_rms_deg": plane["residual_rms_deg"],
                       "newly_explained_count": plane["newly_explained_count"],
                       "test": test})
    return planes


def open_q10_debug_view(record):
    """Native Q10 path and each cached plane's cleaned angular support arc."""
    from .query_ui import bind_safe_camera_keys
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name=f"Q10 NMS {record['point_index']} | voxel {record['voxel_mm']:g} mm "
                            f"| refine {'ON' if record.get('refine_enabled', True) else 'OFF'} "
                            f"| cyan arrow: Q10 {record.get('direction_mode', 'unit_mean')} "
                            "| P1 orange / P2 green / P3 purple | white overlap | dim discarded | Left/Right: step",
                             width=1050, height=760, left=35, top=65):
        raise RuntimeError("Cannot open Q10 path debug view")
    bind_safe_camera_keys(vis)
    closed, geometries = False, []
    step = len(record["trace"]["positions_mm"]) - 1

    def cloud(points, colors):
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(points)
        result.colors = o3d.utility.Vector3dVector(colors)
        return result

    def line(points, pairs, color):
        result = o3d.geometry.LineSet()
        result.points = o3d.utility.Vector3dVector(points)
        result.lines = o3d.utility.Vector2iVector(pairs)
        result.paint_uniform_color(color)
        return result

    def set_step(value):
        nonlocal step, geometries
        step = int(np.clip(value, 0, len(record["trace"]["positions_mm"]) - 1))
        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)
        data = q10_debug_voxel_primitives(record, step)
        origin = record["trace"]["positions_mm"][step]
        radius = record["radius_mm"]
        geometries = [cloud(data["points_mm"], data["point_colors"])]
        direction = q10_direction_primitives(record, step)
        if direction is not None:
            u, v, d = make_tangent_basis(direction["direction"])
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=.018 * radius, cone_radius=.05 * radius,
                cylinder_height=.78 * radius, cone_height=.22 * radius, resolution=16)
            arrow.rotate(np.column_stack((u, v, d)), center=np.zeros(3))
            arrow.translate(direction["origin_mm"])
            arrow.paint_uniform_color([.05, 1., 1.])
            arrow.compute_vertex_normals()
            geometries.append(arrow)
        path = np.vstack((record["position"], record["trace"]["positions_mm"]))
        geometries.append(line(path, [(i, i+1) for i in range(len(path)-1)], [.95, .95, .95]))
        for position, color in ((record["position"], [1., .05, .05]), (origin, [1., .65, .05]),
                                 (record["refined_position_mm"], _lite_record_color(record))):
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.35, radius * .025), resolution=8)
            marker.translate(position)
            marker.paint_uniform_color(color)
            marker.compute_vertex_normals()
            geometries.append(marker)
        def cap(center, axis, cutoff, color, scale):
            if not np.isfinite(axis).all() or not np.isfinite(cutoff):
                return
            u, v, d = make_tangent_basis(axis)
            phi = np.linspace(0, 2*np.pi, 73)
            rim = center + scale * (cutoff * d + np.sqrt(max(0., 1-cutoff*cutoff)) *
                                   (np.cos(phi)[:, None]*u + np.sin(phi)[:, None]*v))
            vertices = np.vstack((center, rim))
            pairs = [(i, i+1) for i in range(1, len(vertices)-1)] + [(0, i) for i in range(1, len(vertices), 12)]
            geometries.append(line(vertices, pairs, color))
        cap(origin, record["trace"]["mean_axes"][step], record["trace"]["q10_values"][step], [.05, .85, .85], radius)
        for plane in q10_spherical_plane_primitives(record):
            center = record["refined_position_mm"]
            arc = center + .8*radius*plane["kept_arc"]
            if len(arc):
                geometries.append(line(arc, [(i, i+1) for i in range(len(arc)-1)], plane["color"]))
        for geometry in geometries:
            vis.add_geometry(geometry, reset_bounding_box=False)
        vis.update_renderer()
    try:
        set_step(step)
        vis.reset_view_point(True)
        direction = record["trace"]["mean_axes"][-1]
        if np.isfinite(direction).all():
            u, v, d = make_tangent_basis(direction)
            control = vis.get_view_control()
            control.set_lookat(.5 * (record["position"] + record["refined_position_mm"]))
            control.set_front(normalize(1.2 * u + .8 * v + 1.5 * d))
            control.set_up(v)
            control.set_zoom(.7)
        vis.get_render_option().background_color = np.array([.025, .025, .025])
        vis.get_render_option().point_size = 5.
        vis.register_key_callback(263, lambda _v: (set_step(step-1), False)[1])
        vis.register_key_callback(262, lambda _v: (set_step(step+1), False)[1])
    except Exception:
        vis.destroy_window()
        raise

    def close():
        nonlocal closed
        if not closed:
            closed = True
            vis.destroy_window()

    def poll():
        if closed:
            return False
        if not vis.poll_events():
            close()
            return False
        vis.update_renderer()
        return True
    return {"poll": poll, "close": close, "set_step": set_step}


def pick_lite_validator_candidate(vis, records, pixel_x, pixel_y, radius_px=14., *, position_key="position"):
    """Project unchanged candidate positions using the existing NMS-view camera."""
    if not records:
        return None
    camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
    positions = np.asarray([record[position_key] for record in records])
    xyz = np.column_stack((positions, np.ones(len(positions)))) @ np.asarray(camera.extrinsic).T
    projected = xyz[:, :3] @ np.asarray(camera.intrinsic.intrinsic_matrix).T
    valid = np.isfinite(projected).all(axis=1) & (xyz[:, 2] > 1e-10)
    pixels = np.full((len(records), 2), np.inf)
    pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
    distance2 = np.sum((pixels - [pixel_x, pixel_y]) ** 2, axis=1)
    choices = np.flatnonzero(valid & (distance2 <= radius_px ** 2))
    if not len(choices):
        return None
    index = min(choices, key=lambda i: (distance2[i], xyz[i, 2], i))
    return records[index]
