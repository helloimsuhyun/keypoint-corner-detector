# Point-cloud input package

The three input sources return the same `PointCloudFrame`. All 3-D coordinates
are `float32` millimeters in the camera frame: +X right, +Y down, +Z forward.

```python
from pointcloud_inputs import (
    capture_realsense_point_cloud,
    capture_stl_point_cloud,
    load_kinect_point_cloud,
)
```

## 1. STL virtual camera

```python
frame = capture_stl_point_cloud(
    "data/laser_pose_box.stl",
    stl_unit_mm=1.0,
    output_ply="outputs/input/stl.ply",
)
```

Use the mouse to set the virtual camera, then press `P`. Press `Q` to cancel.

## 2. Kinect 7-Scenes

An extracted sequence directory, `seq-XX.zip`, scene directory, or outer scene
ZIP can be passed. The nested-sequence cache is only needed for an outer ZIP.

```python
frame = load_kinect_point_cloud(
    "~/datasets/7scenes/redkitchen/seq-03.zip",
    frame_index=740,
    output_ply="outputs/input/kinect-frame-000740.ply",
)
```

7-Scenes RGB and depth are not registered. Consequently Kinect PLY output is
geometry-only. `frame.unregistered_rgb` contains the raw reference image, while
`frame.preview_rgb()` returns the depth-aligned image to use for clicking.

## 3. Live RealSense

```python
frame = capture_realsense_point_cloud(
    width=640,
    height=480,
    fps=30,
    output_ply="outputs/input/realsense.ply",
)
```

After warm-up, press `P` or Space to capture. Press `Q` to cancel. For headless
one-shot capture, pass `interactive=False`.

## Query-click and full FPS usage

Both algorithm modes consume the same frame:

```python
# Click mode: x/y are pixels in frame.preview_rgb().
query_index = frame.query_index_at_pixel(x, y, max_radius_px=2)
if query_index is not None:
    query_mm = frame.points_mm[query_index]

# Full-scene mode: pass all points to voxel/FPS processing.
points_mm = frame.points_mm
```

Useful fields:

- `points_mm`: `(N, 3)` point array for query or FPS algorithms.
- `depth_m`: original organized depth image.
- `intrinsics`: camera calibration.
- `pixel_to_point_index`: exact click-to-point lookup (`-1` means invalid).
- `rgb`: pixel-aligned color for STL and RealSense.
- `colors_rgb`: color per 3-D point when registration is valid.
- `metadata`: source and calibration details.

Calling `frame.save_ply(path)` saves the same point array at any time.

## Visualization and query utilities

All query utilities return a `QuerySelection`. Its indices always refer to the
original `frame.points_mm`, so a result can be passed directly to either the
detector or the visualization helpers.

### Inspect an input cloud

```python
from pointcloud_inputs import show_point_cloud

show_point_cloud(frame)
```

STL and RealSense clouds use aligned RGB colors. Kinect is intentionally shown
without raw RGB vertex colors because 7-Scenes RGB and depth are unregistered.

### Select full-scene FPS queries

```python
from pointcloud_inputs import select_fps_queries, show_point_cloud, show_original

queries = select_fps_queries(
    frame,
    count=500,
    min_spacing_mm=8.0,  # optional early-stop distance
)
print(queries.indices, queries.points_mm)

show_point_cloud(frame, queries)
show_original(frame, queries)
```

FPS is deterministic. Unless `start_index` is supplied, it starts at the point
nearest the candidate centroid. Use `candidate_indices` to restrict sampling to
a prefiltered or voxel-representative subset while preserving original indices.

### Select one query from the original image

```python
from pointcloud_inputs import select_query_from_original

query = select_query_from_original(frame, max_radius_px=2)
if query is not None:
    query_index = query.index
    query_mm = query.point_mm
```

Click the aligned original and the function returns immediately. For Kinect it
shows the depth preview rather than the unregistered RGB frame.

For programmatic selection without a GUI:

```python
from pointcloud_inputs import query_from_pixel

query = query_from_pixel(frame, u=320, v=240, max_radius_px=2)
```

To pick directly in the 3-D cloud instead:

```python
from pointcloud_inputs import select_queries_from_point_cloud

query = select_queries_from_point_cloud(frame, max_queries=1)
```

Use Shift+left-click in Open3D to pick, then press `Q` to finish.
