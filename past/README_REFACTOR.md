# Modular corner detector (refactored from stl_tangent_boundary_demo.py)

This project splits the supplied 2,509-line script without changing the math in
`plane_projection.py`, `corner_2d.py`, `validator.py`, or `pipeline.py`.

## Layout

- `pointcloud_inputs/`: unchanged STL / Kinect 7-Scenes / RealSense capture, FPS, click map and visualization.
- `corner_detector/__main__.py`: **only** `python -m corner_detector` entry point.
- `corner_detector/runner.py`: capture, set up, query/FPS execution.
- `corner_detector/settings.py`: CLI arguments and `BoundaryConfig`.
- `corner_detector/preprocessing.py`: optional synthetic noise + 3D voxel processing.
- `corner_detector/normal_estimation.py`: load SAH backend.
- `corner_detector/pipeline.py`: one-query projection -> 2D corner -> 3D verification, iterative localization and 3D refinement.
- `corner_detector/plane_projection.py`: 3D tangent plane geometry.
- `corner_detector/corner_2d.py`: 2D boundary estimation, tensor and Hough.
- `corner_detector/validator.py`: 3D EXT and corner verification.
- `corner_detector/fps.py`: multi-query evaluation and 3D corner NMS.
- `corner_detector/query_debug.py`: single-query evaluation and debug data.
- `corner_detector/query_ui.py`: Tk frozen-frame GUI and navigation.
- `corner_detector/plots.py`: 2D visual diagnostic plots.
- `corner_detector/view3d.py`: Open3D debug visualizations.
- `corner_detector/diagnostics.py`: render/cache iteration frames.
- `surface_aware_harris3d/`: original backend sources (native C++ build outputs excluded).

There is **no** `corner_detector/main.py`. Legacy `stl_tangent_boundary_demo.py` is a
small wrapper kept so existing command lines still work.

## Examples

```bash
# STL camera / query click
python -m corner_detector --source stl --input data/laser_pose_box.stl --mode query
# STL camera / FPS
python -m corner_detector --input data/laser_pose_box.stl --mode fps --fps-count 128
# Kinect frame
python -m corner_detector --source kinect --input ~/datasets/7scenes/redkitchen/seq-03.zip --frame-index 740 --mode query
# Live RealSense
python -m corner_detector --source realsense --mode fps --fps-count 128
# Original entrypoint still accepted
python stl_tangent_boundary_demo.py --input data/laser_pose_box.stl --mode query
```

`--help` lists original scales, threshold and debug options.
The same `PointCloudFrame` interface is used for all sources.

The legacy debug windows require Tk, matplotlib, Pillow and Open3D.
The numerical pipeline needs numpy, scipy and the supported SAH backend.
The supplied SAH package includes native `pclpybridge` C++ sources, not a
portable prebuilt binary; build/install it on the target environment.

### Point-index invariant

`pointcloud_inputs` FPS indices are indices in the array **passed to FPS**.
In Query mode, a clicked pixel is mapped into the original cloud and then
snapped onto the 3D voxel cloud. All `pipeline` indices refer to that voxel
cloud. Do not use an index from the original frame to index the voxel array.

### What is verified

Syntax compilation, import smoke checks and CLI help can be tested without a
sensor. Full GUI, RealSense and SAH+pclpybridge integration require the target
machine and are not claimed as validated.
