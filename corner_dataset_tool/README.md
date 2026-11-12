# Corner Dataset Tool v1.2 (RealSense + Open3D)

Two independent programs: **capture RealSense point clouds** and **annotate multiple 3D corners with a non-destructive ROI**. Coordinates are in **millimetres** in the depth-aligned-to-color optical coordinate system: X right, Y down, Z forward.

Saved scenes can also be run through an Ahmed (IROS 2018) MATLAB-to-Python port,
the original Sommer (RA-L 2020) repository and PCL Harris3D, with either the saved ROI or the
complete scene as input. Add `--visualize` for one synchronized window with
`GT / Sommer / Ahmed / Harris3D` panels by default; `view_baselines.py` reopens saved results
without recomputing. Ahmed defaults to Python and requires no MATLAB; select
`--ahmed-backend matlab` for the original MATLAB path.
See [baseline setup, commands and result contracts](baselines/README.md).

## 전체 데이터셋 베이스라인 비교

베이스라인 설치를 마친 뒤 **프로젝트 루트 (`keypoint-corner-detector/`)에서** 실행한다:

```bash
OPENBLAS_NUM_THREADS=1 corner_dataset_tool/.venv/bin/python corner_dataset_tool/run_baselines.py \
  corner_dataset_tool/dataset --mode roi \
  --preset scene2 --visualize
```

데이터셋의 모든 장면을 번호순으로 처리한다. 각 장면의 Sommer·Ahmed·Harris3D 결과를 계산·저장한 뒤
**GT / Sommer / Ahmed (Python port) / PCL Harris3D** 비교 창을 연다. **창의 X 버튼으로 닫으면 다음 장면을
계산해 표시한다.** 드래그로 회전하고 휠로 확대·축소하며, 패널 간 시점은 연동된다.
전체 실행을 중단하려면 실행한 터미널에서 `Ctrl+C`를 누른다.

`--mode roi`는 장면별 저장된 ROI를 사용한다. 전체 점군으로 비교하려면 `--mode scene`으로
바꾼다. `scene2`는 scene_0002 GT로 튜닝한 값을 모든 장면에 동일하게 적용하며,
다른 장면에서 Sommer 1개·Ahmed 4개가 나오는 것을 보장하지 않는다.
결과는 `corner_dataset_tool/baseline_results/<UTC timestamp>/`에 저장된다.
`--visualize`는 `--method`에 일부 방법만 적어도 나머지를 자동으로 계산해 세 방법을 모두 표시한다.
저장 결과에 없는 방법은 재계산 없이 `NOT_SAVED`로 표시한다.
패널 순서·저장 결과 다시 보기는 [시각화 사용법](baselines/README.md#gt--방법별-동시-시각화)을 참고한다.

## 현재 corner_detector / Q10 Lite 실행

베이스라인과 별도로 현재 검출기를 저장된 데이터에 실행할 수 있다. **프로젝트 루트에서**:

```bash
# 한 장면의 저장 ROI + 기존 검출기/Lite 시각화
python3 -m corner_detector --source dataset --mode detector --input corner_dataset_tool/dataset/scene_0001

# 전체 데이터셋: 번호순으로 실행, 각 장면의 창을 닫으면 다음 장면
python3 -m corner_detector --source dataset --mode detector --input corner_dataset_tool/dataset

# 전체 점군, GUI 없이 계산/로그만
python3 -m corner_detector --source dataset --mode detector --input corner_dataset_tool/dataset --dataset-region scene --no-3d-view
```

기본 `--dataset-region roi`는 저장된 `annotations.json`의 ROI를 사용한다. `scene`은 전체 XYZ를 사용한다.
원본 점·RGB·ROI·GT 파일은 수정하지 않고, GT는 검출 계산에 사용하지 않는다.
검출 파라미터는 `corner_detector/settings.py`와 기존 detector/Q10 CLI를 공유한다.
Q10 수렴값이 하한 미달이면 RANSAC 전에 바로 REJECT한다. 통과한 점만 단위구 평면·양방향 검사를 수행한다.
기존 Lite 마커를 클릭해 Q10/단위구/평면별 장미도를 확인할 수 있다.
현재 데이터셋 입력은 detector 모드 전용이며, 결과 저장·베이스라인 정량 비교는 추가하지 않는다.
전체 실행은 `Ctrl+C`로 중단한다. 카메라 없이 저장된 `points.npz`만으로 실행한다.

## Standalone Corner Lab: 씬 번호로 열기 / RealSense 실행

프로젝트 루트에서:

```bash
# 저장 데이터셋 scene_0002 열기 (숫자만 바꾸면 해당 씬 선택)
python3 corner_lab_open3d_cone.py --scene 2

# 데이터셋이 다른 위치에 있을 때
python3 corner_lab_open3d_cone.py --scene 2 --dataset /path/to/dataset

# RealSense 카메라로 시작한 뒤 CAPTURE로 점군 저장/분석
python3 corner_lab_open3d_cone.py --camera
```

기본 데이터셋 경로는 스크립트 옆의 `corner_dataset_tool/dataset`이다.
`--dataset`에 `corner_dataset_tool` 폴더를 지정해도 그 안의 `dataset/`을 찾는다.
씬 번호는 폴더의 번호이며 폴더 목록의 순번이 아니다. 없는 씬이나 0 이하 번호는 창을 열기 전에 오류로 알린다.
`--open corner_dataset_tool/dataset/scene_0002/points.npz` 또는 장면 폴더 경로로도 열 수 있다.

lab은 ROI로 자르지 않은 전체 원본 XYZ/RGB를 읽어 raw point index를 유지한다.
장면에 lab용 `corners.json`이 없으면 `annotations.json`의 GT를 `gt_001` 같은 이름으로 가져온다.
화면의 **Saved corners** 목록에서 GT를 선택해 Q10/단위구/평면별 양방향성을 확인한다.
원본 `points.npz`와 `annotations.json`은 수정하지 않는다. lab에서 라벨 추가·보정·저장을 하면
기존 lab 규칙대로 별도 `corners.json`에 저장하며, 다음에 열 때 이 파일을 우선 사용한다.
기존 `cloud.npz`, `--demo`, `--camera` 사용법과 계산은 유지한다.

## Install (Ubuntu/Linux, Python 3.10–3.12 recommended)

```bash
cd corner_dataset_tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pyrealsense2` needs a compatible RealSense device, USB connection and platform support. On Ubuntu, camera permissions/udev rules may also need setup. If your Python version doesn't have a supported wheel, switch to Python 3.10–3.12. Open3D needs a local **desktop GUI session** (X11/Wayland + working OpenGL); SSH/headless sessions will not display the annotator.

## 1. Record a RealSense scene

```bash
python realsense_capture.py --output dataset
# SPACE: save scene_0001, scene_0002, ...
# Q / Escape: quit
```

Defaults are synchronized 640×480@30 RGB and depth, with depth aligned to the color frame. If a camera rejects this profile, try another supported width/height/fps. Saving one scene takes a single aligned RGB-D frame; this is **not a temporal average**.

## 2. Annotate

```bash
python corner_annotator.py dataset/scene_0001
```

- **Left / Right arrow key**: jump to the previous/next *existing* scene folder, in numeric order (`scene199` → `scene200`, or `scene_0199` → `scene_0200`). The two **< Previous / Next >** GUI buttons do the same thing. Scene folders are searched in the current scene's parent directory and must contain `points.npz`. For numbered scenes only the same prefix is considered; missing numbers are skipped. At either end of the list, navigation stops.
- **Auto-save when switching scenes:** edited ROI and all Corner GTs are saved to the *current scene's* `annotations.json` before changing scene. The next scene loads its previously saved ROI/GTs if they exist. Switching resets Undo, list selection and the camera view; `raw.ply` and background are unchanged. On load/save failure the current scene remains open. Editing ROI numeric boxes without clicking **Apply ROI** does not constitute a saved ROI change. `roi.ply` is only updated on explicit **Export ROI.ply**.
- **Arrow-key caveat:** the global Left/Right shortcuts also apply while numeric inputs have keyboard focus, so use the panel's number fields via mouse selection before typing, or use the GUI scene buttons to navigate explicitly.
- **Ctrl + left click** on a visible point: add a Corner GT. Repeat to add multiple GTs. Selection is done in screen pixels using the Open3D camera's projection and existing original points, not with a rendered depth image.
- **Pick mode: plain left click adds GT** checkbox: turn it ON if Ctrl+click does not register on your desktop environment. Then left-click on a point; turn it OFF to restore left-drag rotation. This also works without keyboard modifiers. Default is OFF.
- **Pick radius (px)**: default 12. If a very sparse point cloud does not pick reliably, try 16–20 px. Empty space outside the radius is rejected. The GT always references an **existing raw point** (never a virtual coordinate).
- **After click**, a red GT marker/number and a status message `Added GT #... raw index ...` should appear. If not, inspect the status text at the bottom of the side panel or run from a terminal for error output.
- **Shift + left click**: move ROI center to the picked point, preserving its X/Y/Z size.
- **ROI CENTER / ROI SIZE (mm)**: edit six numerical fields, then press **Apply ROI**. `150mm box` sets all three sizes to 150 mm and applies. `Full cloud` resets the ROI to encompass every raw point. `Frame ROI` changes camera view, not data.
- **VIEW MODE**: Full cloud; ROI only; ROI + grey background. Background greying is a visualization effect only.
- Select a labeled corner from the list and choose **Delete selected**. **Undo** reverts the last ROI or corner edit.
- **Save labels** writes the annotation JSON. **Export ROI.ply** also saves the cropped point cloud and an array mapping each cropped index back to the raw point index. **Reload saved labels** loads the last saved annotation, discarding unsaved changes.

Zoom in and rotate the point cloud for precise selection. The 3D box is axis-aligned (AABB); it is *not* a freehand mesh cropping tool in v1.

## Scene directory

```text
dataset/
└── scene_0001/
    ├── raw.ply            # complete original cloud (millimetres)
    ├── points.npz         # exact stable point order, RGB, original depth pixel IDs
    ├── rgb.png            # aligned color frame
    ├── depth.npy          # original uint16 depth aligned to color frame
    ├── metadata.json      # depth scale, camera intrinsics, timestamp, frame no.
    ├── annotations.json   # editable ROI + multiple Corner GT (after save)
    ├── roi.ply            # ROI points (after export)
    └── roi_indices.npy    # roi.ply[i] -> raw.ply[roi_indices[i]]
```

**Important:** `raw.ply` and `points.npz` are immutable originals. A label stores `raw_point_index`, `xyz_mm`, `pixel_index`; changing the ROI does **not** change the raw index or delete GTs. Corner GTs outside the current ROI remain in the JSON and are marked `[OUTSIDE ROI]` in the list. You can restore the entire background using `Full cloud` or the corresponding view mode.

### Example annotation

```json
{
  "schema_version": 1,
  "scene_id": "scene_0001",
  "unit": "mm",
  "coordinate_frame": "color_optical",
  "roi": {"type": "aabb", "min_mm": [-100, -100, 400], "max_mm": [100, 100, 800]},
  "corners": [
    {"id": 1, "raw_point_index": 15230, "xyz_mm": [12.5, 41.2, 635.1], "pixel_index": 24411, "inside_roi": true}
  ]
}
```

## Test without RealSense

```bash
python make_demo_scene.py --output dataset/demo_corner
python corner_annotator.py dataset/demo_corner
python -m pytest tests -q
```

The demo contains a noisy 3-plane corner and an unrelated background. Use `150mm box` with the ROI centered around (45,45,595), or Shift-click on the corner to move its center.

## Engineering notes / limitations

- **No in-place raw edit.** ROI selection never erases background; exporting writes new derivative files.
- **Stable index:** `points.npz` is the ground truth ordering, not PLY readback order. For registration evaluation, load XYZ from `points.npz` and use `roi_indices.npy` as the index mapping.
- **Precision:** single-frame RealSense samples can be noisy or missing at geometric vertices. Chosen label is the nearest *observed* point, not a guaranteed exact physical intersection.
- **Display pick (v1.1):** project selectable raw points to the screen with the current camera matrices; choose a close actual point inside the adjustable pixel radius. This avoids rendering-depth readback errors and ROI line picking. In ROI + grey background mode, only points INSIDE the ROI can be labeled (switch to Full cloud mode to label outside). Exact mesh-style occlusion is not evaluated; use rotation and zoom when several surfaces overlap.
- **ROI model:** v1 uses an axis-aligned box edited by numeric fields and Shift-click. No rotation handles, polygon crop, or multi-ROI management.
- **Manual saving:** remember to save after editing. `Export ROI.ply` saves annotations as well.

## v1.1 click-selection fix

The original v1 code rendered a depth image on click, unprojected one pixel into a world-space XYZ, and rejected picks farther than ~5 mm from a visible point. This was sensitive to Open3D depth readback, point size, graphics drivers, and the ROI wireframe. The v1.1 selector uses direct camera projection of the actual raw points and a pixel-based picking tolerance. It also adds the modifier-free Pick mode. All saved data formats and ROI preservation semantics are unchanged.

## v1.2 scene navigation

- Scene order is **natural numeric** rather than alphabetical; e.g. `scene9` before `scene10`.
- Both window-level arrow keys and visible **< Previous / Next >** buttons perform the same navigation.
- Scene annotations are auto-saved before switching, but raw point clouds are never overwritten.
- A missing/corrupt destination scene or failed save never silently discards unsaved annotations.
