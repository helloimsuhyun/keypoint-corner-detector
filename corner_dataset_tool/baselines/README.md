# Dataset baseline 실행

`run_baselines.py`는 저장된 `points.npz`에 다음 baseline을 실행한다.
Sommer와 PCL은 원본 C++ 구현을 사용하고, Ahmed는 저자 MATLAB 코드의 Python 포트를 기본 실행한다.
기존 `corner_detector`의 알고리즘과 설정은 바꾸지 않는다.

| `--method` | 실행하는 구현 | 고정 버전 |
|---|---|---|
| `ahmed` | [Merium88/Point-cloud-Edge-and-Corner-Detection](https://github.com/Merium88/Point-cloud-Edge-and-Corner-Detection)의 `edgedetection.m` / `corner_detection.m` Python 포트; 원본 MATLAB 실행도 선택 가능 | 참조 커밋 `e8aa7a4254cf67c3bdd6a82a7efb67d51c47994c` |
| `orthogonal` | [c-sommer/orthogonal-planes](https://github.com/c-sommer/orthogonal-planes)의 C++ 함수와 `ply_corners` 단계 순서 | `1a05583ce7913308c5a523f90bd34fa6608c8864` |
| `harris3d` | [PointCloudLibrary/pcl](https://github.com/PointCloudLibrary/pcl)의 `pcl::HarrisKeypoint3D` | 빌드한 시스템 PCL 버전을 결과에 기록 |

저장소는 `vendor/`, 빌드와 생성한 어댑터는 `build/`에 둔다. 두 디렉터리는 Git에서 제외한다.
다운로드한 저자 소스는 수정하지 않는다. 재설치 시 로컬 수정이 있으면 덮어쓰지 않고 중단한다.

## 설치

아래 명령은 프로젝트 루트에서 실행한다. Python에는 NumPy, SciPy가 필요하다.
시각화 없이 실행할 때는 카메라·Open3D·GUI가 필요하지 않다.
비교 화면에는 Open3D, Matplotlib, Pillow와 데스크톱 디스플레이가 필요하다.
기존 도구의 가상환경에서는 `corner_dataset_tool/.venv/bin/python -m pip install -r corner_dataset_tool/requirements.txt`로 준비한다.

```bash
python3 -m pip install numpy scipy

# Ubuntu 22.04의 Ceres 2.0 패키지와 호환.
# 필요한 패키지가 없을 때 설치한다.
sudo apt-get install git cmake g++ libpcl-dev libopencv-dev libceres-dev

python3 corner_dataset_tool/baselines/setup.py --method harris3d
python3 corner_dataset_tool/baselines/setup.py --method orthogonal
# 원본 MATLAB backend를 사용할 때만 필요. Python 포트에는 다운로드/빌드 불필요.
python3 corner_dataset_tool/baselines/setup.py --method ahmed
```

`--method all`로 한 번에 준비할 수도 있다. Sommer에는 Ceres **1.14 이상, 2.2 미만**과
저자 저장소에 고정된 Sophus·CLI11 서브모듈을 사용한다. Ceres 2.2 이상은 저자 코드의
`LocalParameterization` API를 제거했으므로 지원하지 않는다. Pangolin 뷰어는 필요하지 않다.
Ceres를 별도 위치에 설치했다면 `CMAKE_PREFIX_PATH`로 지정한다.

Ahmed 기본 실행은 **NumPy·SciPy만 사용하며 MATLAB이 필요 없다**.
`--ahmed-backend matlab`으로 원본 함수를 선택할 때만 **MATLAB R2019a 이상**, Statistics and Machine Learning Toolbox
(`knnsearch`, `pca`, `kmeans`, `pdist2`), Computer Vision Toolbox
(`pointCloud`, `pcdownsample`)가 필요하다. `--matlab /path/to/matlab`로 실행 파일을 지정할 수 있다.
기본 `--ahmed-backend python`과 원본 `matlab`은 명시적으로 구분하며 자동 fallback은 없다.

## 데이터셋 기본 설정

2026-11-09부터 기본 `--preset dataset`은 작은 장면 비교에서 선택한 값을 사용한다.
기존 데모값은 `--preset author`로 복원한다. 자세한 비교와 낮은 검출 성능의 한계는
[튜닝 기록](../../docs/BASELINE_TUNING.md)에 있다.

| 설정 | author | dataset (기본) |
|---|---:|---:|
| Sommer normal / dmin / dmax, mm | 20 / 100 / 1000 | 10 / 5 / 100 |
| Sommer 최소 투표 / refinement stride | 5 / 10 | 3 / 5 |
| Ahmed voxel / analysis voxel, mm | 10 / 5 | 5 / 2.5 |
| Ahmed edge K / 문턱 배수 / thinning K | 400 / 4.5 / 150 | 100 / 2 / 30 |
| Ahmed corner K / L | 20 / 10 | 20 / 10 |

개별 인자는 preset보다 우선한다. Ahmed는 `--ahmed-voxel-mm`,
`--ahmed-edge-neighbors`, `--ahmed-edge-threshold`, `--ahmed-thinning-neighbors`,
`--ahmed-corner-neighbors`, `--ahmed-pca-neighbors`, `--ahmed-span-mm`,
`--ahmed-angle-min-deg`, `--ahmed-angle-max-deg`로 조절할 수 있다.
PCA K는 edge 방향과 cloud normal 양쪽에 적용한다. span은 patch의 각 좌표축 폭을
비교하는 mm 문턱이고, 각도는 원본 수식의 엄격한 하한/상한이다.
두 기존 preset의 PCA/span/각도 기본값은 10 / 5 mm / 50–150°로 유지한다.
voxel 인자는 최초·edge grid에 적용하고
analysis grid에는 그 절반을 적용한다. MATLAB backend는 원본 고정값을 유지하므로
이 수치 override를 받지 않는다. 해석된 실제 값은 `run.json`에 모두 기록한다.
Sommer는 기존 거리·법선·투표·stride 인자와 `--sommer-voxel-mm`를 받는다.
입력 샘플링의 기본값 0은 비활성화다. 양수이면 voxel 평균에 가장 가까운 실제 관측점
하나를 골라 원본 ID를 보존한다. 평균 좌표를 새 scene 점으로 만들지 않는다.

## 실행

scene_0002 ROI에 맞춘 `--preset scene2`는 Sommer 1개, Ahmed Python 4개를 검출한다.
GT #1–#4에 대한 Ahmed 오차는 5.5 / 3.9 / 7.3 / 3.1 mm, Sommer는 GT #1에서 8.8 mm다.
Ahmed 값은 voxel 5 mm, edge K=100, 배수 2, thinning K=5, corner K=27,
PCA K=5, span 8 mm, 각도 53.3–116°, L=1, seed=1이다. Sommer는 dataset 값과 같다.
**같은 장면 GT로 튜닝한 결과이며 다른 장면이나 전체 씬 성능을 보장하지 않는다.**
기본 preset은 계속 `dataset`이다. 자세한 기록은 [튜닝 보고서](../../docs/BASELINE_TUNING.md)를 참고한다.

```bash
corner_dataset_tool/.venv/bin/python corner_dataset_tool/run_baselines.py \
  corner_dataset_tool/dataset/scene_0002 --mode roi \
  --preset scene2 --visualize
```

```bash
# 한 씬: 저장한 AABB ROI 안의 점만 검출기에 입력
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset/scene_0001 \
  --method harris3d orthogonal --mode roi

# 같은 씬: 전체 원본 점군 입력
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset/scene_0001 \
  --method harris3d orthogonal --mode scene

# MATLAB 없이 저자 로직의 Python 포트 실행
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset/scene_0001 \
  --method ahmed --mode roi

# MATLAB 환경이 있을 때 원본 함수 실행
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset/scene_0001 \
  --method ahmed --ahmed-backend matlab --mode roi

# 전체 데이터셋을 자연수 씬 순서로 배치 실행
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset \
  --method harris3d orthogonal --mode roi

# 특정 씬 선택; --scene을 반복하거나 패턴 사용
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset \
  --scene 'scene_000[1-3]' --method harris3d --mode scene

# 런타임 설치 없이 입력과 실행 명령만 준비
python3 corner_dataset_tool/run_baselines.py corner_dataset_tool/dataset/scene_0001 \
  --method ahmed orthogonal harris3d --mode roi --prepare-only
```

`python3 -m corner_dataset_tool.run_baselines ...`도 동일하다. ROI 모드는
`annotations.json`이 없거나 ROI가 비어 있으면 실패한다. 전체 씬으로 몰래 대체하지 않는다.
`roi.ply`와 `roi_indices.npy`는 사용하지 않으므로 ROI를 편집하고 PLY를 다시 내보내지 않았어도
저장된 JSON 범위가 적용된다. 실행 중 GT는 검출기 입력이나 필터로 사용하지 않는다.
dataset preset의 사전 설정 선택에는 튜닝 기록에 명시한 장면의 GT를 사용했다.

## GT / 방법별 동시 시각화

### 전체 데이터셋을 한 장면씩 넘겨 보기

베이스라인 설치 후 프로젝트 루트 (`keypoint-corner-detector/`)에서 실행한다:

```bash
OPENBLAS_NUM_THREADS=1 corner_dataset_tool/.venv/bin/python corner_dataset_tool/run_baselines.py \
  corner_dataset_tool/dataset --mode roi \
  --preset scene2 --visualize
```

장면을 번호순으로 계산·저장하고 **GT / Sommer / Ahmed (Python port) / PCL Harris3D** 비교 창을 연다.
**현재 창의 X 버튼을 눌러 닫으면 다음 장면을 계산한 뒤 새 비교 창이 열린다.**
드래그로 회전, 휠로 확대·축소하며 모든 패널의 시점이 연동된다.
전체 실행 중단은 실행한 터미널의 `Ctrl+C`를 사용한다.
각 장면은 볼 차례가 되었을 때 계산하므로, 첫 창이 열렸다고 전체 계산이 끝난 것은 아니다.

`--mode roi`는 저장한 ROI를 사용한다. 전체 점군은 `--mode scene`으로 바꿔 실행한다.
`--preset scene2`는 scene_0002 튜닝값을 모든 장면에 동일하게 적용하며,
다른 장면에서 검출 개수나 정확도를 보장하지 않는다. `--method` 생략 시 세 방법을 모두 실행한다.
`--visualize`에서는 일부 방법만 지정해도 나머지를 자동으로 추가해 세 방법을 계산·표시한다.
시각화 없이 계산할 때는 `--method ahmed`처럼 일부 방법만 실행할 수 있다.
장면별 결과는 기본 `corner_dataset_tool/baseline_results/<UTC timestamp>/`에 남고,
저장 결과를 `view_baselines.py`로 다시 열면 재계산 없이 같은 방식으로 넘겨 볼 수 있다.

### 한 장면 실행 및 저장 결과 다시 보기

```bash
# MATLAB 없이 GT / Harris3D / Sommer / Ahmed (Python port) 동시 표시
corner_dataset_tool/.venv/bin/python corner_dataset_tool/run_baselines.py \
  corner_dataset_tool/dataset/scene_0001 \
  --method harris3d orthogonal ahmed --mode roi --visualize

# 저장한 결과를 재계산 없이 다시 보기
corner_dataset_tool/.venv/bin/python corner_dataset_tool/view_baselines.py \
  corner_dataset_tool/baseline_results/smoke_20261109_roi

# 패널 순서 지정: GT / Sommer / Harris3D / Ahmed
corner_dataset_tool/.venv/bin/python corner_dataset_tool/view_baselines.py \
  corner_dataset_tool/baseline_results/smoke_20261109_roi \
  --method orthogonal harris3d --scene scene_0001
```

`--method` 뒤에 여러 방법을 공백으로 나열하면 해당 패널 순서를 우선한다.
GT는 맨 왼쪽이고, 지정하지 않은 방법은 뒤에 추가해 항상 세 방법을 모두 표시한다.
기본 순서는 Sommer / Ahmed / Harris3D다. 모든 패널은 **동일한 입력 점군과 동기화된 카메라**를 사용한다.
어느 패널에서 회전·이동·확대·축소하든 나머지도 따라간다. GT는 초록색, 예측은 빨간색이다.
패널 제목에 방법과 검출 개수를 표시하며, 실패/미실행 결과는 `FAILED`/`PREPARED`로 구분한다.
과거 저장 결과에 빠진 방법은 `NOT_SAVED`로 표시하고, 마우스를 올리면 재실행 안내를 볼 수 있다.
저장 뷰어는 빠진 방법을 자동 계산하지 않는다. 실제 예측은 실행기를 `--visualize`로 다시 실행해 저장한다.
실패 이유는 제목에 마우스를 올리면 보인다. 성공한 빈 검출은 `0 corners`다.

ROI 실행 화면에는 저장된 ROI 입력, scene 실행 화면에는 전체 입력이 나온다.
가상 교차점/정제 위치는 저장 좌표 그대로 표시하며 원본 점으로 스냅하지 않는다.
다시 보기에서도 실행 당시 저장한 GT·입력 인덱스를 사용하고 현재 annotation을 다시 읽지 않는다.
`--visualize` 배치 실행은 씬별로 모든 방법을 계산·저장한 뒤 비교 창을 열며,
창을 닫으면 다음 씬으로 진행한다. 저장 결과 뷰어도 같은 방식으로 씬을 순서대로 연다.
저장 뷰어에서도 세 방법의 패널을 항상 표시한다. MATLAB 없이도
이미 저장된 Ahmed 결과를 볼 수 있다. `--prepare-only --visualize`는 함께 사용할 수 없다.

기본 출력은 `corner_dataset_tool/baseline_results/<UTC timestamp>/`이다.
`--output /path/to/new-run`으로 변경할 수 있다. 기존 출력 폴더와 입력 데이터셋 내부에는
쓰지 않는다. 씬 원본·ROI·GT 파일은 읽기만 한다.

## 결과

```text
<run>/
  manifest.json
  scene_0001/roi/
    input_xyz_m.txt
    input_selection.npz
    harris3d/
      predictions.csv
      predictions.npz
      backend_corners_m.csv
      run.json
      stage_1.log
    orthogonal/...
    ahmed/...
```

- `predictions.npz`: `xyz_mm (K,3)`, `response (K,)`,
  `nearest_raw_point_index (K,)`, `nearest_raw_distance_mm (K,)`.
  CSV도 동일한 의미다. **출력 좌표를 원본 점으로 스냅하지 않는다.**
  가장 가까운 raw ID는 검색 정보이며, 가상 교차점이나 정제된 Harris 위치의 실제 support ID가 아니다.
  점수가 없는 방법은 `response=NaN`이다.
- `input_selection.npz`: 입력 순서 → 원본 인덱스 `raw_point_indices`,
  선택한 영역의 `gt_ids`, `gt_raw_point_indices`, `gt_xyz_mm`.
  ROI 밖 GT는 원본 JSON에 그대로 남고 ROI 실행의 비교용 배열에서만 제외된다.
- `run.json`, `manifest.json`: 소스 커밋/PCL 버전, 명령, 파라미터, 입력 SHA256,
  입력 점 수, 상태와 실행 시간. `elapsed_seconds`는 전처리·파일 I/O·프로세스 시작을 포함하므로
  논문의 순수 알고리즘 시간과 동일하지 않다.
- Sommer는 `normals_m.csv`, float32 `oriented_m.ply`, 유효 normal 점의 원본 ID인
  `normal_raw_indices.npy`도 보존한다.
  입력 샘플링이 켜져 있으면 `sampled_xyz_m.txt`, `sampled_raw_indices.npy`도 보존한다.
  normal CSV의 `input_index`는 이 샘플 입력 안의 ID이며, `normal_raw_indices.npy`가
  실제 원본 ID로 변환한 값이다. 공통 화면의 배경은 샘플링 전의 동일 ROI/scene이다.
- Ahmed는 원래 L=1…10 예측 전체를 `ahmed_all_levels.mat`에 보존한다.
  `--ahmed-level`(기본 10, scene2는 1)은 공통 CSV/NPZ로 내보낼 결과만 선택한다.
  Python 포트는 `ahmed_stages.npz`에 초기 voxel, edge 거리/후보, 최종 voxel cloud,
  thinning edge, PCA 방향/법선과 L별 boolean mask도 보존한다.
  `edge_initial_indices`는 `initial_cloud_m`, `edge_indices`는 `cloud_m` 안의 인덱스다.
  이 인덱스는 원본 scene 인덱스가 아니다.
  Python 포트 결과에는 `source.implementation=python_port`, 포트 소스 SHA256,
  `matlab_parity_verified=false`가 기록되고 화면에도 `(Python port)`가 붙는다.
- 검출 0개는 헤더가 있는 빈 결과와 `status=ok`로 저장한다. 크래시·잘못된 출력·시간 초과는
  `status=failed`로 기록하며 0개 성공으로 취급하지 않는다. 다음 방법/씬은 계속 실행한다.
  실패가 하나라도 있으면 종료 코드는 1이다. `--timeout`은 각 외부 단계의 제한 시간(초)이다.
- `--prepare-only`는 `status=prepared`로 저장하며 검출 결과를 만들지 않는다.
  실제 실행은 런타임 준비 후 새 출력 폴더에서 실행기를 다시 호출한다.
  특히 Sommer 명령 사이에는 실행기가 normal CSV → PLY 변환을 수행한다.

## 저자 코드와 입출력 어댑터의 경계

**단위:** 데이터셋/공통 출력은 mm, 외부 실행은 m다. 입력을 1000으로 나누고
출력을 1000배 한다. ROI는 mm 원본에서 먼저 적용한다. 원점 이동·추가 NMS·코너 병합·GT 기반 필터는 없다.

**Sommer:** 공식 `ply_corners/src/main.cpp`를 복사한 생성 파일에서 입력 경로,
CSV 출력, seed, GUI 호출만 바꾼다. PPF 투표 → 그래프 클러스터링/축소 → outlier filtering →
coarse refinement → 삼각형 탐색 → `Corner::eval_candidate` → `Corner::refine`를 유지한다.
PPF·그래프·Corner 구현은 원본 C++ 파일을 직접 컴파일한다. 공식 예제는 normals가 있는
PLY를 요구하므로 데이터셋에 없는 normals만 PCL 반경 PCA로 추정한다.
원점 카메라 방향으로 normal을 정렬하고 유효 normal 점만 전달한다.
공식 기본값 `--normal-radius-mm 20 --dmin-mm 100 --dmax-mm 1000 --min-votes 5 --sampling 10`은
`--preset author`에 보존되며, 기본 dataset 값은 위 표를 따른다.
`sampling`은 입력 다운샘플링 옵션이 아닌 원래 refinement 단계의 stride다.
소규모 ROI 설정 비교는 수행했지만 확인 장면에서 Sommer의 개선은 확인되지 않았다. 직교 세 평면 코너를 대상으로 하므로
모든 각도의 코너를 검출하는 방법과 범위가 다르다. 저자 README도 공개 버전이 일부
subsampling/clustering을 단순화한 데모라고 설명한다.

**Ahmed Python 포트:** `baselines/ahmed.py`는 논문 설명 대신 위 참조 커밋의 MATLAB
계산 순서를 따른다. author 설정은 최초 10 mm gridAverage → 400-NN centroid 이동량과 비자기 최근접
거리의 4.5배 비교 → 150-NN 평균 thinning → 같은 위치가 된 모든 점 삭제 →
edge 10 mm / cloud 5 mm gridAverage → 가장 가까운 cloud 점에 edge 배치 →
10-NN PCA의 첫 성분(curvature)·세 번째 성분(normal) → 20-NN edge patch →
XYZ 범위가 5 mm를 넘는 축 수에 따라 k=2/3 → k-means → L=1…10 판정을 수행한다.
각도는 원본대로 `atan2(norm(cross(u,v)),dot(u,v))*180/3.142`, 엄격한 50<각도<150 조건이다.
dataset 설정은 위 공간 크기·K·배수만 위 표의 값으로 바꾼다.
3-cluster 크기 조건은 원본의 **OR**를 유지한다. L마다 k-means를 다시 실행하므로
L별 검출 개수가 단조 증가한다고 가정하지 않는다. 추가 NMS/코너 병합은 없다.

gridAverage는 bounding-box minimum에서 시작하는 half-open 격자를 사용하고 X가 가장
빠르게 바뀌는 순서로 voxel을 나열한다. KNN은 자기 자신을 포함하고 동률은 작은 입력 ID로
해결한다. 중복 thinning 점은 NxN 거리 행렬 없이 동일 행의 개수로 모든 중복 멤버를 제거한다.
이는 한 점만 남기는 일반적인 unique와 다르다.
[MATLAB gridAverage 문서](https://www.mathworks.com/help/pointcloud/ref/pcdownsample.html),
[KNN 동률 규칙](https://www.mathworks.com/help/stats/knnsearch.html)을 참조했다.

PCA는 원본의 두 차례 중심화를 유지하고 SVD를 사용한다. loading 중 절댓값이 가장 큰
성분이 양수가 되게 한다. k-means는 squared Euclidean, k-means++ 초기화, 1 replicate,
최대 100회, 빈 cluster의 farthest-point singleton 처리를 구현했다.
[PCA 문서](https://www.mathworks.com/help/stats/pca.html),
[k-means 문서](https://www.mathworks.com/help/stats/kmeans.html)의 기본 동작을 따른다.
**MATLAB과 bitwise 동등성을 검증한 것은 아니다.** RNG 소비 순서, grid 경계/순서,
SVD의 퇴화 축과 내부 k-means 구현 차이로 예측이 달라질 수 있으므로 Python 포트로 보고해야 한다.

**Ahmed MATLAB backend:** 원본 `edgedetection.m`을 직접 호출한다. 생성한
`corner_detection.m` 사본은 반환값/결과 cell만 추가하고 마지막 plot만 제거한다.
두 backend 모두 예제의 작업물 전용 passthrough 범위와 X=0 점 삭제를 저장된 ROI/전체
씬 선택으로 대체한다. Python에서 edge가 0개이거나 thinning 후 모두 제거되면 빈 예측을
반환한다. 그 외 고정 KNN 지원이 부족한 입력은 오류로 기록하고 k를 조용히 줄이지 않는다.
원본 MATLAB의 `pdist2(thin_edge,thin_edge)`는 edge 점 수에 대해 제곱 메모리를 사용한다.

**Harris3D:** PCL의 HARRIS 응답, normal estimation, NMS, refinement를 직접 사용한다.
기본 `--harris-radius-mm 20 --harris-threshold 1e-6`, refinement ON이다.
`--no-harris-refine`으로 끌 수 있다. 단일 스레드로 실행한다.
이는 이 데이터셋에서 튜닝한 최적 성능 설정이 아니다.

`--seed 1`을 두 방법에 전달한다. Python 포트는 NumPy RandomState/MT19937,
MATLAB은 Twister, C++은 `srand`다. backend 사이 같은 seed가 같은 결과를 뜻하지 않는다.
GT를 기반으로 seed나 L을 고르지 않는다.

## 검증

```bash
python3 -m unittest discover -s tests -p test_dataset_baselines.py -v
python3 -m unittest discover -s tests -p test_ahmed_python.py -v
```

PCL 바이너리가 있으면 실제 평면 입력에서 Harris 0개와 카메라 방향 normal의 예상 수치를 검사한다.
빌드가 없으면 네이티브 통합 테스트만 skip한다. ROI 경계/단위/인덱스/가상 코너 좌표 보존,
빈 결과와 실패 구분, 비파괴 출력도 검사한다.
`test_dataset_baseline_visualization.py`는 패널 순서·저장 GT 사용·공유 카메라·읽기 전용 결과 로딩을 검사한다.
`test_baseline_presets.py`는 수치 override, preset 해석과 샘플 입력의 raw ID 매핑을 검사한다.
소규모 ROI 설정 선택에는 일대일 GT matching을 사용했다. 전체 씬의 미라벨 배경 처리 및
사용자의 detector와의 정식 정량 비교는 아직 수행하지 않았다.
