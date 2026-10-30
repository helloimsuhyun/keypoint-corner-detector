# `corner_detector` package

활성 validator의 평면 추출 경로는 하나다.

```text
query-core organized AHC
  -> per-region RANSAC/full-resolution refit
  -> repeated compatible-plane merge
  -> final convex hull
  -> padded boundary/Hough/3-D finite-edge graph/refiner
```

`query_debug.inspect_query`와 detector NMS seed는 모두
`validator_pipeline.run_validator_pipeline`을 호출한다. 평면 추출 backend 선택이나
별도 plane-cluster 활성화 옵션은 없다.

`cornervins_plane_extraction.py`가 AHC, RANSAC/refit, 병합, hull을 소유하고,
`preprocessing.py`가 organized pixel을 voxel scene-index domain으로 연결한다.
평면 fitting은 exact query radius 안에서만 수행된다.

이전의 multiscale Harris, SAH affinity graph, 독립 Edge3D 진단 경로는 삭제되었다.

## CED Query → Local Extremity 가설 (`ball`)

기존 `detector`와 `validator`는 그대로 유지한다. `ball`은 기존 CED → PCA →
3-D NMS 후보를 먼저 계산한 뒤, 후보별 원본 XYZ ROI에서 정점 가설을 찾는다.
클릭은 가장 가까운 **기존 CED NMS Query**의 결과를 선택한다. Query 위치를
이동하거나 FPS로 새 Query를 만들지 않는다. 프레임별 결과를 캐시하므로 재클릭이나
다시 시각화할 때 수치를 재계산하지 않는다.

```bash
python3 -m corner_detector --mode ball --source stl --input data/laser_pose_cross.stl
python3 -m corner_detector --mode ball --source kinect --input /path/to/sequence
python3 -m corner_detector --mode ball --source realsense
```

STL 경로는 실제 존재하는 파일로 지정한다. 프레임을 확보하면 전체 결과 창이 먼저 열린다.
`CED → PCA 고유값 비율 → NMS` Query는 빨강, 각 Query에 Ball 로직을 적용한
최종 후보는 자홍색으로 **기존 NMS의 Fusion 히트맵 위에** 표시한다.
배경은 기존 NMS 화면과 같은 voxel 점군·유효 마스크·Q99 선형 색상 범위를 사용하고,
Ball 후보는 원본 관측점 좌표를 유지한다. 전체 화면과 클릭 상세는 동일한 설정과
캐시된 결과를 공유한다. CED NMS 반경(`nms_radius_mm`)과 Ball 내부 NMS 반경
(`ball_nms_radius_x × voxel_3d_mm`)은 별도 단계의 파라미터다. 동일한 원본 점의 중복 표지만
생략하며, 가까운 다른 후보를 합치거나 추가 NMS를 적용하지 않는다.
이 후보들은 아직 기하 검증 전 가설이다.

RGB/depth 창을 클릭하면 아래 세 3-D 상세 창도 동시에 열린다.

1. 클러스터링 결과: ROI의 object component별 색상.
2. Ball raw heatmap: NMS 이전 `raw_votes` 전체. 0표는 회색, 양수는 0부터
   최댓값까지 선형 색상으로 표시한다. 제목·터미널에 범위와 0표 비율이 나온다.
   Soft의 소수점 vote도 그대로 표시하며 수치 자체는 변경하지 않는다.
3. NMS 결과: 회색 ROI 위에 최종 **검증 전 정점 가설**을 자홍색으로 표시.

세 창 모두 원본 점군, 빨간 원래 Query, 노란 후보 구, 회색 padding 구를 유지한다.
전체 결과 창은 클릭으로 상세 Query를 바꿔도 유지된다. `--no-3d-view`는 3-D 창을 열지 않는다.
R은 재촬영, Q/Esc는 종료다. 기존 C/H 전환과 별도 2-D ball 요약 창은 제거했다.
터미널에 vote·component·Query
이동 거리(원래 Query에서 가설까지의 거리)가 표시된다. CED 후보가 없으면 새 Query를
만들지 않고 가설을 생성할 수 없다고 알린다.

Query를 선택하면 **Ball point directions** 창도 열린다.

1. 왼쪽 3-D ROI에서 확인할 실제 점을 클릭한다. 낮은 점수로 탈락한 점도 선택할 수 있다.
2. 오른쪽 구면의 기본 **Contribution heatmap**에서 각 방향이 선택점에 실제로 준
   정규화 투표 기여도를 확인한다. 0은 회색이고, 양수는 선형 색상으로 표시한다.
   색상 범위는 점별 `0~최대 방향 기여도`이므로 수치 범례도 함께 확인한다.
   방향별 합은 선택한 scale의 V를 재현하고, 화면에는 저장된 scale V와 전체 V/S도 나온다.
   기본 256개 Fibonacci 방향을 그대로 사용하며 보간 방향은 만들지 않는다.
   메뉴를 **Top 1% directions**로 바꾸면 기존 상위 1% 방향을 금색으로 볼 수 있다.
3. 드래그로 회전하고 오른쪽 버튼 드래그로 확대·축소한다. 여러 반경을 설정했다면
   Scale 메뉴로 비교한다. 선택점의 원본 index·component·방향 수·기존 V/S가 표시된다.

판정은 같은 컴포넌트의 **Padding 포함 확장 ROI**에서 `투영값 ≥ 선형 99분위`다.
동점은 모두 포함하므로 정확히 1%의 점만 표시하는 방식은 아니다. Padding의 점은
방향을 확인할 수 있지만 실제 기여도 히트맵에서는 최종 투표에서 제외되어 0이다.
기여도는 현재 Hard/Dense/Partition/Sparse 방식과 같은 방향별 정규화 및 cutoff를
반영한다. 상위 1% 밖의 점도 Soft 기여도가 양수일 수 있다. 표시 전환은 계산된
진단 배열을 재사용하며 점수·NMS·FP 판정 기준을 바꾸지 않는다.
Query 변경·재촬영·종료 시 진단 창도 정리되며, `--no-3d-view`에서는 열리지 않는다.

ball의 길이는 모두 `voxel_3d_mm` 배수로 설정한다. 연구 기준값은 voxel 10 mm에서
core 3배=30 mm, padding 1배=10 mm, 거리 연결 반경 0.45배=4.5 mm,
peak NMS 0.5배=5 mm다. 실제 사용값은 아래 설정 블록을 따른다.
방향 256개, 최소 방향 비율은 0.04다. 기존 voxel/CED/PCA/NMS 기본값은
변경하지 않았다. 특히 voxel은 10 mm 그대로지만 투표는 원본 관측점을 사용한다.

기본값을 수정할 곳은 `settings.py` 상단 `SettingsDefaults`의 **Ball/extremity256**
블록이다. `ball_radii_x`, `ball_padding_x`, `ball_directions`,
`ball_object_clustering`, `ball_cluster_radius_x`, `ball_nms_radius_x`,
`ball_nms_method`, `ball_min_vote_fraction`을 한곳에서 관리한다.
CLI와 라이브러리 기본 설정이 이 값을 공유한다. 설정을 해석할 때 `_x`에
3-D voxel 크기를 곱해 내부 `ExtremityConfig`의 mm 값으로 변환한다.
`query_radius_x`는 2-D voxel 기준이지만, ball은 **3-D voxel 기준**이다.
직접 사용하는 `ExtremityConfig(radii_mm=...)` 라이브러리 인터페이스는 계속 mm다.
기존 `--ball-*-mm` 길이 옵션은 `--ball-*-x`로 교체했다.

```bash
# Clustering OFF / padding OFF
python3 -m corner_detector --mode ball --source realsense --no-ball-object-clustering --ball-padding-x 0
# Multi-scale: 동일 Query에서 1.2/3배 core (기본 voxel 10 mm이면 12/30 mm)
python3 -m corner_detector --mode ball --source realsense --ball-radii-x 1.2 3
# 같은 CED-fed 경로를 화면 없이 실행 (입력 확보에 필요한 STL/센서 UI는 별도)
python3 -m corner_detector --mode extremity256 --source kinect --input /path/to/sequence --no-3d-view
```

모든 component를 독립적으로 평가하므로 Query가 속하지 않은 작은 물체도 대상이다.
연결성은 instance segmentation이 아니며 접촉 물체는 합쳐지고 낮은 밀도에서는
과분할될 수 있다. 작은 조각을 자동 제거하는 필터는 넣지 않았다. `verified=False`
가설을 확정 코너로 해석하지 않는다.

라이브러리 진입점은 `extremity.refine_ced_candidates(scene, nms_indices, cfg,
raw_points_mm=raw)`다. 각 Query의 독립 결과가 `records`에 저장된다.
`position_mm`는 실제 원본 관측점, `raw_point_index`는 원본 인덱스이고,
`point_index`/`validator_seed_indices`는 가장 가까운 voxel 장면 인덱스다.
기존 `run_validator_candidates`에 `validator_seed_indices`와 원래 voxel 장면 및
organized context를 넘겨 후속 검증할 수 있다. 거친 voxel 때문에 서로 다른 원본
가설이 같은 validator seed에 매핑될 수 있으며 매핑 거리를 함께 기록한다.

## 재현 가능한 합성 실험

```bash
python3 -m corner_detector.extremity_synthetic --seeds 20 --output outputs/extremity_local_run
python3 -m unittest discover -s tests -v
```

16개 실제 XYZ 장면, 기본 3 mm/σ0.12 mm 및 σ0.5 mm/6 mm 밀도 조건을 사용한다.
동일한 CED Query로 A–D와 이상적인 component label 참조 실험을 수행한다.
GT 근처 Query 실험은 별도로 구분한다. `per_query.csv`, `per_scene.csv`,
`ced_baseline.csv`, mean/median/P95 요약 CSV, CPU/thread/config metadata,
GT 좌표 manifest, NPZ, 3-D PNG/회전 가능한 HTML, `REPORT.md`를 생성한다.
기존 결과 디렉터리에 덮어쓰지 않으므로 재실행 시 새 경로를 지정한다.

이번 결과와 한계 분석은 [실험 보고서](../outputs/extremity_local_20261029_final/REPORT.md)에
정리되어 있다. 방법·단위·인덱스 계약은 [methods ledger](../docs/PIPELINE_METHODS.md)를 따른다.

## Quantile Soft Voting 비교

기본 `hard`를 유지하면서 아래 옵션으로 연속적인 vote를 비교할 수 있다.

```bash
# 정확한 partition + 선형 보간 분위수, dense sigmoid
python3 -m corner_detector --mode ball --source realsense --ball-voting-mode soft_partition
# np.quantile 기준 구현
python3 -m corner_detector --mode ball --source realsense --ball-voting-mode soft_dense
# 근사 Sparse (alpha 4/6/8 비교 후 선택)
python3 -m corner_detector --mode ball --source realsense --ball-voting-mode soft_sparse --ball-soft-alpha 8
```

설정은 같은 `SettingsDefaults` 블록의 `ball_voting_mode`, `ball_soft_tau_x`,
`ball_soft_alpha`, `ball_voting_dtype`에서 관리한다. `ball_soft_tau_x=0.025`는
voxel 10 mm일 때 tau 0.25 mm다. `--ball-voting-dtype float32`로 정밀도를 비교한다.
확장 ROI 전체에서 방향별 99% 분위수와 sigmoid를 계산·정규화한 뒤 내부 반경만
남긴다. 위치는 실제 관측점이며 smoothing·가중 좌표 평균은 사용하지 않는다.

```bash
python3 -m corner_detector.soft_voting_benchmark --seeds 100 --output outputs/soft_voting_run
```

7개 장면·6개 노이즈·동일 seed/Query로 Hard, Dense, Partition, Sparse(alpha 4/6/8)를
float32/64 모두 비교한다. 실제 CED Query와 GT 근처 진단 Query를 분리한다.
실험은 30/10 mm ROI, tau 0.25 mm를 명시적으로 사용하며 사용자 설정값을 덮어쓰지 않는다.
Median/P95·검출률·위치 일관성·단계별 시간 CSV/그래프, 1/2/3면 3-D 비교 그림 및
실패 분석을 생성한다. 결과는 [Soft Voting 보고서](../outputs/soft_voting_20261029/REPORT.md)에 있다.

기존 0.04 점수 게이트는 Soft의 분산된 vote를 과하게 제거할 수 있다.
모든 방식에 동일하게 게이트 0을 적용한 대조 실험도 별도로 실행했다.

```bash
python3 -m corner_detector.soft_voting_benchmark --seeds 30 --min-vote-fraction 0 --output outputs/soft_voting_ungated_run
```

게이트 제거는 낮은 vote의 가짜 peak와 NMS 비용도 늘린다. 기본값을 자동으로
바꾸지 않으며, [두 실험의 결론](../outputs/soft_voting_20261029/CONCLUSION.md)을 함께 읽는다.
