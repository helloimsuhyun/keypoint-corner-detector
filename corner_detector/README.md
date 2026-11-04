# `corner_detector` package

`total`은 **detector → 모든 NMS query의 기존 validator → scene 최종 코너/기여 에지**를 자동 실행한다.

```bash
python3 -m corner_detector --mode total --source stl --input /path/to/model.stl
```

결과 창은 두 개뿐이다: (1) detector Fusion 히트맵/NMS, (2) 원본 회색 점군 위
최종 코너(빨강)/Boundary 에지(초록), 기여 공통 교선(청록), 실제 연장(노란 점선).
5 mm validator voxel에서 코너 마커 반경은 4 mm, edge 막대 반경은 0.75 mm다.
R은 재촬영, Q/Esc는 두 창 종료이며 `--no-3d-view`는 계산/로그만 실행한다.
각 NMS query에서 기존 validator가 선택한 코너 하나를 모으고, 별도의 병합/NMS는 추가하지 않는다.
설정은 기존 detector·`validator_*` 값을 공유한다. `--no-validator-geometry`와 함께 사용할 수 없다.
KD-tree를 공유하고 Before/After·organized 픽셀 매핑·후보별 긴 로그를 생략한다.
중간 query 결과는 쌓지 않고 최종 기하/상태만 보관하며, 에지·코너 mesh는 색상별로 묶어 그린다.
속도 배수는 측정하지 않았다. 아래 `validator`의 단일 query 디버그 동작은 유지한다.

기본 `detector`는 CED → bilateral → weighted PCA → Fusion → 3-D NMS까지만 실행한다.
후단 validator는 자동 실행하지 않으며 후보는 기하 검증 전 상태다. `--only`는 NMS
화면만 표시하는 옵션으로 유지한다. `fine`과 별도 `validator` 모드는 그대로 사용할 수 있다.

detector와 validator의 Coarse Fusion 히트맵은 **프레임 전체 Coarse Fusion Q99**를
공통 색상 상한으로 사용한다. 같은 점수는 같은 색으로 표시하며 invalid 점은 회색이다.
Coarse 점수·threshold·NMS는 유지한다.

`validator`는 **Coarse NMS → query 반경의 원본 점 crop → 로컬 voxel → guided RANSAC**을 실행한다.
`validator_fine_*`와 Fine 재점수화·NMS·Top-K·추가 클릭은 제거했다. query 좌표는 그대로 유지한다.
반경은 `validator_ransac_radius_x × validator_ransac_voxel_mm`, voxel은
`validator_ransac_voxel_mm`으로 설정한다. 2번 패널에는 회색 voxel 입력 점군과 점 개수를 표시한다.
RANSAC 뒤에는 support 재수집 → 평면 Hull →
유한 교선·경계 에지 → 코너 검증을 실행한다. **서로 다른 방향의 유한 에지 두 개 이상**이
만나고 원래 Coarse query 중심 gate를 통과하면 `ACCEPT`다. 최종 좌표는 `corner_position_mm`,
지지 에지는 `corner_edges`와 `corner_edge_ids`에 저장된다. 터미널에도 좌표와 에지 끝점을 출력한다.
에지·코너 주변의 Point Support 거리/최소 개수 검증은 제거했다. 평면 support와 Hull은 유지하며,
최종 ACCEPT는 남은 기하 조건 통과를 뜻한다. `point_support_checked=False`로 구분한다.
`validator`와 `total`의 최종 코너는 통과한 후보 중 공통 교선 포함 → 직선 피팅 RMS →
필요한 총 외삽 길이 → query 거리 순으로 선택한다. 유효 교선 후보가 없으면 Boundary-only를 사용한다.
Boundary-only도 모든 기여 Line으로 3D 최소제곱 위치를 계산하며, 처음 두 선의 중간점에 고정하지 않는다.
기본 Line 거리/외삽/그룹 범위는 각각 5 mm, query gate는 10 mm다.
기존 3-D 세 번째 패널은 RANSAC 점군·평면·법선만 표시한다. 후속 결과는 별도 창으로 나눈다.

- **5번:** 여러 평면 사이의 유한 교선 에지만 청록색으로 표시한다.
- **4번:** Padded Hull → ROI clipping → TLS/Node 상태 → Before/After를 표시한다(기존 7번 이동).
- **6번:** 선택한 NMS query 하나에 대해 2×2의 네 패널로 과정을 표시한다.
  **6-1:** 평면별 원본 Boundary(RGB)와 공통 교선(청록).
  **6-2:** 남은 구간(회색/청록), 제거 구간(자홍색 점선), overlap 거리 반경(파란 반투명 원통).
  **6-3:** 실제 사용한 끝점 연장(노란 점선), 교점 후보(흰 점), LS 추정 코너와 선택된 기여 에지(초록).
  **6-4:** 원본 raw 점군(회색) 위 선택 코너/Boundary 기여 에지(초록), 기여 공통 교선(청록), 실제 연장(노란 점선).
  Boundary와 공통 교선의 기여 ID/개수를 따로 표시하며, 공통 교선이 없는 Boundary-only 코너도 명시한다.
  LS 코너와 edge 직선 사이 잔차는 얇은 회색 연결선으로 구분한다.

Validator는 선택한 후보 하나만 계산한다. 전체 scene을 자동 순회하지 않으며,
창 순서는 4 → 5 → 6이고 6번이 마지막이다. 7번 창은 없다.

교선과 중복되는 Boundary는 원래 TLS 직선을 보존한 채 구간 마스크로 제거한다.
마스크 끝점은 코너로 사용하지 않는다. 교선 위 Vertex 그룹은 구속 LS로, 여러 교선의
공통 Vertex는 3D Multi-Line LS로 정밀화하고 기존 각도·유한 구간·query gate를 다시 검사한다.
독립 overlap/vertex 임계값은 `settings.py`의 `validator_edge_overlap_*`,
`validator_vertex_*` 및 대응 CLI에서 조절한다.

5번의 회색 Hull 외곽선은 패치 범위이며 확정된 에지가 아니다.
4번은 ROI 구와 Node 상태를, 6번은 Edge 처리 및 최종 코너를 표시한다.
각 창은 독립적으로 닫을 수 있고, 재촬영·종료 시 전체를 정리한다.
`validator_roi_padding_mm=10.0`은 voxel과 독립적인 추가 반경이며,
support 반경은 정확히 core+padding이다(현재 75+10=85 mm).
기존 support 반경·voxel 배수 padding·바깥 경계 margin 설정은 제거했다.
Padded Hull의 각 선분을 core와 평면의 정확한 단면 원으로 먼저 자른 뒤 TLS 병합한다.
양 끝이 밖이어도 core를 관통하는 선분은 유지하고 ROI 테두리 연결선은 만들지 않는다.
원래 내부 Hull Node는 ACTIVE, 잘라 생긴 Node는 INACTIVE이며 병합 후에도 상태를 유지한다.
같은 평면의 코너는 공통 ACTIVE 끝점을 공유하는 이웃 Line에만 허용한다.
평면 간 교선/LS 경로와 query gate 10 mm는 유지하며, 최종 코너도 core 안이어야 한다.
4번 창은 선택한 query의 A: padded Hull, B: clipping, C: TLS/끝점 상태,
D: core-only Before와 padded After를 보여준다. D의 카메라는 함께 움직인다.
S를 누르면 `outputs/validator_roi/query_<id>_<timestamp>/`에
`A.png`, `B.png`, `C.png`, `D_before_after.png`를 저장한다.
Before 비교는 동일 RANSAC 모델/core voxel로 Hull·TLS·그래프만 추가 계산한다.
유한 padding으로 가림 경계나 내부 Hull bridge까지 판별할 수는 없다.
`--no-validator-geometry`는 기존 평면 추출까지만 실행한다.

```bash
python3 -m corner_detector --mode validator --source stl --input /path/to/model.stl
```

기존 평면·그래프 validator의 평면 추출 경로는 하나다.

```text
query-core organized AHC
  -> per-region RANSAC/full-resolution refit
  -> repeated compatible-plane merge
  -> final convex hull
  -> padded boundary/Hough/3-D finite-edge graph/refiner
```

`query_debug.inspect_query`와 라이브러리에서 명시적으로 검증하는 detector NMS seed는
`validator_pipeline.run_validator_pipeline`을 호출한다. 평면 추출 backend 선택이나
별도 plane-cluster 활성화 옵션은 없다.

`cornervins_plane_extraction.py`가 AHC, RANSAC/refit, 병합, hull을 소유하고,
`preprocessing.py`가 organized pixel을 voxel scene-index domain으로 연결한다.
평면 fitting은 exact query radius 안에서만 수행된다.

이전의 multiscale Harris, SAH affinity graph, 독립 Edge3D 진단 경로는 삭제되었다.

## CED Query → Local Extremity 가설 (`ball`)

`ball`은 `detector`와 동일한 기존 CED → PCA →
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
