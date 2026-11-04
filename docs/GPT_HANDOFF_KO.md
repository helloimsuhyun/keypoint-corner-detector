# 다른 GPT에 전달할 현재 개발 브리핑

기준: 2026-10-30 (guided validator Fine 단계 제거 반영), `/home/choisuhyun/keypoint-corner-detector`의 활성 코드.
확인한 HEAD: `ce93bc7` (`sevaral changes`). 이 문서는 해당 시점의 스냅샷이며,
정식 수식·변경 이력은 `docs/PIPELINE_METHODS.md`, 구조 계약은
`docs/ARCHITECTURE.md`에서 관리한다. 아래 내용을 다른 GPT에 그대로 전달할 수 있다.

---

나는 3-D point cloud에서 코너/keypoint를 찾는 실험용 검출기를 개발하고 있다.
현재 코드를 이해한 상태에서 알고리즘과 다음 개발 단계를 논의해 달라.
아래에서 구현된 기능, 실험적 가설, 아직 구현되지 않은 기능을 구분해 달라.

## 1. 전체 상태와 모드 구분

활성 패키지는 `corner_detector/`, 실행 진입점은
`python3 -m corner_detector` → `runner.main`이다.
입력은 STL 기반 합성 관측, Kinect 저장 RGB-D, RealSense다.
입력 어댑터는 `pointcloud_inputs/`, 전처리는 `preprocessing.py`가 담당한다.
모든 내부 길이·좌표는 mm이며, 입력 depth 범위 CLI는 이름대로 m이다.

현재 다음 경로가 공존한다. CLI 기본 모드는 `fine`이다.

| 모드 | 실제 처리 | 결과의 의미 |
|---|---|---|
| `fine` | 클릭 query → organized AHC → 평면 → boundary/Hough → 3-D graph → refiner | query-local 평면·유한 에지·코너 |
| `detector` | 전역 CED/PCA/Fusion/NMS에서 정지; 후단 validator 자동 실행 해제 | 기하 검증 전 후보와 dense 진단 화면 |
| `detector --only` | 동일 계산 후 NMS 화면만 표시 | 기하 검증 전 후보 |
| `total` | 동일 detector/NMS → 모든 query의 기존 guided validator → 최종 코너/기여 에지 집계 | ACCEPT query당 기존 선택 코너 하나와 기여 active edge 구간; 후처리 병합 없음 |
| `validator` | Coarse NMS → raw query crop → local voxel → guided RANSAC → Hull/edge/corner | 최대 3개 평면과 지지된 유한 에지 기반 코너 |
| `ball`, `extremity256` | 기존 CED NMS query별 원본 XYZ 방향 투표 → local peak NMS | `verified=False` 정점 가설 |

**기존 `fine` 모드는 유지하고, `validator` 내부 Fine 단계는 제거했다.**

`total`은 결과 창 두 개만 연다: detector Fusion/Q99/NMS와 원본 회색 점군 위
최종 코너(빨강)/Boundary(초록)·기여 공통 교선(청록)·실제 연장(노란 점선).
5 mm validator voxel에서 코너 반경 4 mm, edge 막대 반경 0.75 mm다.
장면 KD-tree를 detector/validator가 공유하고, organized 픽셀 매핑 및 Before/After
비교를 생략하며, 배치 query는 상태/최종 기하만 보관한다. 색상별 mesh를 묶어 그린다.
기존 수식·샘플링·임계값은 유지하고 새 검출 hyperparameter는 없다.
R은 재촬영, Q/Esc는 두 창 종료다. 테스트/벤치마크/실제 GUI는 실행하지 않았다.
두 모드는 서로 다른 평면/에지 계산 경로를 사용한다.
두 경로의 공통 composition 진입점은 `validator_pipeline.run_validator_pipeline`이고,
설정 객체의 타입으로 분기한다.

## 2. 공통 CED → PCA → Fusion → NMS

주요 함수:

- `edge_suppression.dense_ced_weighted_pca_scores`
- `candidate_extraction.extract_corner_candidates`
- `dense_heatmap.run_dense_heatmap_mode`

각 점 p의 strict-radius 이웃으로 산술 중심 μ를 계산한다. query 자신도 포함한다.
CED는 `clip(||p-μ|| / r_CED, 0, 1)`이다. 최소 이웃 수 미달은 invalid다.
다음으로 공간 거리와 CED 차이를 사용하는 bilateral CED를 계산한다.

PCA는 **원래 CED가 양수인 유효 점의 원래 CED**를 covariance weight로 쓴다.
Bilateral 결과를 PCA weight로 대체하지 않는다.
고유값을 λ1 ≥ λ2 ≥ λ3로 두고 `r=clip(λ2/λ1, 0, 1)`을 계산한다.
포화가 켜져 있으면 `r_sat=τ*tanh(r/τ)`, 꺼져 있으면 `r_sat=r`이다.
최종 점수는 `Fusion=clip(bilateral_CED*r_sat, 0, 1)`이다.
따라서 일반 Harris response와 동일한 검출기라고 설명하면 안 된다.

Coarse/detector의 현 설정은 다음과 같다.

| 설정 | 실제 기본값 |
|---|---:|
| scene voxel | 10 mm |
| CED radius | 8 × scene voxel = 80 mm |
| PCA radius | 1.5 × CED radius = 120 mm |
| bilateral radius / spatial sigma / CED sigma | 80 mm / 40 mm / 0.05 |
| 최소 이웃 / PCA saturation τ | 6 / 0.5 |
| Fusion threshold | finite valid 점수의 global mean |
| NMS radius | 50 mm |

후보 조건은 항상 strict `Fusion > threshold`다.
Coarse NMS는 **radius local maximum**이며 Greedy NMS가 아니다.
inclusive 반경 안의 다른 eligible 점보다 점수가 작으면 탈락하고,
동점에서는 작은 scene index를 우선한다.

## 3. `validator`: Coarse NMS → raw query ROI → local voxel → guided RANSAC

Fine 재점수화/threshold/NMS/Top-K/재선택/fallback은 제거했다.
`validator_fine.py`, `validator_fine_*` 설정/CLI와 관련 캐시·결과 필드도 제거했다.
기존 `--mode fine`은 별도 모드이며 유지한다.

1. 기존 detector와 동일한 CED/bilateral/PCA/Fusion/threshold/NMS로 Coarse 후보를 만든다.
2. 기존 Coarse NMS 후보를 고르면 그 좌표를 이동시키지 않고 원본 raw XYZ를 반경으로 crop한다.
3. crop한 뒤 `validator_ransac_voxel_mm`으로 voxel downsample한다.
4. 그 core voxel만 guided sequential RANSAC에 넣는다. 첫 샘플만 Gaussian 거리 확률,
   나머지 둘은 균등 추출이며, 정수 inlier 수 → RMS 순위와 hard removal을 유지한다.
5. 평면별 support 재수집 → 연결 성분/Hull → 유한 교선·경계 에지 → 코너 판정을 실행한다.

반경은 `validator_ransac_radius_x × validator_ransac_voxel_mm`이다.
이 변경 시점의 설정은 voxel 5 mm, radius 15×=75 mm, inlier gate 1×=5 mm이다.
사용자가 설정을 편집하므로 실행 시 `settings.py`의 실제 값을 우선 확인해야 한다.
support 반경은 core + `validator_roi_padding_mm`(기본 절대 10 mm)이다. 이를
별도 raw shell로 voxel화해 이어 붙이며 RANSAC에는 넣지 않는다.
core 밖 점이 core centroid에 섞이지 않는다. Padded Hull의 원래 선분을 정확한
평면/core 단면 원으로 먼저 자른 후 TLS 병합한다. 관통 선분은 유지하고 ROI 테두리를
따라 새 선분을 만들지 않는다. 원래 내부 Hull Node는 ACTIVE, 새 clipping Node는
INACTIVE로 끝점 ID/출처와 함께 병합 후까지 계승한다. 같은 평면의 Corner는 공통
ACTIVE 끝점을 공유하는 이웃 Line에만 허용하며 비활성 끝점은 Vertex/외삽 근거가 아니다.
비활성 끝점 Line도 내부 교선 근거로는 유지한다. 최종 코너도 core 안이어야 한다.
기존 support 반경·배수 padding·바깥 margin 설정은 제거했다.
서로 다른 grid는 경계 부근 가까운 샘플을 가질 수 있다.

canonical entry는 `run_validator_pipeline`이고 Coarse `query_index`를 받는다.
`roi_raw_point_indices`는 raw ID, `*_ransac_indices`는 query별 `ransac_points_mm`의 ID,
`*_point_indices`는 nearest-Coarse scene adapter다. 여러 native 점이 같은 scene ID에 대응할 수 있다.
최종 코너 좌표는 `corner_position_mm`, 이를 지지하는 에지는 `corner_edges`/`corner_edge_ids`다.
최종 선택은 유효 공통 교선 포함 여부 → 직선 RMS 잔차 → 총 외삽 길이 → query 거리 순이다.
Boundary-only 그룹도 모든 기여 Line으로 3D LS 정밀화하고, 병합 시 전체 Line으로 다시 계산한다.
교선이 하나일 때는 기존 교선 구속 LS를 유지한다. 모든 기존 기하 gate를 통과한 후보만 비교한다.
실험용 30/30/30/50 mm였던 거리/외삽/그룹/query gate 기본값은 5/5/5/10 mm로 좁혔다.
이 정책은 `validator`와 `total`에 공통 적용되며 Point Support 검증은 계속 비활성화다.
서로 다른 방향의 유한 에지 2개 이상이 query gate 안에서 기하학적으로 만나야 ACCEPT다.
에지·코너 Point Support 거리/점수 검증과 두 관련 설정은 제거했다. 평면 support/Hull은 유지한다.
에지 지지점 필드는 빈 값, quality는 중립값 1이며 `point_support_checked=False`다.

3-D main panel은 (1) Coarse Fusion/NMS, (2) 회색 로컬 voxel 입력,
(3) 평면 피팅이다. Coarse를 한 번 클릭하면 즉시 RANSAC까지 실행한다.
별도 창 4는 평면 교선, 5는 평면별 경계 에지, 6은 통합 코너/지지 에지다.
시각화는 저장된 결과만 읽으며 모든 창은 manager가 관리한다.
현재 교선과 중복되는 Boundary 구간은 union mask로 비활성화하고 원본 TLS 선분을 보존한다.
마스크 절단점은 Vertex 근거가 아니다. 교선별 5 mm 전체 범위 그룹에 구속 LS,
여러 교선에 3D Multi-Line LS를 적용하고 기존 corner gate를 재검사한다.
Validator는 선택한 Coarse NMS query 하나만 canonical pipeline으로 처리한다.
전체 후보 자동 순회는 없다. 6번은 원본 평면별/공통 에지 → 제거 구간 →
실제 사용한 끝점 연장·교점 후보·LS 추정 코너 → 원본 점군 위 선택 코너/기여 에지를
2×2의 네 패널로 표시한다. 6-3/6-4 노랑 점선은 설정상의 전체 허용 길이가 아니라
추정 코너의 edge 방향 투영에 실제 필요한 외삽 구간이며, 잔차는 얇은 회색 연결선이다.
6-4는 원본 raw 점군을 프레임 시작에 한 번 올리고 query 변경 시 최종 overlay만 교체한다.
선택 코너의 Boundary 기여 에지는 초록, 공통 교선 기여 에지는 청록으로 구분하고
타입별 기여 ID/개수를 표시한다. Boundary-only 코너에는 기여 공통 교선이 없다고 명시한다.
창 순서는 4 → 5 → 6이고 6번이 마지막이다. 기존 7번 디버그 화면은 4번으로 옮겼고,
기존 평면별 Boundary 전용 화면은 제거했다. 새 4번은 동일 query의
padded Hull / clipping / TLS 상태 계승 / Before-After를 표시한다.
Before는 같은 RANSAC 모델과 core voxel의 support/Hull/TLS/그래프만 추가 계산하며
최종 결과에 영향을 주지 않는다. D 카메라는 연동되고 S로 A/B/C 및 D 비교 PNG를
`outputs/validator_roi/query_<id>_<timestamp>/`에 저장한다. 실제 GUI/저장은 미검증이다.

## 4. `fine` / 명시적 라이브러리 호출의 기하 검증 경로

기본 `detector`의 자동 연결은 해제되었다. `DetectorConfig.validator=None`이 기본이고,
라이브러리에서 명시적으로 `ValidatorConfig`를 전달하면 기존 검증을 실행할 수 있다.
이 경로는 `ValidatorConfig`를 받고 organized RGB-D context가 필요하다.
Unorganized XYZ만 있을 때 AHC용 image adjacency를 만들어 주는 fallback은 없다.

```text
정확한 query ball 안의 organized pixel
  → image block 기반 AHC segmentation/refinement
  → region별 deterministic RANSAC + full-resolution inlier 재분류 + LS refit
  → compatible patch 반복 병합 + convex hull
  → padded tangent projection + continuous 2-D boundary evidence
  → Gaussian boundary weights + weighted Hough
  → seed별 연속 (rho, theta) 최적화 + support profile에서 유한 선분
  → 원래 support의 3-D PCA/TLS + finite-edge complete-link merge
  → plane adjacency + graph-local robust plane/edge refinement + 코너
```

기본 query radius는 `10 × voxel_2d_mm(10)=100 mm`이고 projection halo는 50 mm다.
Plane fitting은 query core 안에서만 한다. Halo는 downstream boundary context만 보충한다.
AHC block은 3 **pixels**, ROI 최소 region은 100 pixels, region RANSAC hypothesis voxel은
50 mm, full-resolution gate는 15 mm, 최소 inlier ratio는 0.90이다.
Patch merge 기준은 unoriented normal angle < 5도, symmetric separation < 20 mm다.
이 숫자는 새 guided RANSAC의 5 mm voxel / 2.5 mm gate와 별개다.

`corner_2d.py`는 density 보정된 continuous boundary statistic을,
`boundary_support.py`는 `w=exp(-0.5*(distance/sigma)^2)`를 담당한다.
`line_refinement_2d.py`는 Hough seed의 두 파라미터만 최적화하며,
endpoint는 support profile에서 결정한다. 3-D 복원은 관측 support에 PCA/TLS를 적용한다.
`refiner.py`는 연결 graph 단위로 Huber 기반 plane/edge refinement와 corner 생성을 수행한다.
기존 Hough/PCA 결과와 초기·최종 diagnostics를 보존한다.
서로 다른 query에서 나온 최종 에지·코너를 scene 전체에서 중복 제거하는 단계는 없다.

## 5. Ball/extremity 실험 경로

이미 있는 CED NMS query마다 raw XYZ의 padded ROI를 별도로 평가한다.
선택적으로 Euclidean connected components를 만들고 component마다 256개 Fibonacci
방향으로 투영하여 extremity vote를 모은다. Query가 속하지 않은 component도 평가한다.
FPS query 생성, query 이동, query 간 vote 합산은 하지 않는다.

Hard는 방향별 argmax 관측점 하나에 투표한다.
Soft는 방향별 linear Q99를 기준으로 sigmoid weight를 계산하고 padded support 전체에서
합을 1로 정규화한 다음 core 밖의 mass를 버린다.
`soft_dense`는 참조 구현, `soft_partition`은 같은 수식의 partition 구현,
`soft_sparse`는 cutoff를 둔 근사다. 최종 위치는 실제 raw 관측점이고 좌표 평균은 없다.
마지막에 query-local peak NMS를 적용하지만 결과는 검증 전 가설이다.

현재 `SettingsDefaults`의 실제 Ball 기본값은 다음과 같다.

- core 100 mm, padding 25 mm, component 연결 반경 15 mm, NMS 반경 30 mm.
- 256 directions, Greedy NMS, 최소 vote fraction 0.
- voting=`soft_partition`, tau=0.25 mm, dtype=`float64`.

README의 30/10/4.5/5 mm와 hard/0.04 설명은 연구 기준·과거 기본값이다.
현재 live 기본값으로 인용하면 안 된다. 접촉한 물체가 합쳐지거나 희박한 표면이
조각나는 component 한계가 있고, 관측점만 반환하므로 미관측 꼭짓점을 복원하지 않는다.
후속 검증용 `validator_seed_indices`를 만들지만 자동으로 검증하는 경로는 아니다.

## 6. 변경할 때 반드시 지킬 계약

- `query_index`는 전달한 Coarse voxel scene의 ID다. 원본 click/raw ID와 교환하면 안 된다.
- 기존 plane/projection/boundary/edge/refiner public support ID는 scene domain을 유지한다.
- 새 validator의 RANSAC ID는 query별 로컬 cloud를 가리키며 Coarse scene adapter와 구분한다.
  Plane mask는 저장된 RANSAC ROI를 가리킨다.
- Nearest-Coarse adapter는 many-to-one일 수 있다. Adapter ID로 native 좌표나 support count를 대체하지 않는다.
- Ball의 raw/local/scene ID도 구분한다. 원본 가설 여러 개가 같은 voxel seed로 매핑될 수 있다.
- 각 알고리즘의 소유 모듈을 지키고 composition은 `validator_pipeline.py`에 둔다.
- 시각화는 `visualization/manager.py`를 통해 접근한다.
  2-D plot은 `plots.py`, 3-D geometry는 `view3d.py`, interaction은 `query_ui.py` 소유다.
- Diagnostics는 저장 수치를 변경하지 않는다. Display cache 교체와 알고리즘 재계산을 구분한다.
- `/past/`는 참조·분석·수정·import하지 않는다.
- 숫자 변경 요청이 없으면 수치 동작을 보존한다. 변경 전 파일/함수와 보존할 불변식을 명시한다.
- 알고리즘/관련 설정/시각화 의미 변경 시 `docs/PIPELINE_METHODS.md`도 같은 작업에서 갱신한다.

## 7. 검증 범위

Fine 제거 변경은 raw crop→voxel 순서, centroid, halo 분리, 원래 query 유지,
캐시/인덱스 연결, CLI/import와 headless UI 이벤트·화면 분리를 회귀 확인한다.
사용자 요청에 따라 지침 15번의 synthetic geometry 실험은 수행하지 않는다.
실제 STL GUI/RealSense/Kinect 장치 경로는 실행하지 않았다.
정확한 실행 결과는 이번 변경의 완료 보고를 참고한다.

## 8. 다음 논의에서 요청하는 것

먼저 위 구현 상태를 전제로 질문에 답해 달라.
새 `validator`의 평면 이후 노드/에지/코너 단계를 수정할 때는 이미 있는 경로의 재사용 가능성과
두 경로의 서로 다른 support/index/평면 추정 의미를 함께 검토해 달라.
구현 제안은 현재 존재하는 기능과 새로 추가해야 할 기능을 명시하고,
수치 변경이 있으면 score·support·좌표·단위·결과 의미에 미치는 영향을 설명해 달라.
새 알고리즘의 구체적인 요구는 별도로 정하며, 이 브리핑만으로 코드 변경을 승인한 것은 아니다.
