# `corner_detector` package

`settings.py`의 `fusion_mode`로 Fusion 방식을 선택한다.

```python
fusion_mode = "local_minimum"  # 현재 선택: CED × Normal PCA lambda1의 Min-Max Local Minimum
# fusion_mode = "pca_ratio"     # 기존 CED × weighted PCA ratio
local_minimum_tau = 0.005
local_minimum_fusion_threshold = 0.2
post_nms_lambda1_ratio_max = 0.20  # 현재 사용자 설정; None이면 OFF
```

기존 모드는 기존 `fusion_score_threshold`를 사용한다. 새 모드는 Lab과 같이
`L=(주변 최대 lambda1 - 현재 lambda1)/(주변 최대-최소 lambda1 + tau*R_PCA²)`이며,
Normal PCA 반경과 min/max 반경은 모두 PCA 반경이다. CED/bilateral 모드는 기존 설정을 따른다.

```bash
python3 -m corner_detector --source demo --mode detector --fusion-mode local_minimum
python3 -m corner_detector --source realsense --mode detector --fusion-mode local_minimum --post-nms-lambda1-ratio-max 0.2
```

후단 게이트는 **NMS 후 CED 가중 PCA의 가장 큰 고유값 / PCA 반경² ≤ 상한**만 남긴다.
같으면 통과하고 weighted PCA가 invalid이면 제외한다. NMS를 다시 하거나 탈락한 최대점의
이웃을 대신 뽑지 않으며, 통과한 원래 scene ID만 Q10·후단 validator로 전달한다.
새 점수에 쓰는 Normal PCA와 게이트에 쓰는 CED 가중 PCA는 구분한다.

Lab의 **Demo 형상**을 카메라나 입력 파일 없이 detector에서 실행할 수 있다.

```bash
python3 -m corner_detector --source demo --mode detector
python3 -m corner_detector --source demo --mode detector --no-3d-view
```

`corner_lab_open3d_cone.py`와 동일한 seed 7의 세 직교 평면·코너점 12,601개를
메모리에서 생성한다. 원본 XYZ/RGB는 같고, voxel·CED·PCA·Fusion·NMS·Q10/MaxGap은
현재 `settings.py`와 CLI 설정을 사용한다. Lab의 점수나 검출 설정을 복사하지 않는다.
`--only`도 사용할 수 있다. 현재 demo는 detector 모드 전용이며 파일을 저장하지 않는다.

`total`은 **detector → 모든 NMS query의 기존 validator → scene 최종 코너/기여 에지**를 자동 실행한다.

```bash
python3 -m corner_detector --mode total --source stl --input /path/to/model.stl
```

결과 창은 두 개뿐이다: (1) detector Fusion 히트맵/NMS, (2) 원본 회색 점군 위
최종 코너(빨강)/Boundary 에지(초록), 기여 공통 교선(청록), 실제 연장(노란 점선).
5 mm validator voxel에서 코너 마커 반경은 4 mm, edge 막대 반경은 0.75 mm다.
R은 재촬영, Q/Esc는 두 창 종료이며 `--no-3d-view`는 계산/로그만 실행한다.
각 NMS query에서 생성한 Corner 후보 전체를 모으고, query 간 병합/NMS는 추가하지 않는다.
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

Detector는 voxel과 독립적인 CED 반경을 기준으로 PCA·bilateral 공간 범위를 설정한다.
아래는 반경 관계를 설명하는 설정 예이며, 사용자가 조정한 실제 초기값은 `settings.py`를 따른다.

```python
centroid_radius_mm = 40.0  # CED 탐색 반경
pca_radius_factor = 1.5              # PCA = CED × 1.5
pca_ced_cutoff = 0.0                 # PCA 이웃의 원본 CED 최소값; lab처럼 쓰려면 0.10
bilateral_mode = "ced"              # "ced": CED 필터 / "fusion": Fusion 필터 / "off": 생략
bilateral_radius_factor = 1.0        # bilateral 반경 = CED × 1.0
bilateral_sigma_spatial_factor = 0.5 # 공간 sigma = CED × 0.5
bilateral_sigma_ced = 0.05           # 선택한 CED/Fusion 점수 차이의 sigma, 거리 단위 없음
nms_radius_factor = 0.6              # NMS 반경 = CED × 0.6
```

CLI는 `--centroid-radius-mm`, `--pca-radius-factor`다. CED를 40 mm로 바꾸면
기본 PCA는 60 mm, bilateral 반경은 40 mm, 공간 sigma는 20 mm가 된다.
`bilateral_mode` / `--bilateral-mode`는 `ced`, `fusion`, `off` 중 하나를 선택한다.
`ced`는 CED를 필터링한 뒤 PCA와 곱하고(기존 기본 동작), `fusion`은 원본 CED × PCA를
모두 계산한 다음 **Fusion 점수 자체를 필터링**한다. `off`는 필터 없이 원본 CED × PCA다.
PCA의 가중치는 세 모드 모두 원본 CED이며, threshold/NMS는 최종 Fusion을 사용한다.
`pca_ced_cutoff` / `--pca-ced-cutoff`는 0~1의 무차원 값이다. PCA 이웃의 원본 CED가
이 값 이상인 점만 가중 공분산에 사용한다(같으면 포함). 기본 0.0은 기존 양수 CED 지지를 유지하고,
0.10이면 CED <0.10인 이웃을 제외한다. 쿼리 자신의 CED를 탈락시키는 조건은 아니며,
남은 지지가 부족하면 PCA/Fusion이 invalid가 된다. CED 자체와 bilateral CED는 바꾸지 않는다.
기존 `bilateral_enabled` boolean은 제거했다.
공간 범위는 `--bilateral-radius-factor`, `--bilateral-sigma-spatial-factor`로 설정하며
공간 mm 입력과 `None` 분기는 제거했다. `--bilateral-sigma-ced`는 선택한 CED/Fusion 점수 차이의
허용 폭이므로 길이 배수로 바꾸지 않는다. 독립 입력 `--pca-radius-mm`는 제거했다.
NMS도 `--nms-radius-factor`로 설정한다. CED 40 mm / factor 0.6이면 NMS 24 mm이며,
factor 1.0이면 CED와 NMS 반경이 같다. `--nms-radius-mm`와 `None` 분기는 제거했다.
최소 이웃 수는 `min_neighbors` 설정을 사용한다(현재 기본값 5). voxel은 점군 샘플링만 바꾸며
탐색 반경을 자동으로 바꾸지 않는다. 공유 detector를 쓰는 validator/total의 Coarse 및
ball/extremity 쿼리에도 적용한다. 기본 detector 자체의 다중 스케일 재검출은 수행하지 않는다.
이전 `--centroid-radius-x`는 별도 fine 모드에서만 사용한다.

Lite 전체 검사는 `q10_validator_enabled`로 제어하며, refine ON에서 **Q10 보정 → 단위구 RANSAC → 평면별 양방향성 검사**를 실행한다.

```python
q10_validator_enabled = True  # 전체 후단 검사
q10_refine_enabled = True     # Q10 위치 보정 + q10_min_q10 게이트
```

`q10_refine_enabled=False` / `--no-q10-refine`면 **위치 보정과 최소 Q10 판정을 함께 생략**한다.
Q10/CC 계산과 analysis voxel 스냅도 하지 않으며, 원래 NMS 좌표에서 RANSAC·MaxGap 검사만 실행한다.
전체 검사를 끄려면 기존처럼 `q10_validator_enabled=False` / `--no-q10-validator`를 사용한다.
Refine OFF에서도 MaxGap으로 탈락할 수 있고, 후단 plane validator에는 ACCEPT 후보만 전달한다.
디버그에는 refine/min gate OFF를 표시하고 Q10 임계선과 방향 화살표는 그리지 않는다.

아래 Q10 보정·최소 Q10 조건은 refine ON일 때 적용한다.
전체 raw cloud를 `q10_voxel_mm`로 한 번 줄이고, 원래 NMS점을 중심으로 NMS 반경 안에서만 한 번 탐색한다.
`q10_direction_mode`의 축으로 비가중 cosine Q10을 계산해 유효 voxel 중심점을 선택한다.
최소 10개 support와 초기 support 수의 80% 이상을 요구하며, 검색 창은 이동하지 않는다.

Q10의 구형 query CC는 `q10_refine_cc_enabled` / `--[no-]q10-refine-cc`로 제어한다.
ON이면 전체 R 이웃의 non-self median NN × `q10_cc_factor`를 연결 반경으로 쓰고,
실제 XYZ(mm)에서 query 성분을 선택한다. OFF이면 전체 R 이웃으로 Q10을 계산하며 NN/그래프를 생략한다.
현재 기본값은 `settings.py`를 따른다. cloud·반경·CC·방향 설정이 같은 프레임에서 수치 결과를 캐시한다.

최종 위치의 전체 R 이웃을 단위방향으로 바꿔 원점을 지나는 대원을 최대 3개 RANSAC으로 맞춘다.
각 평면은 모든 방향을 독립적으로 분류하므로 교차점은 여러 평면에 소속될 수 있다.
각 평면의 투영 각도를 원형으로 정렬해 `q10_gap_link_deg` 이내 이웃을 연결하고, 점이 가장 많은 그룹을 유지한다.
현재 lab과 동일하게 10° bin을 사용해 **희소한 양 끝 bin만** 제거한다. 기준은
`max(2, ceil(점이 있는 bin 점 수 중앙값 × 0.5))`이며, 내부의 희소한 bin은 유지한다.
정리한 각도의 `span = 360° - 최대 빈 각도`를 계산한다.
정리 후 점이 남아 있으면 개수·유지 비율 조건 없이 span으로 판정한다.
어느 평면이든 **span ≥ `q10_gap_edge_span_deg`면 REJECT**한다. 모든 평면에 점이 남고
span이 기준 미만이면 CORNER다. 평면을 못 찾거나 정리 후 빈 평면이 있으면 UNKNOWN이다.
RANSAC 최소 지지점 조건은 유지하며, 최종 단계에 별도 3D CC는 없다.

RANSAC 설정은 기존 `q10_sphere_*`를 유지한다(현재 detector 허용각은 settings.py 참조).
새 판정은 `q10_gap_link_deg`, `q10_gap_edge_span_deg`,
`q10_gap_tail_enabled`, `q10_gap_tail_bin_width_deg`,
`q10_gap_tail_density_fraction`, `q10_gap_tail_min_bin_count`로 조절한다.
CLI는 `--q10-gap-*`, tail 토글은 `--[no-]q10-gap-tail`이다.
기존 `q10_gap_min_points`, `q10_gap_min_retained_fraction` 설정과 CLI는 삭제했다.
기존 N+/N-/B 계산과 축 탐색, beta·양쪽 최소점·B 임계값·축 간격 설정/CLI는 삭제했다.
수치 함수는 `spherical_antipodal.py`에 있으며 standalone GUI를 import하지 않는다.

Refine ON에서 최종 품질은 보정 trace의 같은 Q10을 재사용한다. Q10 < `q10_min_q10`이면 **RANSAC 전에 즉시 탈락**한다(같으면 통과).
Q10이 유효하지 않으면 RANSAC을 생략하고 INDETERMINATE로 유지한다. 미달점의 진단에는 `RANSAC SKIPPED`를 표시한다.
Q10 또는 양방향성의 알려진 실패가 있으면 REJECT, 두 검사가 모두 통과하면 ACCEPT,
나머지는 INDETERMINATE로 유지한다. 이전 `--q10-threshold` 별칭도 제거했다.
원래 CED·PCA·Fusion·NMS 좌표/scene ID는 유지하고 수렴 좌표는 별도로 저장한다.

```bash
python3 -m corner_detector --mode detector --source realsense
python3 -m corner_detector --mode detector --source realsense --q10-sphere-iterations 300 --q10-gap-link-deg 20 --q10-gap-edge-span-deg 160
python3 -m corner_detector --mode detector --source realsense --no-q10-validator
```

진단은 저장된 모든 평면의 소속·유지 마스크를 읽는다. **P1 주황 / P2 초록 / P3 보라**이며,
여러 평면 소속점은 **흰색**, 어느 소속 평면에서도 유지되지 않은 점은 **30% 밝기**로 표시한다.
파랑은 평가 제외 voxel, 회색은 어느 평면에도 소속되지 않은 평가점이다.
점·투영점·arc는 같은 평면색을 사용하고, Q10 방향 화살표만 청록색이다.
최종 검사는 연결성 검사가 아니므로 이 색으로 3D 연결 여부를 주장하지 않는다.

원래 NMS는 빨강, 수렴점은 주황, 최종 통과/탈락/판단불가는 초록/빨강/주황으로 표시한다.
RealSense의 최종 결과창은 원래 RGB 점군 위에 **최종 통과 코너만 초록색**으로 표시한다.
Q10 마커는 refinement 후 좌표를 사용하며, `--only`에서도 이 RGB 최종 뷰는 유지한다.
각 마커를 클릭하면 단계 선택 슬라이더, 수렴 경로, 단위구·Q10 히스토그램,
RANSAC 대원과 평면별 정리한 arc·span·유지점 수를 볼 수 있다.
오른쪽 장미도는 평면별 원래 10° 각도 분포와 같은 평면색의 유지 arc, 어두운 × 표시의 제외점을 보여준다.
대원·투영점·장미도의 색은 평면별로 같으며, 교차점은 해당하는 모든 평면에 표시한다.
카드의 span/gap/유지점/전체·tail 제외점/판정은 저장값을 그대로 읽고, 근거가 부족하면 UNKNOWN이다.
상단에는 Q10·양방향 검사를 구분하고 하단에는 각도·임계값·미소속/중복 소속 수를 표시한다.
단계 슬라이더를 바꿔도 최종 평면 검사는 고정된다. 별도 native 3D 경로 창에서도 좌우 키로 단계를 본다.
`--no-q10-debug`는 클릭 디버그를 끄고 수치 결과는 유지한다.

```bash
python3 -m corner_detector --mode detector --source realsense --only
```

`validator`는 **Coarse NMS → 설정에 따른 Q10 필터·수렴 → 시작점 반경의 원본 점 crop → 로컬 voxel → guided RANSAC**을 실행한다.
`validator_fine_*`와 Fine 재점수화·NMS·Top-K·추가 클릭은 제거했다. validator 내부에서는 전달받은 시작 좌표를 유지한다.
반경은 `validator_ransac_radius_x × validator_ransac_voxel_mm`, voxel은
`validator_ransac_voxel_mm`으로 설정한다. 2번 패널에는 회색 voxel 입력 점군과 점 개수를 표시한다.
RANSAC 뒤에는 support 재수집 → 평면 Hull →
유한 교선·경계 에지 → Endpoint 기반 Corner 후보 생성·통합을 실행한다.

Corner 단계는 `validator_corner_graph.generate_corners(edges, min_angle_deg=20.0,
corner_radius_mm=5.0)`로 독립 실행할 수 있다. 각도 필터 → 무한 직선 교점/최근접점 →
양쪽 Edge의 가까운 Endpoint까지 거리 검사 → 공통 Edge·동일 Endpoint를 쓰는 후보 통합 순서다.
통합 그룹의 모든 원래 교점 간 거리가 반경 이내여야 하며, Endpoint Gap이 가장 작은
후보의 위치를 그대로 사용한다. 평균이나 추가 LS 보정은 없다.

사용자 조절값은 `validator_min_angle_deg`, `validator_corner_radius_mm` 두 개이며,
CLI는 `--validator-min-angle-deg`, `--validator-corner-radius-mm`다. 독립 함수의 실험 초기값은
20°/5 mm이고, 현재 작업 트리에서 사용자가 조정한 CLI 기본값 45°/15 mm는 유지한다.
3D 꼬인 직선의 교차 오차는 Edge가 가진 기존 fitting tolerance로 검사한다.
결과 `corners`는 `position`, `edge_ids`, `endpoint_ids`, `intersection_residual`,
`endpoint_gap`을 담은 목록이다. Endpoint ID는 각 Edge의 0=start, 1=end다.

query 거리 gate, 별도 외삽 허용량, Vertex LS·랭킹·최종 하나 선택은 제거했다.
모든 후보를 반환하며 `validation_status`는 `CANDIDATES` 또는 `NO_CANDIDATES`,
`verified=False`다. 실제 물체 Corner인지에 대한 구조 검증은 하지 않는다.
유효 Edge의 원래 양 끝 좌표를 사용하므로 ROI clipping이나 마스킹 관련 끝점도
후보를 만들 수 있다. Edge 추출·TLS·겹침 마스킹 자체는 바꾸지 않는다.

주 창은 기존 Coarse 선택·query-local voxel·RANSAC 평면 패널을 유지한다.
보조 창은 4: Hull/clipping/TLS/Before–After, 5: 평면 교선,
6: 원본 Edge → 겹침 마스킹 → Endpoint Corner 통합 → 원본 raw 점군 위 모든 후보다.
Corner와 Boundary 기여 Edge는 초록, 공통 교선은 청록, Endpoint 연결은 노랑으로 표시한다.
각 창은 중앙 visualization manager로 관리하며 숫자 결과를 변경하지 않는다.
`total` 모드도 query당 모든 후보와 연결 Edge를 수집하고 query 간 통합은 하지 않는다.

Padded Hull과 core clipping, 원래 ACTIVE/CLIPPED Node 정보 및 마스킹 구간은 보존한다.
이 정보로 평면·Edge를 다시 계산하거나 Corner 단계에서 추가 구조 검증을 하지 않는다.
`validator_roi_padding_mm`는 voxel과 독립적인 추가 반경이며 support는 core+padding이다.
4번 Before/After 비교는 같은 RANSAC 모델로 Hull·TLS·Edge·Corner만 다시 계산한다.
S-key PNG 저장과 R/Q/Esc 창 제어는 유지한다. `--no-validator-geometry`는 평면 추출까지만 실행한다.

```bash
python3 -m corner_detector --mode detector --source stl --input /home/choisuhyun/keypoint-corner-detector/data/laser_pose_box.stl
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

## 저장 데이터셋에서 detector / Lite 실행

프로젝트 루트에서 실행한다. 단일 장면 또는 데이터셋 전체를 입력할 수 있다.

```bash
python3 -m corner_detector --source dataset --mode detector --input corner_dataset_tool/dataset/scene_0001
python3 -m corner_detector --source dataset --mode detector --input corner_dataset_tool/dataset
```

기본은 저장된 ROI이며 `annotations.json`이 필요하다. 전체 점군은 `--dataset-region scene`,
화면 없이 계산/로그만 실행하려면 `--no-3d-view`를 붙인다. `--only`, `--q10-*` 등 기존 검출기 설정을 공유한다.
전체 폴더는 장면 번호순으로 처리하며, 화면을 닫으면 다음 장면으로 진행한다. 전체 중지는 `Ctrl+C`다.
원본 XYZ/RGB·ROI·GT는 읽기만 하고 GT로 검출하거나 설정을 조절하지 않는다.
결과는 화면/로그로 확인하며 파일 저장은 추가하지 않는다. 현재 dataset source는 detector 모드만 지원한다.

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

Q10 방향축은 `settings.py`의 `q10_direction_mode` 또는
`--q10-direction-mode unit_mean|unit_median|centroid`로 선택한다.
기본 `unit_mean`은 기존 단위방향 벡터 평균, `unit_median`은 단위방향 벡터의
성분별 중앙값, `centroid`는 같은 Q10 이웃 점군의 원래 XYZ 무게중심 방향이다.
각 축을 정규화한 뒤 기존 cosine 하위 10% 값을 계산한다. 무게중심 모드도
분위수 자체에는 가중치를 추가하지 않는다. 축이 정의되지 않으면 기존처럼
INDETERMINATE로 처리한다. 선택 축은 캐시되어 최종 Q10 판정에도 그대로 쓰인다.
클릭 디버그의 실제 voxel 점군에는 선택 위치에서 R 길이의 굵은 청록색 화살표가
표시되며, step 변경에 따라 움직인다. 화살표 길이는 시각화용 R이다.

별도 `--mode heatmap`과 Weighted Local Cone 설정/분기/전용 화면은 제거했다.
detector의 CED·PCA·Fusion 계산, NMS 및 점수 heatmap 화면은 유지한다.

`validator`와 `total`도 detector와 같은 Q10 ON/OFF 및 세부 설정을 따른다.
Q10 ON이면 ACCEPT 후보만 수렴 좌표를 validator 시작점으로 쓴다.
REJECT와 INDETERMINATE는 제외하며, ACCEPT가 없으면 validator 후보도 0개다. Q10 OFF이면 원래 NMS 좌표다.
원래 scene ID·점군·NMS 배열은 유지하고 별도 시작 좌표를 전달한다.
클릭 마커·픽킹·ROI·RANSAC 모두 같은 시작 좌표를 사용한다.

```bash
python3 -m corner_detector --mode validator --source realsense --q10-validator
python3 -m corner_detector --mode total --source realsense --q10-validator
python3 -m corner_detector --mode validator --source realsense --no-q10-validator
```

새 `new-validator` 모드는 기존 validator와 같은 raw crop → local voxel → guided
RANSAC → support/CC → 평면별 2D UV 투영까지만 실행한다. Convex Hull·Edge·Corner는
계산하지 않으며, 별도 창에서 평면별 U/V(mm) 점군을 확인한다. 기존 validator는 유지한다.

```bash
python3 -m corner_detector --mode new-validator --source realsense
python3 -m corner_detector --mode new-validator --source realsense \
  --new-validator-ransac-voxel-mm 5 --new-validator-ransac-radius-x 15 \
  --new-validator-roi-padding-mm 10 --new-validator-plane-connect-radius-x 4
```

`settings.py`의 `new_validator_*` 12개 파라미터는 기존 `validator_*`와 독립적이며,
초기값만 현재 validator 값과 같다. `*_x`는 새 모드의 RANSAC voxel에 대한 배수다.
detector/Q10 설정은 공유하며 Q10 ON일 때 최종 ACCEPT의 수렴점만 시작점으로 쓴다.
각 평면의 2D query→무게중심 방향을 기준으로 cosine Q10을 계산한다.
공통 `new_validator_min_q10` / `--new-validator-min-q10`은 초기값 0.10이다.
평면이 1/2/3개이면 각각 1/2/3개 모두 Q10 >= 임계값이어야 ACCEPT다.
하나라도 미달이면 REJECT, 점 부족·무게중심 방향 미정·평면 0개는 판단 불가로 통과하지 않는다.
이 판정은 2D 분포 검사이며 Hull·Edge·Corner는 계속 계산하지 않는다.
