# Ball 방향별 실제 기여도 히트맵

2026-10-29. `Ball point directions`의 기본 표시를 실제 방향별 투표 기여도
히트맵으로 추가했다. Query를 선택한 뒤 왼쪽 ROI 점을 클릭하면 오른쪽 단위
구면의 256개 방향에 해당 점이 받은 정규화 투표량을 색으로 표시한다.
`Top 1% directions` 메뉴는 이전의 Q99 포함 여부를 별도로 보여 준다.

히트맵은 방향별로 선택점에 배정된 표다. 선택점의 방향들끼리 합을 1로
정규화하지 않는다. Hard는 실제 argmax의 0/1표, Soft는 방향별 전체 탐색점
가중치 합으로 나눈 표이며 Sparse는 cutoff도 반영한다. Candidate Radius 밖의
점(Padding 포함)은 실제 기여도가 0이다. 점별 색상 범위는 선형 `0~최대 방향
기여도`다. 서로 다른 점의 같은 색이 같은 수치를 뜻하지 않으므로 범례와
합계를 같이 확인해야 한다. 저장된 점수와 NMS에는 변화가 없다.

예시는 이전 Q99 진단과 동일한 `one_face` 무노이즈 XYZ 장면과 GT 근처 Query
`[6,6,0]`을 사용한다. 예시 Query는 CED 출력이 아니다. 실제 GUI는 기존 CED
NMS Query를 그대로 사용한다. r=30 mm, padding=10 mm, 256방향,
soft_partition, tau=0.25 mm, 연결 반경=4.5 mm, local-maximum NMS=5 mm,
gate=0.002의 고정 예시이며 프로젝트 설정을 변경하지 않았다.

| 선택점 | 상위 1% 방향 수 | 방향 기여도 합 | 저장된 scale V | 최대 방향 기여도 |
|---|---:|---:|---:|---:|
| 코너 | 63 | 22.096626 | 22.096626 | 0.419112 |
| 직선 경계 | 0 | 0.400022 | 0.400022 | 0.076934 |
| 평면 내부 | 0 | 0.000018 | 0.000018 | 0.000014 |
| NMS에 남은 FP | 0 | 0.598219 | 0.598219 | 0.092793 |

기여도 합과 저장된 V의 최대 차이는 약 `3.6e-15`다. 상위 1% 방향이 없어도
Soft 표가 누적될 수 있음을 보여 주는 단일 장면 예시이며, 일반적인 FP 제거
성능이나 새 판정 기준을 주장하지 않는다.

- `*_heatmap.png`: 코너·직선 경계·평면 내부·FP의 실제 기여도 구면 히트맵.
- `*_top1.png`: 같은 선택점의 기존 Q99 이진 표시.
- `direction_contributions.npz`: 방향 벡터와 점별 `direction_votes`,
  `direction_scores`, `projection`, `threshold`, `top1_mask` 배열.
- `summary.csv`, `metadata.json`: 합계 검증 수치와 설정/Query 출처.

Hard/Dense/Partition/Sparse, float32/64에서 방향별 수식과 scale V/최종 점수
재구성을 검사했다. 전체 120개 headless 테스트와 CLI/import 검사를 통과했다.
실제 Tk/Open3D GUI·센서는 실행하지 않았다.
