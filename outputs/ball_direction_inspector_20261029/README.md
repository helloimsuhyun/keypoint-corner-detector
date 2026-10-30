# Ball 점별 방향 분포 진단 예시

2026-10-29. 실제 XYZ 합성 데이터 `one_face`, 간격 3 mm, 노이즈 0 mm에서
GT 근처 Query `[6, 6, 0]`을 사용한 진단 예시다. 이 예시 Query는 CED 출력이
아니며, GUI의 실제 경로는 기존 CED NMS Query를 그대로 사용한다.

예시 설정은 r=30 mm, padding=10 mm, 연결 반경=4.5 mm, 256방향,
soft_partition, tau=0.25 mm, local-maximum NMS=5 mm, score gate=0.002다.
실제 프로젝트의 `settings.py` 값을 변경하지 않았다. 전체 설정과 Query 출처는
`metadata.json`에 기록했다.

| 선택점 | 좌표 (mm) | Q99 이상 방향 수 | 기존 Ball 점수 |
|---|---|---:|---:|
| 코너 | (0, 0, 0) | 63/256 | 0.086315 |
| 직선 경계 | (15, 0, 0) | 0/256 | 0.001563 |
| 평면 내부 | (15, 15, 0) | 0/256 | 0.0000000703 |
| NMS에 남은 FP | (0, 33, 0) | 0/256 | 0.002337 |

`corner.png`, `edge.png`, `interior.png`, `false_peak.png`는 실제 진단 figure의
headless 렌더다. 선택한 점은 왼쪽 청록색 테두리, 해당 점이 방향별 Q99 이상인
방향은 오른쪽 금색 점/선으로 표시한다. 기존 soft weight가 양수인 방향 전체를
표시하는 것은 아니다. `directions.npz`에는 공통 단위 방향 벡터와 각 점의
`top1_mask`, `projection`, `threshold` 배열이 있다. 수치는 `summary.csv`에 저장했다.

단일 무노이즈 장면의 예시이므로 일반적인 FP 제거 성능을 입증하지 않는다.
이 작업에서는 새로운 FP 필터나 점수를 적용하지 않았다. 실제 Tk/Open3D 창이나
센서는 실행하지 않았으며, GUI 이벤트 연결은 headless mock 테스트로 검사했다.
