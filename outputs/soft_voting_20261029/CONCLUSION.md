# Quantile Soft Voting 구현 및 검증 결론 — 2026-10-29

## 연구 질문에 대한 답

**이번 실험에서 Soft Voting이 Hard보다 전반적으로 강건한 localization을 제공한다는 근거는 얻지 못했다.**
방향별 가중치는 연속적이지만, 마지막 관측점 peak 선택은 여전히 이산적이다.
관측 간격 3 mm, tau 0.25 mm에서 기본 1·2·3면 코너는 두 방식이 대개 같은 실제 관측점을 선택했다.
정점 주변 샘플 누락에서는 일부 추가 검출이 있었지만, 오검출 증가와 함께 해석해야 한다.
평균 좌표나 Gaussian smoothing을 넣어 오차를 인위적으로 줄이지 않았다.

**Partition 최적화는 Dense 정확도를 유지하면서 빨라졌다.**
Sparse는 sigmoid 수를 크게 줄였지만 전체 시간 개선은 제한적이며, 낮은 vote의 후보 집합은 바뀔 수 있었다.
따라서 정확한 대체가 필요하면 `soft_partition`, Sparse는 명시적인 근사 비교 옵션으로 두는 것이 맞다.
라이브 기본값은 기존 `hard`를 유지했다.

## 공정한 비교 조건

- 기존 CED → PCA → NMS 코드는 그대로 호출했다. 같은 점군에서 생성한 동일 CED Query를 모든 방식에 전달했다.
- GT 근처 Query 실험은 별도 진단이며 실제 detector에 추가하지 않았다.
- 7개 XYZ 장면, 노이즈 σ=0/0.05/0.1/0.2/0.35/0.5 mm.
- r=30 mm, padding=10 mm, 256 방향, tau=0.25 mm, component 연결 4.5 mm, 국소 NMS 5 mm.
- 기본 게이트 0.04 실험은 조건별 **100 seed**, 동일 게이트 0 대조 실험은 **30 seed**.
- Hard / np.quantile Dense / Partition Dense / Sparse alpha=4,6,8 모두 float32/64로 비교했다.
- 매칭은 5 mm 이내 1:1 대응. 오류는 매칭된 관측에 한정하므로 recall·FP를 함께 제시했다.
- 노이즈별 오류 그래프는 해당 노이즈의 매칭 관측을 합친 Median/P95다. 아래 표는 별도로 명시한 조건별 통계의 macro 평균이다.
- CPU: AMD Ryzen 7 8845HS. BLAS/OpenMP thread 수는 metadata에 기록했다. 생성·매칭·I/O·그리기 시간은 제외했다.

사용자가 설정한 라이브 반경·클러스터·NMS 값은 덮어쓰지 않았다.
두 실험의 seed 수가 다르므로 서로 다른 게이트의 수치 차이를 순수한 게이트 효과의 정밀 추정치로 해석하지 않는다.

## 100-seed, 게이트 0.04: 실제 CED Query

아래는 장면·노이즈 조건별 통계의 macro 평균이다. 시간은 query당 평균 ms다.

| 방법 (float64) | 5 mm recall | FP/frame | Query 시간 | 모든 Query 추가 시간/frame |
|---|---:|---:|---:|---:|
| Hard | 0.627 | 0.41 | 0.619 ms | 2.582 ms |
| Dense, np.quantile | 0.529 | 0.40 | 1.267 ms | 4.973 ms |
| Dense, partition | 0.529 | 0.40 | 1.064 ms | 4.225 ms |
| Sparse, alpha=8 | 0.529 | 0.40 | 0.928 ms | 3.720 ms |

Partition은 Dense 대비 전체 query 시간 약 **1.19배**, 분위수 계산은 약 **1.91배** 빨랐다.
분위수 오차는 점검한 입력에서 0 mm였고, Dense/Partition의 vote map L1 차이와 후보 집합 차이도 없었다.
두 순위값 사이 선형 보간을 유지했다. 설치된 NumPy 1.21.5도 이미 내부에서 partition을 사용하므로,
이를 “전체 정렬을 제거해 얻은 이득”으로 설명하면 잘못이다.

기존 0.04 게이트에서는 Soft의 vote가 여러 점으로 나뉘어 2·3면 코너, 두 코너 장면 등에서 탈락했다.
GT-near macro recall도 Hard 0.857, Soft 0.143이었다. 이 결과를 숨기거나 Soft만 임계값을 낮추지 않았다.
오류 곡선의 P95가 낮아 보이더라도 검출된 GT 집합이 다르므로 강건성 개선의 증거가 아니다.

## 30-seed, 동일 게이트 0: peak 자체의 비교

| 방법 (GT-near, float64) | Recall | FP/query | 매칭 오차 median의 macro 평균 |
|---|---:|---:|---:|
| Hard | 0.870 | 3.22 | 0.505 mm |
| Partition Dense Soft | 0.883 | 17.28 | 0.672 mm |

Soft는 기본 1·2·3면과 한 ROI의 두 코너를 다시 검출했지만, 낮은 vote의 peak도 다수 남았다.
같은 GT에 둘 다 성공한 관측만 비교하면 median의 macro 평균은 Hard 0.5051 mm, Soft 0.4986 mm로 차이가 작았다.
추가로 회복된 누락-정점 GT는 더 큰 오차를 가지므로 전체 매칭 오차 median의 macro 평균은 오히려 커졌다.
GT-near FP는 약 5.4배였다. vote threshold와 NMS 비용까지 포함한 trade-off가 중요하다.

Sparse alpha=8은 Dense와 매칭된 GT 위치가 같았지만 전체 후보 Jaccard는 약 0.9699였다.
alpha=4는 약 0.7599로 더 많이 바뀌었다. 무시한 작은 꼬리 가중치도 낮은 peak의 순위와 존재 여부에 영향을 준다.
alpha=8을 Dense와 완전히 같은 알고리즘으로 취급하지 않는다.

## 정밀도와 시간

float32/64의 공통 성공 GT 관측점 위치는 이번 실험에서 같았다.
이는 모든 입력에서 보장되는 성질이 아니며, 근소한 동점이나 cutoff 부근에서는 차이가 날 수 있다.
게이트 0.04의 Partition query 시간은 float64 1.064 ms, float32 1.004 ms였다.

voxel 전처리 + 기존 CED/PCA/NMS + 모든 Query의 Partition Soft를 합친 지연은
median **24.093 ms**, P95 **32.995 ms**였다. median 역수는 약 41.5 FPS에 해당한다.
이는 최대 수천 점의 합성 장면을 사용한 CPU 수치 파이프라인이며 capture·I/O·렌더링은 제외했다.
사용자가 별도로 설정한 더 큰 ROI나 전체 센서 프레임에서의 실시간 성능을 검증한 결과는 아니다.

## Heatmap 진단과 수정

확인한 Hard 합성 장면에서는 ROI 점의 약 98%가 0표였다. 선형 0..max 색상에서는 대부분 같은 색이 되는 것이 정상이다.
이제 0표는 회색, 양수 raw vote mass는 실제 양수 최솟값..최댓값의 로그 색상으로 표시한다.
창 제목·터미널에 범위와 0표 비율을 표시하며 소수점 soft mass를 정수로 자르지 않는다.
원본 vote·정규화 점수·NMS에는 표시용 변환이 적용되지 않는다.
0표가 많은 장면은 여전히 회색 점이 많아야 하며, 없는 차이를 색으로 만들지 않았다.

## 실패 원인 및 다음 사용 판단

1. 0.04 per-point 게이트는 Hard와 Soft에서 같은 검출 의미를 갖지 않는다. Soft의 낮아진 peak 높이를 고려하지 않으면 누락된다.
2. 게이트를 제거하면 내부 core 경계에 작은 soft peak들이 남아 FP와 NMS 비용이 증가한다.
3. Euclidean connectivity는 outlier rejection이 아니다. 고립된 outlier component는 모든 방식에서 강한 자기 vote를 받는다.
4. 정점 관측이 없으면 반환 좌표는 주변의 실제 샘플에 제한된다. 이번 구현은 없는 점을 복원하지 않는다.
5. 사용자 데이터에서 검증하기 전에는 기존 Hard를 유지하는 편이 합리적이다. Soft를 비교할 때 정확한 기준 구현은 `soft_partition`이다.
6. CED가 Query를 만들지 못한 영역이나 모든 Query core 바깥의 정점은 이 국소 모듈로 복구하지 못한다.

## 결과 파일

- [주 실험 전체 표](REPORT.md), [조건별 summary](summary.csv), [paired 비교](paired_comparisons.csv)
- [게이트 0 대조 실험](../soft_voting_20261029_ungated/REPORT.md)
- [Median/P95 그래프](noise_localization.png), [검출률/FP](noise_detection.png), [위치 일관성](repeatability.png)
- [단계별 시간](stage_times.png), [trade-off](tradeoff.png)
- [1면 3-D 비교](one_face_voting.png), [2면](two_faces_voting.png), [3면](three_faces_voting.png)

검증 범위: headless 수치·시각화 생성·테스트. 실제 STL GUI, RealSense, Kinect 장비 검증은 수행하지 않았다.
