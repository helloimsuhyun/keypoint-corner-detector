# Corner keypoint detector

Kinect/RealSense/STL의 organized depth 입력에서 query 단위 평면과 평면 경계
그래프를 분석하는 실험 코드다. 현재 평면 추출은 CornerVINS 방식 하나만 사용한다.

## Kinect fine 실행

```bash
cd /home/choisuhyun/keypoint-corner-detector

python3 -m corner_detector \
  --source kinect \
  --input /home/choisuhyun/datasets/7scenes/redkitchen/seq-03 \
  --frame-index 740 \
  --mode fine
```

별도 backend 또는 plane-clustering 활성화 옵션은 필요 없다. 실행 직후 frozen
RGB/depth Tk 창이 바로 열린다. 그 창에서 query를 클릭하면 exact query radius
안의 organized pixel에 대해 다음 과정이 자동 실행되며, 다른 위치를 다시 클릭해
query를 변경할 수 있다.

```text
AHC segmentation -> region RANSAC/refit -> repeated plane merge -> convex hull
```

기본적으로 이어서 padded 2-D boundary/Hough와 3-D plane-edge graph도 실행한다.
평면 추출과 1/4–4/4 시각화만 보고 싶으면 다음 옵션을 추가한다.

```bash
--no-plane-edge-graph --no-plane-edge-debug-2d
```

표준 평면 시각화는 모두 전체 scene을 유지하면서 query ROI와 반경을 함께 표시한다.

1. Coarse AHC region (Feng Algorithm 3)
2. Refined AHC region (Feng Algorithm 4)
3. RANSAC을 통과하고 least-squares refit된 평면과 normal
4. 병합이 끝난 최종 평면과 convex hull

AHC block은 mm가 아니라 image pixel 단위다. 앞면 평면 근사에서 block side가
`b`, depth가 `z`, focal length가 `fx, fy`이면 실제 폭은 대략
`b*z/fx`, `b*z/fy` mm다.

## 주요 기본값

| 항목 | 기본값 |
|---|---:|
| 3-D voxel | 10 mm |
| query radius | 100 mm |
| AHC block | 3 pixels |
| local minimum region | 100 pixels |
| RANSAC inlier threshold | 15 mm |
| minimum inlier ratio | 0.90 |
| merge angle / distance | 5° / 20 mm |
| projection halo / maximum radius | 50 mm / 150 mm |

이전의 `--multiscale-harris`, `--sah-graph-analysis`, `--edge3d` 진단 경로와
관련 설정은 삭제되었다.

상세 수식과 데이터 계약은 [docs/PIPELINE_METHODS.md](docs/PIPELINE_METHODS.md),
모듈 경계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참고한다.
