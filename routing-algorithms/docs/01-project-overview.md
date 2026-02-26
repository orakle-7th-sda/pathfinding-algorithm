# 프로젝트 개요

## 목적

본 프로젝트는 DEX(탈중앙화 거래소) 애그리게이터에서 사용하는 라우팅/분할 알고리즘을 Python으로 구현하고, 동일한 데이터 모델(`Pool`, `SwapRequest`, `SwapResult`) 위에서 비교·확장 가능한 형태로 유지하는 것을 목표로 한다.

## 프로젝트 구조

```
routing-algorithms/
├── algorithms/
│   ├── base.py                 # 공통 데이터 모델/베이스 클래스
│   ├── single/                 # 단일 경로 라우팅
│   │   ├── naive.py
│   │   ├── bfs_routing.py
│   │   ├── dijkstra.py
│   │   ├── a_star.py
│   │   └── k_best.py           # (신규) top-k 후보 재평가 라우팅
│   └── composite/              # 복합 경로/분할
│       ├── simple_split.py
│       ├── greedy_split.py
│       ├── multi_hop.py
│       ├── dp_routing.py
│       ├── convex_split.py
│       └── kkt_split.py        # (신규) KKT 기반 최적 분할
├── mock_data/
│   └── data_generator.py
├── examples/
│   └── educational_examples.py
├── main.py
├── requirements.txt
└── docs/
```

## 알고리즘 구성

- **Single Path (5개)**
  - Naive Brute Force, BFS Routing, Dijkstra Routing, A* Search Routing
  - **K-Best Routing**: spot 기준 top-k 후보를 뽑은 뒤 실제 `amount_out`으로 재평가

- **Composite Path (6개)**
  - Simple Split, Greedy Split, Multi-Hop Routing, DP Split Routing, Convex Optimization Split
  - **KKT Optimal Split**: constant-product AMM 가정에서 KKT 조건으로 빠르게 최적 분할

## 실행 예시

프로젝트 루트에서:

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "routing-algorithms")

from mock_data.data_generator import DataGenerator
from algorithms.base import SwapRequest
from algorithms.single import KBestRouting
from algorithms.composite import KKTOptimalSplit

pools = DataGenerator(seed=42).generate_standard_test_pools()
request = SwapRequest("ETH", "USDC", 250.0)

print(KBestRouting(pools).execute_with_timing(request).to_dict())
print(KKTOptimalSplit(pools).execute_with_timing(request).to_dict())
PY
```

## 이번 고도화 요약

- `algorithms/single/k_best.py` 추가
  - 단순 spot 최적 경로만 고르는 방식의 한계를 줄이기 위해, 후보 경로 다건 평가 방식을 도입했다.
- `algorithms/composite/kkt_split.py` 추가
  - 수치 미분 기반 최적화 대비 더 빠르고 안정적인 분할 계산을 제공한다.
