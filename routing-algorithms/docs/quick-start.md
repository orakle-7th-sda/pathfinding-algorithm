# Quick Start Guide

## 5분 만에 시작하기

### 1. 즉시 실행 (의존성 없음)

```bash
cd routing-algorithms
python3 main.py --quick
```

교육용 알고리즘 예제가 즉시 실행됩니다.

### 2. DEX 벤치마크 실행

```bash
python3 main.py
```

DEX 라우팅 알고리즘 성능 비교가 실행됩니다.

### 3. 전체 기능 (차트 포함)

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 모든 벤치마크 실행
python3 main.py --all
```

## 주요 명령어

| 명령어 | 실행 내용 | 소요 시간 |
|--------|-----------|-----------|
| `python3 main.py --quick` | 교육용 알고리즘 데모 | ~10초 |
| `python3 main.py` | DEX 벤치마크 (기본) | ~1분 |
| `python3 main.py --bellman-ford` | BF 효율성 분석 (S1-S12) | ~2-3분 |
| `python3 main.py --extreme` | 극한 테스트 (5000 ETH) | ~30초 |
| `python3 main.py --all` | 모든 벤치마크 | ~5분 |

## 출력 확인

벤치마크 실행 후 다음 위치에서 결과 확인:

```bash
# 결과 JSON
cat output/results/summary.json

# 차트 (matplotlib 설치 시)
ls output/charts/

# Bellman-Ford 분석
cat output/results/bellman_ford_analysis.csv
```

## 예제 코드

### 단일 알고리즘 사용

```python
from algorithms.single import DijkstraRouting
from algorithms.base import SwapRequest
from mock_data.data_generator import DataGenerator

# 데이터 생성
generator = DataGenerator(seed=42)
pools = generator.generate_standard_test_pools()

# 알고리즘 실행
algo = DijkstraRouting(pools)
request = SwapRequest("ETH", "USDC", 10.0)
result = algo.execute_with_timing(request)

print(f"Output: {result.total_amount_out:.2f} USDC")
print(f"Time: {result.execution_time_ms:.4f} ms")
```

### 여러 알고리즘 비교

```python
from algorithms.single import DijkstraRouting, BFSRouting
from algorithms.composite import ConvexSplit

algorithms = [
    DijkstraRouting(pools),
    BFSRouting(pools),
    ConvexSplit(pools)
]

for algo in algorithms:
    result = algo.execute_with_timing(request)
    print(f"{algo.metadata.name}: {result.total_amount_out:.2f}")
```

### 교육용 예제

```python
from examples.educational_examples import (
    pathfinding_dijkstra,
    pathfinding_bellman_ford,
    build_sample_graph
)

graph = build_sample_graph()

# Dijkstra
path1, cost1 = pathfinding_dijkstra(graph, "A", "E")
print(f"Dijkstra: {path1}, cost={cost1}")

# Bellman-Ford
path2, cost2 = pathfinding_bellman_ford(graph, "A", "E")
print(f"Bellman-Ford: {path2}, cost={cost2}")
```

## 문제 해결

### ImportError 발생 시

```bash
# Python 경로 확인
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 main.py
```

### matplotlib 없을 시

차트는 생성되지 않지만, 결과 JSON은 정상 출력됩니다.

```bash
# matplotlib 설치
pip install matplotlib numpy
```

### 느린 실행 시간

대형 시나리오는 시간이 걸립니다:
- S7-S9 (200 노드): 각 30초-1분
- `--quick` 옵션으로 빠른 테스트 가능

## 다음 단계

1. [전체 README](README.md) 읽기
2. [통합 요약](INTEGRATION_SUMMARY.md) 확인
3. [문서](docs/) 탐색
4. 커스텀 벤치마크 작성

---

**즐거운 라우팅 알고리즘 탐험되세요!**
