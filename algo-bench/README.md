# algo-bench

어그리게이션·패스파인딩 알고리즘 예제와 Bellman-Ford 벤치마크 프로젝트입니다.

## 요구 사항

- **Python 3.8+**
- **필요 라이브러리**: `requirements.txt` (차트 생성 시 `matplotlib` 필요, `pip install -r requirements.txt`로 설치)

## 설치 및 실행

### 1. 저장소 받은 뒤 가상환경 + 의존성 설치

```bash
cd algo-bench
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 실행 방법

**프로젝트 루트** (`algo-bench/`) 에서 실행합니다. 출력 파일은 `data/`(CSV), `chart/`(PNG)에 저장됩니다.

```bash
source .venv/bin/activate
python src/aggregation_pathfinding_examples.py
python src/test_aggregation_pathfinding.py
python src/chart_pathfinding.py
python src/chart_aggregation.py
python src/benchmark_vs_bellman_ford.py
python src/chart_benchmark_vs_bf.py
```

가상환경을 켜지 않고 실행하려면 `python` 대신 `.venv/bin/python`을 사용할 수 있습니다.  
**참고**: 차트 스크립트(`chart_*.py`)는 `matplotlib`가 없으면 CSV만 저장하고 PNG는 생성하지 않습니다.

### 3. 파일별 실행 명령

| 파일 | 실행 | 출력 |
|------|------|------|
| aggregation_pathfinding_examples.py | `python src/aggregation_pathfinding_examples.py` | (콘솔 출력) |
| test_aggregation_pathfinding.py | `python src/test_aggregation_pathfinding.py` | (테스트 결과) |
| chart_pathfinding.py | `python src/chart_pathfinding.py` | `data/chart_pathfinding.csv`, `chart/chart_pathfinding.png` |
| chart_aggregation.py | `python src/chart_aggregation.py` | `data/chart_aggregation.csv`, `chart/chart_aggregation.png` |
| benchmark_vs_bellman_ford.py | `python src/benchmark_vs_bellman_ford.py` | `data/benchmark_vs_bellman_ford.csv` |
| chart_benchmark_vs_bf.py | `python src/chart_benchmark_vs_bf.py` | `chart/benchmark_vs_bellman_ford.png` (입력: `data/benchmark_vs_bellman_ford.csv`) |

`cd src` 후 `python <파일명>.py` 로 실행해도 됩니다. 이때도 CSV/PNG는 프로젝트 루트의 `data/`, `chart/`에 저장됩니다.

## 각 파일이 하는 일

| 파일                                | 하는 일                                                                      |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| aggregation_pathfinding_examples.py | 어그리게이션·패스파인딩 알고리즘 예제 실행 (데모 출력)                       |
| chart_pathfinding.py                | 패스파인딩 알고리즘 비교 차트 생성 → `data/chart_pathfinding.csv`, `chart/chart_pathfinding.png` |
| chart_aggregation.py                | 어그리게이션 Basic vs Optimized 비교 차트 생성 → `data/chart_aggregation.csv`, `chart/chart_aggregation.png` |
| test_aggregation_pathfinding.py     | 위 알고리즘들 입력/기대 출력 검증용 테스트 실행                              |
| benchmark_vs_bellman_ford.py        | Bellman-Ford 대비 효율 벤치마크 (S1~S12, C1/C2/C5/C6/C8) → `data/benchmark_vs_bellman_ford.csv` |
| chart_benchmark_vs_bf.py            | 벤치마크 CSV(`data/`)로 시나리오×알고리즘 실행 시간 차트 생성 → `chart/benchmark_vs_bellman_ford.png` |
