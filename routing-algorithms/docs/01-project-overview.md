# 프로젝트 개요

## 목적

본 프로젝트는 DEX(탈중앙화 거래소) 애그리게이터에서 사용되는 패스파인딩 및 애그리게이팅 알고리즘을 Python으로 구현하고, extreme_benchmark에서 정의한 시나리오로만 벤치마크하여 성능과 출력량을 비교하는 것을 목표로 한다.

## 프로젝트 구조

```
dex-aggregator-algorithms/
├── algorithms/                 # 라우팅/분할 알고리즘 구현
│   ├── base.py                 # 기본 클래스 및 인터페이스 (Pool, SwapRequest, BaseAlgorithm)
│   ├── single/                 # 단일 경로 알고리즘
│   │   ├── naive.py            # Naive Brute Force
│   │   ├── bfs_routing.py       # BFS 기반 라우팅
│   │   ├── dijkstra.py         # Dijkstra 최단 경로
│   │   └── a_star.py            # A* 휴리스틱 탐색
│   └── composite/              # 복합 경로 알고리즘
│       ├── simple_split.py     # 균등 분할
│       ├── greedy_split.py     # 탐욕적 분할
│       ├── multi_hop.py        # 멀티홉 라우팅
│       ├── dp_routing.py       # 동적 프로그래밍 라우팅
│       └── convex_split.py     # 컨벡스 최적화 분할
├── mock_data/
│   └── data_generator.py       # Mock 풀/스왑 요청 생성기
├── benchmarks/
│   ├── benchmark_runner.py     # 벤치마크 실행 및 결과 수집
│   └── chart_generator.py     # 성능 비교 차트 생성
├── results/                    # 벤치마크 결과 (JSON)
│   ├── benchmark_results.json
│   └── summary.json
├── extreme_benchmark.py         # 벤치마크 진입점 (유일한 벤치마크 실행 스크립트)
├── comparison_chart.py         # 비교 차트 생성
├── requirements.txt            # Python 의존성
└── docs/                       # 문서
```

## 설치 방법

### 최소 설치 (차트 없이 실행)

의존성 없이 표준 라이브러리만으로 벤치마크 실행이 가능하다.

```bash
python3 extreme_benchmark.py
```

차트 생성 없이 실행하려면 matplotlib/numpy가 없어도 된다(차트만 비활성화됨).

### 전체 설치 (차트 생성 포함)

가상환경을 만들고 의존성을 설치한 뒤 실행한다.

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`에는 주로 `matplotlib`, `numpy` 등 차트 생성에 필요한 패키지가 포함된다.

## 실행 방법

벤치마크는 **extreme_benchmark.py**만 사용한다. 다른 진입점(main.py 등)은 사용하지 않는다.

| 명령 | 설명 | 소요 시간 |
|------|------|-----------|
| `python3 extreme_benchmark.py` | 극한 시나리오 벤치마크 6종 실행 및 차트 생성 | 약 2분 |
| `python3 comparison_chart.py` | 기존 결과 기반 비교 차트만 생성 (결과 JSON 필요) | 수 초 |

extreme_benchmark.py는 `create_extreme_scenarios()`로 6개 시나리오를 만들고, 시나리오마다 전용 풀로 BenchmarkRunner를 돌린 뒤 결과를 출력·저장하고, ChartGenerator가 사용 가능하면 시나리오별 출력 비교·실행 시간·히트맵 차트를 생성한다.

## 도출 결과 개요

- **results/benchmark_results.json**: extreme_benchmark의 6개 시나리오에 대해, 시나리오별로 각 알고리즘의 `amount_out`, `execution_time_ms`, `price_impact`, `improvement_vs_baseline` 등이 기록된다.
- **results/summary.json**: 위 6개 시나리오에 대한 알고리즘별 평균 출력량, 평균 실행 시간, 베이스라인 대비 평균 개선률, 승률(win rate) 등 요약 통계가 저장된다.
- **PNG 차트**: `output_comparison_*.png`(출력량 비교), `execution_time_*.png`(실행 시간), `performance_heatmap.png`(시나리오×알고리즘 성능 히트맵) 등이 프로젝트 루트 또는 지정 출력 디렉터리(예: output/extreme_charts)에 생성된다.

자세한 데이터 해석과 결론은 `docs/04-data-analysis-and-conclusions.md`를 참고한다.
