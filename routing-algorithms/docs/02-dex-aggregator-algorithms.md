# DEX Aggregator 알고리즘 분석

## 분류 체계

구현된 알고리즘은 다음 두 가지로 구분된다.

- **Single Path**: 토큰 A에서 B로 가는 **단일 최적 경로** 한 개를 선택한다.
- **Composite Path**: 여러 풀에 금액을 **분할**하거나, 중간 토큰을 거치는 **멀티홉** 경로를 고려한다.

모든 알고리즘은 `algorithms/base.py`의 `BaseAlgorithm`을 상속하며, 풀 목록으로 `token_graph`를 구성하고 `find_best_route(SwapRequest) -> SwapResult`를 구현한다. 풀의 출력량은 constant product (x*y=k) 및 수수료를 반영한 `get_amount_out`으로 계산된다.

---

## Single Path 알고리즘

### 1. Naive Brute Force

- **파일**: `algorithms/single/naive.py`
- **역할**: 탐색 가능한 경로를 열거한 뒤, 그중 출력량이 최대인 경로 하나를 선택하는 베이스라인.
- **내부 로직**:
  - DFS로 `token_in`에서 `token_out`까지 **최대 경로 길이 4**(즉 최대 3홉) 이내의 경로를 생성한다.
  - 각 (path, pools)에 대해 `calculate_route_output`로 출력량을 계산하고, 최대 출력량을 주는 경로를 반환한다.
- **특징**: 경로 길이 상한(MAX_PATH_LENGTH=4) 내에서만 최적이 보장된다.

### 2. BFS Routing

- **파일**: `algorithms/single/bfs_routing.py`
- **역할**: **홉 수가 가장 적은** 경로 한 개를 선택한다. 가격·슬리피지는 고려하지 않는다.
- **내부 로직**:
  - BFS 큐에 (현재 토큰, 경로, 풀 리스트)를 넣고, `token_out`에 **처음 도달한** 경로를 반환한다.
  - 해당 경로로 `calculate_route_output`를 호출해 출력량과 가격 충격을 계산한다.
- **특징**: O(V+E)로 빠르고 홉 수를 줄이지만, 출력량 최적화는 하지 않아 대량 거래에서 불리할 수 있다.

### 3. Dijkstra Routing

- **파일**: `algorithms/single/dijkstra.py`
- **역할**: **순간 환율(spot price) 기준으로** 비용이 최소인 단일 경로를 찾는다. 간선 가중치는 `-log(spot_price)`로 고정한다.
- **내부 로직**:
  - 가중치: `-log(pool.get_spot_price(token_in))`. 금액에 의존하지 않아 Dijkstra 최적성이 성립한다.
  - 우선순위 큐로 Dijkstra를 수행해 `token_in`에서 `token_out`까지 가중치 합이 최소인 경로를 구한 뒤, 해당 경로에 대해 `calculate_route_output`로 실제 출력량을 계산한다.
- **특징**: 대량 거래에서는 spot 최적 경로가 슬리피지로 인해 실제 출력이 Naive/MultiHop보다 낮을 수 있어, 차트에서 구분된다. 음수 가중치(차익거래)는 다루지 못한다.

### 4. A* Search Routing

- **파일**: `algorithms/single/a_star.py`
- **역할**: Dijkstra와 동일하게 **spot price**를 간선 가중치로 사용하고, 휴리스틱으로 탐색 노드 수를 줄인다.
- **내부 로직**:
  - 간선 가중치: `-log(spot_price)` (Dijkstra와 동일).
  - 초기화 시 각 풀의 spot price로 (token_a, token_b) 쌍별 best rate를 사전 계산해 휴리스틱에 사용한다.
  - f = g + h로 우선순위 큐를 사용하며, spot 기준 최적 경로를 찾은 뒤 `calculate_route_output`로 실제 출력량을 계산한다.
- **특징**: Dijkstra와 같은 “spot 최적” 경로를 찾으므로, 대량 거래 시 Naive/MultiHop보다 낮은 출력이 나올 수 있다.

### 5. K-Best Routing

- **파일**: `algorithms/single/k_best.py`
- **역할**: 단일 경로 후보를 여러 개(top-k) 뽑은 뒤, **실제 거래 크기(amount_in)** 기준 `amount_out`으로 재평가해 최종 경로를 선택한다.
- **내부 로직**:
  - DFS로 최대 홉 제한 내 단순 경로를 생성한다.
  - 각 경로를 fee 반영 spot 점수(`-log(spot_rate * (1-fee))`)로 정렬한다.
  - 상위 k개 후보만 실제 `calculate_route_output`으로 다시 계산하고, 출력량이 최대인 경로를 반환한다.
- **특징**: Dijkstra/A*의 빠른 후보 선정 장점은 유지하면서, 대량 거래에서 발생하는 spot-실거래 괴리를 줄이는 데 유리하다.

---

## Composite Path 알고리즘

### 1. Simple Split

- **파일**: `algorithms/composite/simple_split.py`
- **역할**: 동일 토큰 쌍에 대한 **모든 풀에 금액을 균등 분배**한다.
- **내부 로직**:
  - `get_pools_for_pair(token_in, token_out)`로 해당 쌍의 풀만 조회한다.
  - `amount_in / len(pools)`씩 나누어 각 풀에 스왑하고, 출력량·가격 충격을 합산해 `SwapResult`를 만든다.
- **특징**: 구현이 단순하고 큰 주문의 가격 충격 분산에는 도움이 되나, 유동성·비율 차이는 반영하지 않아 유동성이 불균형할 때 불리할 수 있다.

### 2. Greedy Split

- **파일**: `algorithms/composite/greedy_split.py`
- **역할**: **한 번에 한 청크씩**, “현재 추가했을 때 출력이 가장 많이 늘어나는 풀”에 할당한다.
- **내부 로직**:
  - 전체 금액을 `ITERATIONS`(100)개 청크로 나눈다.
  - 매 반복마다 각 풀에 “현재 할당량 + 청크”를 넣었을 때의 **한계 출력(marginal output)**을 계산한다.
  - 한계 출력이 최대인 풀에 그 청크를 할당하고, 반복한다.
  - 최종 할당량으로 각 풀의 출력·가격 충격을 구해 라우트 리스트와 총 출력량을 반환한다.
- **특징**: 유동성·비율을 반영해 Simple Split보다 나은 경우가 많으나, 전역 최적은 보장하지 않는다.

### 3. Multi-Hop Routing

- **파일**: `algorithms/composite/multi_hop.py`
- **역할**: **중간 토큰을 거치는 경로**(예: ETH -> WBTC -> USDC)까지 포함해, 출력량이 최대인 **단일 경로**를 선택한다.
- **내부 로직**:
  - DFS로 `token_in`에서 `token_out`까지 **최대 홉 수(MAX_HOPS=4)** 이내의 경로를 나열한다.
  - 각 경로에 대해 `calculate_route_output(path, pools, amount_in)`으로 출력량을 계산한다.
  - 출력량이 최대인 경로 하나를 골라 `SwapRoute` 하나로 반환한다.
  - `find_best_route_with_splits(num_splits)`: 동일하게 경로를 구한 뒤, **출력 비율 상위 N개 경로**에 금액을 균등 분할해 여러 `SwapRoute`로 실행하는 멀티홉+스플릿 버전을 제공한다.
- **특징**: 직통 풀이 없거나 비율이 나쁠 때 중간 토큰 경유가 유리할 수 있다.

### 4. DP Split Routing

- **파일**: `algorithms/composite/dp_routing.py`
- **역할**: 동일 토큰 쌍의 여러 풀에 **입력 금액을 나누어 넣는 비율**을 **동적 프로그래밍**으로 이산화하여 최적화한다.
- **내부 로직**:
  - `DISCRETIZATION_STEPS=100`으로 입력 금액을 100등분한 “단위”로 쪼갠다.
  - 상태: `dp[i][j]` = “처음 i개 풀만 사용하고, j 단위의 입력”으로 얻을 수 있는 **최대 총 출력** (및 그때의 풀별 할당).
  - i번째 풀에 k 단위(0 <= k <= j) 할당한 경우: `dp[i-1][j-k]` + (i번 풀에 k 단위 넣었을 때 출력). 이 중 최대를 `dp[i][j]`에 저장한다.
  - `dp[n_pools][n_steps]`에서 역추적해 풀별 최적 할당량을 구하고, 그에 맞춰 `SwapRoute` 리스트를 만든다.
  - numpy가 있으면 `_dp_optimal_split_optimized`로 동일 로직을 행렬 연산으로 수행할 수 있다.
- **특징**: 이산화 오차는 있으나, 주어진 스텝 수 안에서는 분할 비율이 거의 최적에 가깝고, 비선형 가격 충격도 반영한다.

### 5. Convex Optimization Split

- **파일**: `algorithms/composite/convex_split.py`
- **역할**: 풀별 할당량을 **연속 변수**로 두고, **총 출력량 최대화**를 볼록 최적화 문제로 푼다.
- **내부 로직**:
  - AMM 출력 함수는 할당량에 대해 concave이므로, 목적함수(총 출력)를 최대화하는 문제로 볼 수 있다.
  - **Projected gradient ascent**: 목적은 총 출력 최대화, 제약은 할당 합 = total_amount 및 모든 할당 >= 0(심플렉스). 수치 미분으로 gradient를 구한 뒤 step 진행하고, `_project_simplex`로 심플렉스 위로 투영한다.
  - 수렴할 때까지 반복(최대 1000회, tolerance 1e-8)한 뒤, 할당량이 유의한 풀만 골라 `SwapRoute`를 만든다.
  - 선택적으로 scipy의 `minimize`(SLSQP)로 동일 문제를 푸는 `_optimize_with_scipy`도 구현되어 있다.
- **특징**: 이론적으로 AMM에 맞는 형태이며 연속 최적화라 이산화 오차가 없으나, 수치 안정성·수렴 속도는 구현·파라미터에 의존한다.

### 6. KKT Optimal Split

- **파일**: `algorithms/composite/kkt_split.py`
- **역할**: constant-product AMM의 출력 함수 특성을 이용해, KKT 조건으로 **직접 페어 분할 비율**을 빠르게 계산한다.
- **내부 로직**:
  - 목적함수: `sum_i out_i(a_i)` 최대화, 제약: `sum_i a_i = total_amount`, `a_i >= 0`.
  - 각 풀의 한계 출력(marginal output)이 동일해지는 KKT 조건을 이용해 `a_i(lambda)` 형태로 닫힌식 계산을 구성한다.
  - 라그랑주 승수 `lambda`를 이분 탐색으로 풀어, 최종 allocation을 복원한다.
- **특징**: 수치 미분 기반 gradient 방식 대비 실행 시간이 짧고, 동일 목적함수에서 안정적으로 높은 품질의 분할을 제공한다.

---

## 공통 데이터 구조

- **입력**: `SwapRequest(token_in, token_out, amount_in)`과, `BaseAlgorithm` 생성 시 전달한 `Pool` 리스트.
- **풀**: `get_amount_out`, `get_price_impact`, `get_spot_price`로 constant product 및 수수료를 반영한다.
- **그래프**: `_build_graph()`로 `token_graph[토큰] = [(이웃 토큰, Pool), ...]` 형태로 구성된다.
- **출력**: `SwapResult(routes, total_amount_in, total_amount_out, total_price_impact, execution_time_ms, algorithm_name, gas_estimate 등)`.

Single 계열은 “어떤 단일 경로를 고를지”, Composite 계열은 “같은 쌍의 여러 풀/여러 경로에 금액을 어떻게 나눌지·멀티홉을 허용할지”에 초점을 둔다.

### 출력량이 같아지거나 달라지는 이유

- **Naive / Multi-Hop**: 제한된 경로(또는 홉 수 이내)를 열거한 뒤 **실제 출력량(amount_out)**을 경로마다 계산하고, 최대를 주는 경로를 선택한다. 따라서 “실제 출력 기준 최적”에 가깝다. Naive는 최대 경로 길이 4(3홉), Multi-Hop은 최대 4홉까지 탐색하므로, 최적 경로가 같은 범위 안에 있으면 둘의 결과가 일치할 수 있다.
- **Dijkstra / A\***: 간선 가중치를 **spot price(순간 환율)**만 사용한다. 따라서 “순간 환율 기준 최적” 경로를 찾고, 그 경로에 대해 `calculate_route_output`으로 실제 출력을 계산한다. 대량 거래에서는 슬리피지 때문에 이 경로의 실제 출력이 Naive/Multi-Hop보다 **낮을 수 있어**, 차트에서 구분된다.
- **K-Best**: spot 점수로 후보를 줄인 다음 실제 출력으로 재평가하기 때문에, Dijkstra/A*보다 대량 거래에서 더 안정적인 경로를 선택할 가능성이 높다.
- **BFS**: 홉 수만 최소화하고 가격을 보지 않으므로, 최적이 아닌 경로를 택할 수 있어 출력이 더 낮게 나올 수 있다.
- **KKT Split / Convex Split**: 둘 다 연속 분할 최적화 관점에서 높은 출력량을 목표로 하지만, KKT Split은 CPMM 가정에서 닫힌식+이분 탐색을 써서 계산 효율이 높다.
