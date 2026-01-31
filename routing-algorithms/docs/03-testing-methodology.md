# 테스트 방법

## Mock 데이터 개요

테스트는 실제 체인 대신 **mock 데이터**로 수행된다. Mock 데이터는 `mock_data/data_generator.py`의 `DataGenerator`가 생성하며, Uniswap V2 스타일의 constant product AMM 풀과 스왑 요청을 시뮬레이션한다.

## Mock 데이터 종류

### 1. 토큰 및 가격

- **TOKEN_PRICES**: ETH, WBTC, USDC, USDT, DAI, WETH, LINK, UNI, AAVE, SUSHI, CRV, MKR, COMP, SNX, YFI 등 토큰의 대략적인 USD 가격이 하드코딩되어 있다.
- 풀의 reserve는 “USD 기준 유동성”을 각 토큰 가격으로 나누어 token0/token1 수량으로 환산한다.

### 2. DEX 설정 (DEX_CONFIGS)

| DEX | 수수료 (fee) | 유동성 배수 (liquidity_multiplier) |
|-----|--------------|------------------------------------|
| uniswap_v2 | 0.003 | 1.0 |
| uniswap_v3 | 0.003 | 1.2 |
| sushiswap | 0.003 | 0.7 |
| curve | 0.0004 | 1.5 |
| balancer | 0.002 | 0.5 |
| pancakeswap | 0.0025 | 0.4 |

- 각 풀 생성 시 `base_liquidity_usd * liquidity_multiplier * random_factor(0.5~1.5)`로 총 유동성을 정하고, token0/token1 reserve를 그에 맞춰 계산한다.

### 3. 풀 생성 메서드

- **generate_pool(token0, token1, dex, base_liquidity_usd)**: 한 토큰 쌍·한 DEX에 대한 풀 하나를 생성한다. constant product 풀이며 fee는 DEX 설정을 따른다.
- **generate_pools_for_pair(token0, token1, num_pools, base_liquidity_usd)**: 같은 토큰 쌍에 대해 여러 DEX에 풀을 num_pools개까지 생성한다.
- **generate_token_network(num_tokens, pools_per_pair, connection_probability, base_liquidity_usd)**: num_tokens개 토큰을 랜덤 선택하고, 토큰 쌍마다 connection_probability 확률로 풀을 만들며, 쌍당 pools_per_pair개 풀을 생성해 전체 토큰 그래프를 만든다.
- **generate_standard_test_pools()**: 고정된 “주요 쌍 + 중간 유동성 쌍 + 브릿지/스테이블 쌍”으로 표준 테스트용 풀 집합을 반환한다.  
  - 주요 쌍: ETH-USDC, ETH-USDT, WBTC-USDC, WBTC-ETH, ETH-DAI (고유동성).  
  - 중간 쌍: LINK-ETH, UNI-ETH, AAVE-ETH, LINK-USDC, UNI-USDC.  
  - 브릿지/스테이블: WETH-ETH, USDC-USDT, USDC-DAI.

### 4. 스왑 요청

- **generate_swap_request(token_in, token_out, amount_in)**: `SwapRequest(token_in, token_out, amount_in)`을 그대로 반환한다. 기본값은 ETH -> USDC, amount_in=10.0이다.

### 5. 벤치마크 시나리오 (유일한 시나리오 소스: extreme_benchmark.py)

벤치마크에 사용되는 시나리오는 **extreme_benchmark.py**의 `create_extreme_scenarios()`만 사용한다. `generate_test_scenarios()`는 제거되었으며, 표준 테스트 시나리오는 사용하지 않는다. `create_extreme_scenarios()`가 만드는 시나리오는 알고리즘 간 차이를 극대화하기 위한 것이다.

| 시나리오 이름 | 설명 | 풀 구성 | 스왑 요청 |
|---------------|------|---------|-----------|
| Massive Swap (1000 ETH) | 대량 거래 | generate_standard_test_pools() | 1000 ETH -> USDC |
| Extreme Swap (5000 ETH) | 극대량 거래 | generate_standard_test_pools() | 5000 ETH -> USDC |
| Low Liquidity Crisis (100 ETH) | 저유동성 + 대량 | ETH-USDC/USDT/DAI × uniswap_v2/sushiswap, 풀당 10만 USD | 100 ETH -> USDC |
| Imbalanced Liquidity (500 ETH) | 한 DEX만 고유동성 | Uniswap 5천만, Sushiswap 50만, Curve 20만 USD | 500 ETH -> USDC |
| Many DEXes Split Test (200 ETH) | DEX 다수, 개별 유동성 낮음 | 5개 DEX 각 50만 USD | 200 ETH -> USDC |
| Single DEX Only (500 ETH) | 풀 1개만 존재 | ETH-USDC 풀 1개 (500만 USD) | 500 ETH -> USDC |

## Aggregator 테스트 진행 방식

### 1. 벤치마크 러너 (benchmarks/benchmark_runner.py)

- **BenchmarkRunner(pools)**: 풀 리스트로 초기화하며, Single Path 4종 + Composite Path 5종 총 9개 알고리즘 인스턴스를 생성한다.
- **run_single_benchmark(algorithm, request)**: 한 알고리즘에 대해 `execute_with_timing(request)`를 호출하고, `BenchmarkResult`(algorithm_name, amount_out, effective_price, price_impact, execution_time_ms, gas_estimate, num_routes, improvement_vs_baseline)를 반환한다.
- **run_scenario_benchmark(scenario_name, request)**:  
  - 먼저 Naive Brute Force를 베이스라인으로 실행해 baseline_output을 구한다.  
  - 모든 알고리즘에 대해 run_single_benchmark를 실행하고, 각 결과의 `improvement_vs_baseline`을 `(amount_out - baseline_output) / baseline_output * 100`으로 계산한다.  
  - 시나리오별로 `ScenarioBenchmark(scenario_name, swap_request, results, baseline_output)`를 반환한다.
- **run_all_benchmarks(scenarios)**: (scenario_name, SwapRequest) 리스트를 받아 시나리오마다 run_scenario_benchmark를 호출하고, `ScenarioBenchmark` 리스트를 반환한다.
- **run_scaling_benchmark(request, amounts)**: 동일 request에서 amount_in만 amounts 리스트로 바꿔 가며 각 알고리즘을 실행해, 알고리즘별로 결과 리스트를 모은다.
- **get_summary_statistics(scenarios)**: 시나리오 리스트에서 알고리즘별 총 출력량·총 실행 시간·총 개선률·승률(해당 시나리오에서 최고 출력인 횟수 비율) 등을 집계해 요약 딕셔너리를 반환한다.

### 2. 벤치마크 실행 흐름 (extreme_benchmark.py만 사용)

벤치마크는 **extreme_benchmark.py**만 실행한다. main.py의 표준 시나리오(generate_test_scenarios)는 사용하지 않는다.

- `create_extreme_scenarios()`로 위 “벤치마크 시나리오” 6개 리스트 (name, request, pools)를 만든다.
- 시나리오마다 **해당 시나리오 전용 풀**로 `BenchmarkRunner(pools)`를 새로 만들고, `run_scenario_benchmark(name, request)`만 실행한다. 즉, 시나리오별로 풀 구성이 다르다.
- 각 시나리오 결과에 대해 `runner.print_results(result)`로 출력하고, best/worst 알고리즘 차이(Gap Analysis)를 계산해 출력한다.
- 결과를 `results/benchmark_results.json` 등에 저장하고, `ChartGenerator`가 사용 가능하면 시나리오별 output comparison, improvement, execution time 차트와, 전체 시나리오×알고리즘 히트맵을 생성한다.

### 3. 재현성

- `DataGenerator(seed=42)`로 고정 시드를 사용하면 풀 생성이 재현 가능하다.
- 벤치마크는 동일 풀·동일 SwapRequest에 대해 각 알고리즘을 순차 실행하므로, 입력이 같으면 결과 비교가 공정하다.

## 요약

- Mock 데이터: `DataGenerator`가 토큰 가격·DEX 설정·유동성 배수를 사용해 constant product 풀과 스왑 요청을 생성한다.
- 벤치마크 시나리오: **extreme_benchmark.py**의 `create_extreme_scenarios()`만 사용한다. Massive Swap(1000 ETH), Extreme Swap(5000 ETH), Low Liquidity Crisis, Imbalanced Liquidity, Many DEXes Split Test, Single DEX Only 등 6가지 시나리오가 정의되어 있다. `generate_test_scenarios()`는 제거되어 사용하지 않는다.
- Aggregator 테스트: extreme_benchmark 실행 시, 시나리오마다 해당 풀로 9개 알고리즘을 모두 실행하고, 출력량·실행 시간·베이스라인 대비 개선률·승률을 집계하며, 결과는 JSON과 PNG 차트로 저장된다.
