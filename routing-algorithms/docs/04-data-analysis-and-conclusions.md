# 도출된 데이터 분석 및 결론

## 결과 파일 구성

### results 폴더

- **benchmark_results.json**: 시나리오별로 각 알고리즘의 상세 결과 배열이 저장된다. 각 항목에는 scenario name, swap request(token_in, token_out, amount_in), 그리고 results 배열(algorithm, difficulty, type, amount_in, amount_out, effective_price, price_impact, execution_time_ms, gas_estimate, num_routes, improvement_vs_baseline)이 포함된다.
- **summary.json**: 알고리즘 이름을 키로 하여 difficulty, type, avg_output, avg_time_ms, avg_improvement_pct, win_rate_pct가 저장된다. extreme_benchmark의 6개 시나리오에 대한 집계이다.

### PNG 차트 파일

프로젝트 루트(또는 ChartGenerator 출력 디렉터리)에 다음 차트가 생성된다.

| 파일명 패턴 | 내용 |
|-------------|------|
| output_comparison_*.png | 시나리오별로 알고리즘 간 출력량(amount_out) 비교. 시나리오 이름이 파일명에 포함됨(예: Extreme Swap (5000 ETH), Low Liquidity Crisis (100 ETH)). |
| execution_time_*.png | 시나리오별로 알고리즘 간 실행 시간(ms) 비교. |
| performance_heatmap.png | 시나리오(행) x 알고리즘(열) 히트맵. 색상으로 성능(출력량 또는 개선률)을 표현한다. |

극한 벤치마크를 실행하면 `output/extreme_charts/` 등에 동일한 종류의 차트가 시나리오별로 저장될 수 있다.

---

## results/summary.json 분석

extreme_benchmark(6개 시나리오) 요약에서 읽을 수 있는 내용은 다음과 같다.

- **평균 출력량(avg_output)**: Convex Optimization Split, Greedy Split, DP Split Routing이 약 200,400~200,492로 가장 높고, BFS Routing이 약 185,387로 가장 낮다. Naive/Dijkstra는 약 190,820, Simple Split은 약 198,384, Multi-Hop은 약 190,820 수준이다.
- **평균 실행 시간(avg_time_ms)**: Simple Split과 BFS가 약 0.006~0.010 ms로 가장 빠르고, DP Split이 약 3.3 ms, Convex Optimization이 약 4.6 ms로 가장 느리다. Greedy Split은 약 0.09 ms로 중간이다.
- **베이스라인 대비 평균 개선률(avg_improvement_pct)**: 베이스라인은 Naive Brute Force이다. Composite 알고리즘은 “베이스라인보다 출력이 높은 시나리오”에서 양수 개선률을 보이지만, 요약에서는 시나리오별로 베이스라인 출력이 다르기 때문에 평균 개선률이 음수로 나올 수 있다(예: Simple Split -15.48%, Greedy -15.19%, Convex -15.39%). BFS는 -0.92%로 베이스라인보다 약간 낮다.
- **승률(win_rate_pct)**: 해당 시나리오에서 출력량 1위인 비율. Convex Optimization이 66.7%, Naive Brute Force와 Greedy Split이 각 16.7% 등으로 집계된다. extreme_benchmark 6개 시나리오에서는 Convex가 가장 많이 1위를 차지한다.

---

## Output Comparison 차트 분석

### 1. Extreme Swap (5000 ETH)

- **파일**: output_comparison_Extreme_Swap_(5000_ETH).png
- **내용**: Convex Optimization Split이 약 6,186,785 USDC로 최고, Greedy/DP가 약 6,186,748 USDC로 그 다음이며, Convex와의 차이는 약 0.0006% 수준이다. Simple Split은 약 5,905,387 USDC(약 -4.55%). Naive/Dijkstra 등 단일 경로는 약 4,385,766 USDC(약 -29.1%), BFS Routing은 약 3,715,646 USDC(약 -39.95%)로 최하위이다.
- **결론**: 대량 거래에서는 분할 알고리즘(특히 Greedy 이상)이 필수에 가깝고, BFS는 가격을 고려하지 않아 손실이 크다. Convex와 Greedy의 출력 차이는 실용적으로 무시 가능한 수준이다.

### 2. Low Liquidity Crisis (100 ETH)

- **파일**: output_comparison_Low_Liquidity_Crisis_(100_ETH).png
- **내용**: Convex/Greedy/DP가 약 59,695 USDC(베이스라인 대비 약 +53.7%), Simple Split 약 59,476 USDC(+53.2%), Naive/Dijkstra 약 38,837 USDC(베이스라인), BFS 약 31,189 USDC(약 -19.7%)이다.
- **결론**: 유동성이 낮을 때 분할 알고리즘의 이득이 크고, BFS는 단일 경로보다도 나쁜 결과를 낼 수 있다. 어떤 분할이든 사용하는 것이 유리하다.

### 3. Imbalanced Liquidity (500 ETH)

- **파일**: output_comparison_Imbalanced_Liquidity_(500_ETH).png
- **내용**: Convex 약 948,695 USDC, Greedy/DP 약 948,662 USDC, Naive 등 약 947,604 USDC(베이스라인), Simple Split 약 591,654 USDC(약 -37.6%)이다.
- **결론**: 유동성이 한쪽 DEX에 치우쳐 있을 때 Simple Split(균등 분할)은 작은 풀에 과도하게 배정되어 슬리피지가 커지고 손실이 크다. Greedy/DP/Convex 등 유동성을 고려한 분할이 필요하다.

### 4. Many DEXes Split Test (200 ETH)

- **파일**: output_comparison_Many_DEXes_Split_Test_(200_ETH).png
- **내용**: Convex 약 293,465 USDC(베이스라인 대비 약 +56.8%), Greedy/DP 약 293,431 USDC, Simple 약 275,184 USDC(+47%), Naive 등 약 187,139 USDC, BFS 약 172,257 USDC(약 -7.9%)이다.
- **결론**: DEX가 많고 개별 유동성이 작을 때 분할 효과가 크다. 3개 이상 DEX가 있으면 분할 알고리즘 사용이 유리하다.

### 5. Massive Swap (1000 ETH)

- **파일**: output_comparison_Massive_Swap_(1000_ETH).png
- **내용**: Convex 약 1,751,201 USDC(베이스라인 대비 약 +18.6%), Greedy/DP 약 1,751,189 USDC, Simple 약 1,748,092 USDC(+18.3%), Naive 등 약 1,477,036 USDC이다.
- **결론**: 1000 ETH 규모에서도 분할 알고리즘이 약 18% 수준 이득을 주며, Convex와 Greedy의 차이는 매우 작다.

### 6. Single DEX Only (500 ETH)

- **파일**: output_comparison_Single_DEX_Only_(500_ETH).png
- **내용**: 풀이 하나뿐이므로 모든 알고리즘이 동일한 출력(약 564,744 USDC)을 낸다.
- **결론**: DEX가 1개일 때는 라우팅/분할 알고리즘 선택이 출력에 영향을 주지 않는다. 이 경우에는 실행 속도나 구현 단순성만 고려하면 된다.

---

## Execution Time 차트 분석

- **Extreme Swap (5000 ETH)**: Simple Split이 약 0.024 ms로 가장 빠르고, Convex가 약 11.27 ms로 가장 느리다. Greedy는 약 0.4 ms, DP는 약 5.9 ms 수준이다.
- **Low Liquidity Crisis (100 ETH)**: 풀 수가 적어 모든 알고리즘이 0.01~2.6 ms 범위로 빠르다.
- **Many DEXes Split Test (200 ETH)**: DEX 수가 많을 때 DP가 약 7.3 ms로 가장 느려지고, Convex는 약 1.9 ms, Greedy는 약 0.17 ms로 실용적이다.

종합하면, 실행 속도 순으로는 Simple Split, BFS, Dijkstra 등이 빠르고, Greedy Split이 속도와 출력의 균형이 좋으며, DP와 Convex는 가장 느리지만 출력량은 최상위이다.

---

## performance_heatmap.png

- **내용**: 행은 시나리오(예: Extreme Swap, Low Liquidity Crisis, Imbalanced Liquidity 등), 열은 알고리즘이다. 셀 색상은 해당 시나리오에서의 성능(출력량 또는 베이스라인 대비 개선률)을 나타낸다.
- **해석**: 같은 알고리즘이라도 시나리오에 따라 성능이 달라진다. Convex/Greedy/DP는 대부분 시나리오에서 진한 색(높은 성능)으로 나타나고, BFS는 대량·저유동성 시나리오에서 밝은 색(낮은 성능)으로 나타난다. Imbalanced Liquidity 시나리오에서는 Simple Split만 낮은 성능을 보이는 것을 확인할 수 있다.

---

## 결론 및 권장 사항

### 알고리즘별 특성

- **BFS Routing**: 홉 수만 최소화하고 가격을 고려하지 않아, 대량 거래·저유동성에서 출력이 크게 떨어진다. 소액·단일 DEX에서는 속도만 필요할 때 사용할 수 있다.
- **Simple Split**: 구현이 단순하고 속도가 매우 빠르나, 유동성이 불균형할 때 균등 분할로 인한 슬리피지가 커져 손실이 클 수 있다. 유동성이 비슷한 경우에만 사용하는 것이 안전하다.
- **Greedy Split**: 출력량은 Convex/DP와 거의 동일한 수준이면서 실행 시간은 훨씬 짧다. 다양한 시나리오에서 균형이 좋아 실전에서 우선 고려할 만하다.
- **DP Split Routing**: 이산화 기반으로 거의 최적에 가까운 분할을 주나, 풀 수와 스텝 수에 따라 실행 시간이 길어진다.
- **Convex Optimization Split**: 이론적으로 연속 최적화로 최고 출력에 가깝고, 벤치마크에서도 출력 1위 비율이 높다. 대신 실행 시간이 길어 실시간성이 중요하면 Greedy를 대안으로 할 수 있다.

### 시나리오별 권장

- **소액 거래(예: 10 ETH 미만)**: 출력 차이가 작으므로 BFS 또는 Dijkstra로 속도 우선 선택 가능.
- **중간~대량 거래(10~1000 ETH)**: Greedy Split을 기본으로 하고, 최대 출력이 더 중요하면 Convex 또는 DP를 고려.
- **초대량 거래(1000 ETH 이상)**: 반드시 분할 알고리즘(Greedy 이상) 사용. BFS·단일 경로만 사용하는 것은 피하는 것이 좋다.
- **저유동성**: 분할 알고리즘 사용이 크게 유리하다. BFS는 피하는 것이 좋다.
- **유동성 불균형**: Simple Split은 사용하지 않고, Greedy/DP/Convex 중 하나를 사용한다.
- **DEX 1개만 존재**: 모든 알고리즘이 동일 출력이므로, 가장 빠른 알고리즘(BFS 또는 Simple Split)만 선택하면 된다.

### 정리

- 거래 규모가 클수록 알고리즘 선택에 따른 출력 차이가 커진다(예: 5000 ETH에서 최고·최저 알고리즘 간 약 66% 차이).
- 유동성이 부족하거나 불균형할 때는 “분할 여부”와 “유동성을 고려한 분할 여부”가 결과에 큰 영향을 준다.
- 실무에서는 출력과 속도의 균형을 위해 Greedy Split을 기본으로 두고, 최대 출력이 더 중요할 때 Convex 또는 DP를 선택하는 방식이 합리적이다.
- PNG 차트와 results 폴더의 JSON은 위 결론을 수치·시각적으로 뒷받침하는 자료로 활용할 수 있다.
