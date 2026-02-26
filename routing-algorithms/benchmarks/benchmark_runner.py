"""
Benchmark runner for DEX routing algorithms.

This module provides a light-weight benchmark harness used by `main.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from algorithms.base import BaseAlgorithm, SwapRequest
from algorithms.composite import ConvexSplit, DPRouting, GreedySplit, MultiHopRouting, SimpleSplit
from algorithms.single import (
    AStarRouting,
    BFSRouting,
    BellmanFordRouting,
    DijkstraRouting,
    NaiveBruteForce,
)


@dataclass
class BenchmarkResult:
    """Single algorithm result for one swap request."""

    algorithm_name: str
    difficulty: str
    algorithm_type: str
    amount_in: float
    amount_out: float
    effective_price: float
    price_impact: float
    execution_time_ms: float
    gas_estimate: int
    num_routes: int
    improvement_vs_baseline: float = 0.0
    error: Optional[str] = None


@dataclass
class ScenarioBenchmark:
    """Benchmark results for a named scenario."""

    scenario_name: str
    swap_request: SwapRequest
    results: List[BenchmarkResult]
    baseline_output: float


class BenchmarkRunner:
    """
    Executes benchmark scenarios across all configured routing algorithms.
    """

    def __init__(self, pools):
        self.pools = pools
        # Single-path + composite lineup for baseline comparisons.
        self.algorithms: List[BaseAlgorithm] = [
            NaiveBruteForce(pools),
            BFSRouting(pools),
            DijkstraRouting(pools),
            BellmanFordRouting(pools),
            AStarRouting(pools),
            SimpleSplit(pools),
            GreedySplit(pools),
            MultiHopRouting(pools),
            DPRouting(pools),
            ConvexSplit(pools),
        ]

    def run_single_benchmark(self, algorithm: BaseAlgorithm, request: SwapRequest) -> BenchmarkResult:
        """
        Execute one algorithm for one request.
        """
        meta = algorithm.metadata
        try:
            result = algorithm.execute_with_timing(request)
            return BenchmarkResult(
                algorithm_name=meta.name,
                difficulty=meta.difficulty.value,
                algorithm_type=meta.algorithm_type.value,
                amount_in=result.total_amount_in,
                amount_out=result.total_amount_out,
                effective_price=result.effective_price,
                price_impact=result.total_price_impact,
                execution_time_ms=result.execution_time_ms,
                gas_estimate=result.gas_estimate,
                num_routes=len(result.routes),
            )
        except Exception as exc:  # pragma: no cover - protective fallback
            return BenchmarkResult(
                algorithm_name=meta.name,
                difficulty=meta.difficulty.value,
                algorithm_type=meta.algorithm_type.value,
                amount_in=request.amount_in,
                amount_out=0.0,
                effective_price=0.0,
                price_impact=1.0,
                execution_time_ms=0.0,
                gas_estimate=0,
                num_routes=0,
                error=str(exc),
            )

    def run_scenario_benchmark(self, scenario_name: str, request: SwapRequest) -> ScenarioBenchmark:
        """
        Benchmark all algorithms against a single scenario.
        """
        baseline_algo = self._find_baseline_algorithm()
        baseline_result = self.run_single_benchmark(baseline_algo, request)
        baseline_output = baseline_result.amount_out

        results: List[BenchmarkResult] = []
        for algorithm in self.algorithms:
            if algorithm is baseline_algo:
                result = baseline_result
            else:
                result = self.run_single_benchmark(algorithm, request)

            if baseline_output > 0:
                result.improvement_vs_baseline = (
                    (result.amount_out - baseline_output) / baseline_output
                ) * 100.0
            else:
                result.improvement_vs_baseline = 0.0

            results.append(result)

        return ScenarioBenchmark(
            scenario_name=scenario_name,
            swap_request=request,
            results=results,
            baseline_output=baseline_output,
        )

    def run_all_benchmarks(
        self, scenarios: Sequence[Tuple[str, SwapRequest]]
    ) -> List[ScenarioBenchmark]:
        """
        Run all scenarios and return all results.
        """
        all_results: List[ScenarioBenchmark] = []
        for scenario_name, request in scenarios:
            all_results.append(self.run_scenario_benchmark(scenario_name, request))
        return all_results

    def run_scaling_benchmark(
        self, request: SwapRequest, amounts: Iterable[float]
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Sweep input amounts and collect per-algorithm benchmark results.
        """
        collected: Dict[str, List[BenchmarkResult]] = {
            algo.metadata.name: [] for algo in self.algorithms
        }

        for amount in amounts:
            scaled_request = SwapRequest(
                token_in=request.token_in,
                token_out=request.token_out,
                amount_in=float(amount),
            )
            baseline_output = self.run_single_benchmark(self._find_baseline_algorithm(), scaled_request).amount_out

            for algorithm in self.algorithms:
                result = self.run_single_benchmark(algorithm, scaled_request)
                if baseline_output > 0:
                    result.improvement_vs_baseline = (
                        (result.amount_out - baseline_output) / baseline_output
                    ) * 100.0
                collected[result.algorithm_name].append(result)

        return collected

    def get_summary_statistics(
        self, scenarios: Sequence[ScenarioBenchmark]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate scenario results into per-algorithm summary statistics.
        """
        if not scenarios:
            return {}

        summary: Dict[str, Dict[str, float]] = {}
        wins: Dict[str, int] = {}
        counts: Dict[str, int] = {}

        for scenario in scenarios:
            if not scenario.results:
                continue
            winner = max(scenario.results, key=lambda r: r.amount_out).algorithm_name
            wins[winner] = wins.get(winner, 0) + 1

            for result in scenario.results:
                name = result.algorithm_name
                if name not in summary:
                    summary[name] = {
                        "difficulty": result.difficulty,
                        "type": result.algorithm_type,
                        "total_output": 0.0,
                        "total_time_ms": 0.0,
                        "total_improvement_pct": 0.0,
                    }
                    counts[name] = 0

                summary[name]["total_output"] += result.amount_out
                summary[name]["total_time_ms"] += result.execution_time_ms
                summary[name]["total_improvement_pct"] += result.improvement_vs_baseline
                counts[name] += 1

        num_scenarios = float(len(scenarios))
        finalized: Dict[str, Dict[str, float]] = {}
        for name, acc in summary.items():
            n = max(1, counts[name])
            finalized[name] = {
                "difficulty": acc["difficulty"],
                "type": acc["type"],
                "avg_output": acc["total_output"] / n,
                "avg_time_ms": acc["total_time_ms"] / n,
                "avg_improvement_pct": acc["total_improvement_pct"] / n,
                "win_rate_pct": (wins.get(name, 0) / num_scenarios) * 100.0,
            }
        return finalized

    def print_results(self, scenario: ScenarioBenchmark) -> None:
        """
        Pretty-print one scenario benchmark table.
        """
        print(f"\nScenario: {scenario.scenario_name}")
        print(
            f"Swap: {scenario.swap_request.amount_in:g} "
            f"{scenario.swap_request.token_in} -> {scenario.swap_request.token_out}"
        )
        print(
            f"{'Algorithm':<30} {'Output':>14} {'Imp(%)':>10} {'Time(ms)':>10} {'Routes':>8}"
        )
        print("-" * 78)

        sorted_results = sorted(scenario.results, key=lambda r: r.amount_out, reverse=True)
        for result in sorted_results:
            print(
                f"{result.algorithm_name:<30} "
                f"{result.amount_out:>14.2f} "
                f"{result.improvement_vs_baseline:>10.2f} "
                f"{result.execution_time_ms:>10.3f} "
                f"{result.num_routes:>8}"
            )
            if result.error:
                print(f"  ! {result.algorithm_name} failed: {result.error}")

    def _find_baseline_algorithm(self) -> BaseAlgorithm:
        """
        Baseline is Naive Brute Force if available, else first algorithm.
        """
        for algorithm in self.algorithms:
            if algorithm.metadata.name == "Naive Brute Force":
                return algorithm
        return self.algorithms[0]
