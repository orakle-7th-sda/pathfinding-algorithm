"""
K-Best Candidate Routing.

Generates multiple spot-price candidate paths, then re-ranks by actual
amount_out for the requested trade size.
"""

from typing import List, Optional, Tuple
import math
import time

from ..base import (
    BaseAlgorithm,
    AlgorithmMetadata,
    Difficulty,
    AlgorithmType,
    Pool,
    SwapRequest,
    SwapResult,
    SwapRoute,
)


class KBestRouting(BaseAlgorithm):
    """
    K-Best candidate routing algorithm.

    1) Enumerate simple paths up to max hops.
    2) Rank by spot-based score.
    3) Evaluate top-k candidates with full swap simulation.
    4) Return the best by actual output.
    """

    DEFAULT_K_PATHS = 8
    MAX_HOPS = 4

    def __init__(self, pools: List[Pool], k_paths: int = DEFAULT_K_PATHS):
        super().__init__(pools)
        self.k_paths = max(1, k_paths)

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="K-Best Routing",
            difficulty=Difficulty.MEDIUM,
            algorithm_type=AlgorithmType.SINGLE,
            time_complexity="O(paths + k * hops)",
            space_complexity="O(paths)",
            description=(
                "Enumerates candidate paths, ranks by spot score, then "
                "re-evaluates top-k by actual output."
            ),
            pros=[
                "Bridges gap between spot routing and real output",
                "More robust than pure spot shortest-path",
                "Configurable k for speed/quality tradeoff",
            ],
            cons=[
                "Path enumeration can grow quickly",
                "Still single-route only (no splitting)",
                "Quality depends on max hops and k",
            ],
        )

    def find_best_route(self, request: SwapRequest) -> SwapResult:
        start_time = time.perf_counter()

        if request.token_in == request.token_out:
            route = SwapRoute(
                path=[request.token_in],
                pools=[],
                dexes=[],
                amount_in=request.amount_in,
                amount_out=request.amount_in,
            )
            end_time = time.perf_counter()
            return SwapResult(
                routes=[route],
                total_amount_in=request.amount_in,
                total_amount_out=request.amount_in,
                total_price_impact=0.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name,
                gas_estimate=0,
            )

        all_candidates = self._enumerate_paths(
            request.token_in, request.token_out, self.MAX_HOPS
        )

        if not all_candidates:
            end_time = time.perf_counter()
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name,
            )

        ranked_candidates = sorted(
            all_candidates,
            key=lambda item: self._spot_path_score(item[0], item[1]),
        )
        top_k = ranked_candidates[: self.k_paths]

        best_route: Optional[SwapRoute] = None
        best_output = 0.0

        for path, pools in top_k:
            output = self.calculate_route_output(path, pools, request.amount_in)
            if output > best_output:
                best_output = output
                best_route = SwapRoute(
                    path=path,
                    pools=pools,
                    dexes=[p.dex for p in pools],
                    amount_in=request.amount_in,
                    amount_out=output,
                )

        end_time = time.perf_counter()

        if best_route is None:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name,
            )

        base_gas = 150000
        gas_per_hop = 80000
        gas_estimate = base_gas + max(0, len(best_route.pools) - 1) * gas_per_hop

        return SwapResult(
            routes=[best_route],
            total_amount_in=request.amount_in,
            total_amount_out=best_output,
            total_price_impact=best_route.price_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name,
            gas_estimate=gas_estimate,
        )

    def _enumerate_paths(
        self, start: str, end: str, max_hops: int
    ) -> List[Tuple[List[str], List[Pool]]]:
        results: List[Tuple[List[str], List[Pool]]] = []

        def dfs(
            current: str, path: List[str], pools: List[Pool], visited_tokens: set
        ) -> None:
            if len(pools) > max_hops:
                return
            if current == end and pools:
                results.append((path.copy(), pools.copy()))
                return
            if current not in self.token_graph:
                return
            for next_token, pool in self.token_graph[current]:
                if next_token in visited_tokens:
                    continue
                visited_tokens.add(next_token)
                path.append(next_token)
                pools.append(pool)
                dfs(next_token, path, pools, visited_tokens)
                pools.pop()
                path.pop()
                visited_tokens.remove(next_token)

        dfs(start, [start], [], {start})
        return results

    def _spot_path_score(self, path: List[str], pools: List[Pool]) -> float:
        """
        Lower score is better.
        Uses negative log of fee-adjusted spot rates.
        """
        score = 0.0
        for i, pool in enumerate(pools):
            token_in = path[i]
            rate = pool.get_spot_price(token_in) * (1.0 - pool.fee)
            if rate <= 0:
                return float("inf")
            score += -math.log(rate)
        return score
