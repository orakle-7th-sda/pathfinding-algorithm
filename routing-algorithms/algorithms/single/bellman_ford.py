"""
Bellman-Ford Algorithm for DEX Routing.

Handles spot-weight shortest path even when negative edge weights exist.
"""

from typing import Dict, List, Optional, Tuple
import math
import time

from ..base import (
    AlgorithmMetadata,
    AlgorithmType,
    BaseAlgorithm,
    Difficulty,
    Pool,
    SwapRequest,
    SwapResult,
    SwapRoute,
)


class BellmanFordRouting(BaseAlgorithm):
    """
    Bellman-Ford-based routing algorithm.

    Uses negative log(spot_rate) edge weights and relaxes all edges up to |V|-1
    iterations, so it remains valid when some edges are negative.

    Difficulty: Hard
    Time Complexity: O(VE)
    Space Complexity: O(V)
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Bellman-Ford Routing",
            difficulty=Difficulty.HARD,
            algorithm_type=AlgorithmType.SINGLE,
            time_complexity="O(VE)",
            space_complexity="O(V)",
            description=(
                "Relaxes all graph edges to find a shortest path under spot "
                "negative-log weights, including negative edge cases where "
                "Dijkstra assumptions may not hold."
            ),
            pros=[
                "Works with negative edge weights",
                "Conceptually simple relaxation process",
                "Robust fallback when Dijkstra preconditions fail",
            ],
            cons=[
                "Slower than Dijkstra/A* on large graphs",
                "Still spot-rate-based (large trade slippage ignored)",
                "Negative cycles can make the objective ill-defined",
            ],
        )

    def find_best_route(self, request: SwapRequest) -> SwapResult:
        start_time = time.perf_counter()
        path, pools, negative_cycle_detected = self._bellman_ford_find_path(
            request.token_in, request.token_out
        )
        end_time = time.perf_counter()

        if path is None:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name,
            )

        output = self.calculate_route_output(path, pools, request.amount_in)
        route = SwapRoute(
            path=path,
            pools=pools,
            dexes=[p.dex for p in pools],
            amount_in=request.amount_in,
            amount_out=output,
        )

        # Slightly higher estimate due to extra iterations and bookkeeping.
        gas_estimate = 175000 if negative_cycle_detected else 160000

        return SwapResult(
            routes=[route],
            total_amount_in=request.amount_in,
            total_amount_out=output,
            total_price_impact=route.price_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name,
            gas_estimate=gas_estimate,
        )

    def _get_edge_weight_spot(self, pool: Pool, token_in: str) -> float:
        rate = pool.get_spot_price(token_in)
        if rate <= 0:
            return float("inf")
        return -math.log(rate)

    def _bellman_ford_find_path(
        self, start: str, end: str
    ) -> Tuple[Optional[List[str]], Optional[List[Pool]], bool]:
        if start == end:
            return [start], [], False
        if start not in self.token_graph:
            return None, None, False

        vertices = sorted(self.token_graph.keys())
        distances: Dict[str, float] = {token: float("inf") for token in vertices}
        predecessor: Dict[str, Tuple[Optional[str], Optional[Pool]]] = {
            token: (None, None) for token in vertices
        }
        distances[start] = 0.0

        # Relax edges |V|-1 times.
        for _ in range(max(0, len(vertices) - 1)):
            updated = False
            for token in vertices:
                base = distances[token]
                if not math.isfinite(base):
                    continue
                for next_token, pool in self.token_graph.get(token, []):
                    weight = self._get_edge_weight_spot(pool, token)
                    if not math.isfinite(weight):
                        continue
                    candidate = base + weight
                    if candidate < distances[next_token]:
                        distances[next_token] = candidate
                        predecessor[next_token] = (token, pool)
                        updated = True
            if not updated:
                break

        # Detect reachable negative cycle (informational; path can still be usable).
        negative_cycle_detected = False
        for token in vertices:
            base = distances[token]
            if not math.isfinite(base):
                continue
            for next_token, pool in self.token_graph.get(token, []):
                weight = self._get_edge_weight_spot(pool, token)
                if not math.isfinite(weight):
                    continue
                if base + weight < distances[next_token]:
                    negative_cycle_detected = True
                    break
            if negative_cycle_detected:
                break

        if not math.isfinite(distances.get(end, float("inf"))):
            return None, None, negative_cycle_detected

        # Reconstruct path from predecessor map.
        path: List[str] = []
        pools: List[Pool] = []
        current: Optional[str] = end
        hop_guard = 0
        while current is not None:
            path.append(current)
            prev, pool = predecessor[current]
            if pool is not None:
                pools.append(pool)
            current = prev
            hop_guard += 1
            if hop_guard > len(vertices) + 1:
                fallback_path, fallback_pools = self._fallback_dijkstra_path(start, end)
                return fallback_path, fallback_pools, negative_cycle_detected

        path.reverse()
        pools.reverse()
        if not path or path[0] != start:
            fallback_path, fallback_pools = self._fallback_dijkstra_path(start, end)
            return fallback_path, fallback_pools, negative_cycle_detected
        return path, pools, negative_cycle_detected

    def _fallback_dijkstra_path(
        self, start: str, end: str
    ) -> Tuple[Optional[List[str]], Optional[List[Pool]]]:
        """
        Fallback path reconstruction using Dijkstra when predecessor chain is unstable.

        This can happen when a reachable negative cycle exists under spot weights.
        """
        from .dijkstra import DijkstraRouting

        return DijkstraRouting(self.pools)._dijkstra_find_path(start, end, amount=0.0)
