"""
Dijkstra's Algorithm for DEX Routing.

Finds the path that maximizes output amount by using negative log prices as weights.
"""

import heapq
from typing import List, Optional, Dict, Tuple
import math
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class DijkstraRouting(BaseAlgorithm):
    """
    Dijkstra-based routing algorithm.
    
    Uses negative log of exchange rates as edge weights to convert
    multiplication to addition, allowing Dijkstra's algorithm to find
    the path that maximizes output.
    
    Difficulty: Medium
    Time Complexity: O((V + E) log V) with binary heap
    Space Complexity: O(V) for distance array
    """
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Dijkstra Routing",
            difficulty=Difficulty.MEDIUM,
            algorithm_type=AlgorithmType.SINGLE,
            time_complexity="O((V + E) log V)",
            space_complexity="O(V)",
            description="Finds path optimal for spot (marginal) rates using "
                       "negative log of spot price as edge weight. For large "
                       "trades, actual output may be lower than Naive/MultiHop.",
            pros=[
                "Optimal for spot rates (Dijkstra correctness)",
                "Efficient for sparse graphs",
                "Well-understood algorithm"
            ],
            cons=[
                "Uses spot price only; ignores slippage for large amounts",
                "Cannot handle negative weights (arbitrage)",
                "Single path only (no splitting)"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find the best route using Dijkstra's algorithm.
        
        Uses -log(exchange_rate) as edge weight so that
        minimizing sum of weights = maximizing product of rates.
        """
        start_time = time.perf_counter()
        
        path, pools = self._dijkstra_find_path(
            request.token_in, 
            request.token_out,
            request.amount_in
        )
        
        end_time = time.perf_counter()
        
        if path is None:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name
            )
        
        # Calculate actual output
        output = self.calculate_route_output(path, pools, request.amount_in)
        
        route = SwapRoute(
            path=path,
            pools=pools,
            dexes=[p.dex for p in pools],
            amount_in=request.amount_in,
            amount_out=output
        )
        
        return SwapResult(
            routes=[route],
            total_amount_in=request.amount_in,
            total_amount_out=output,
            total_price_impact=route.price_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name
        )
    
    def _get_edge_weight(self, pool: Pool, token_in: str, amount: float) -> float:
        """
        Calculate edge weight as negative log of effective rate (amount-dependent).
        Used when amount is considered; for fixed weights use _get_edge_weight_spot.
        """
        amount_out = pool.get_amount_out(amount, token_in)
        if amount_out <= 0:
            return float('inf')
        effective_rate = amount_out / amount
        if effective_rate <= 0:
            return float('inf')
        return -math.log(effective_rate)

    def _get_edge_weight_spot(self, pool: Pool, token_in: str) -> float:
        """
        Calculate edge weight as negative log of spot (marginal) rate.
        Does not depend on amount, so Dijkstra's optimality holds.
        The path found is optimal for infinitesimal trades; for large trades
        actual output may be lower than Naive/MultiHop due to slippage.
        """
        rate = pool.get_spot_price(token_in)
        if rate <= 0:
            return float('inf')
        return -math.log(rate)

    def _dijkstra_find_path(self, start: str, end: str,
                            amount: float) -> Tuple[Optional[List[str]],
                                                    Optional[List[Pool]]]:
        """
        Dijkstra's algorithm using spot-price edge weights.
        Finds the path that is optimal for marginal (spot) rates.
        For large trades, this path may be suboptimal due to slippage,
        so output can be lower than Naive/MultiHop which maximize actual output.
        """
        if start == end:
            return [start], []
        
        if start not in self.token_graph:
            return None, None
        
        distances: Dict[str, Tuple[float, Optional[str], Optional[Pool]]] = {}
        distances[start] = (0.0, None, None)
        pq = [(0.0, start)]
        visited = set()

        while pq:
            dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                break
            if current not in self.token_graph:
                continue
            for next_token, pool in self.token_graph[current]:
                if next_token in visited:
                    continue
                weight = self._get_edge_weight_spot(pool, current)
                new_dist = dist + weight
                if next_token not in distances or new_dist < distances[next_token][0]:
                    distances[next_token] = (new_dist, current, pool)
                    heapq.heappush(pq, (new_dist, next_token))

        if end not in distances:
            return None, None
        path = []
        pools = []
        current = end
        while current is not None:
            path.append(current)
            _, prev, pool = distances[current]
            if pool is not None:
                pools.append(pool)
            current = prev
        path.reverse()
        pools.reverse()
        return path, pools
