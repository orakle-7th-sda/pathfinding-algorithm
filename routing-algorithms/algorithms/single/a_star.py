"""
A* Search Algorithm for DEX Routing.

Uses heuristics to guide the search towards the goal more efficiently.
"""

import heapq
from typing import List, Optional, Dict, Tuple, Callable
import math
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class AStarRouting(BaseAlgorithm):
    """
    A* Search routing algorithm.
    
    Uses a heuristic function to estimate remaining cost to goal,
    potentially finding optimal paths faster than Dijkstra.
    
    Difficulty: Hard
    Time Complexity: O(E) in best case, O(V^2) in worst case
    Space Complexity: O(V)
    """
    
    def __init__(self, pools: List[Pool]):
        super().__init__(pools)
        self._precompute_heuristics()
    
    def _precompute_heuristics(self):
        """Precompute best rates for heuristic estimation."""
        # Store best known rates between token pairs
        self.best_rates: Dict[Tuple[str, str], float] = {}
        
        for pool in self.pools:
            # Rate from token0 to token1
            rate01 = pool.get_spot_price(pool.token0)
            key01 = (pool.token0, pool.token1)
            if key01 not in self.best_rates or rate01 > self.best_rates[key01]:
                self.best_rates[key01] = rate01
            
            # Rate from token1 to token0
            rate10 = pool.get_spot_price(pool.token1)
            key10 = (pool.token1, pool.token0)
            if key10 not in self.best_rates or rate10 > self.best_rates[key10]:
                self.best_rates[key10] = rate10
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="A* Search Routing",
            difficulty=Difficulty.HARD,
            algorithm_type=AlgorithmType.SINGLE,
            time_complexity="O(E) best, O(V^2) worst",
            space_complexity="O(V)",
            description="Uses heuristic-guided search to find optimal routes "
                       "more efficiently than Dijkstra.",
            pros=[
                "Faster than Dijkstra in many cases",
                "Optimal if heuristic is admissible",
                "Explores fewer nodes",
                "Good for large sparse graphs"
            ],
            cons=[
                "Heuristic design is challenging",
                "May degrade to Dijkstra performance",
                "More complex implementation",
                "Heuristic quality affects performance"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find the best route using A* search.
        """
        start_time = time.perf_counter()
        
        path, pools = self._astar_find_path(
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
    
    def _heuristic(self, current: str, goal: str, _amount: float = 0) -> float:
        """
        Estimate the cost from current token to goal.
        
        Uses the best known direct rate if available,
        otherwise estimates based on average rates.
        
        Returns negative log of estimated rate (admissible heuristic).
        """
        if current == goal:
            return 0.0
        
        # Check if we have a direct rate estimate
        key = (current, goal)
        if key in self.best_rates:
            # Use best known rate as optimistic estimate
            best_rate = self.best_rates[key]
            return -math.log(best_rate) if best_rate > 0 else 0.0
        
        # No direct rate, use optimistic estimate (0 cost)
        # This ensures admissibility but may not be tight
        return 0.0
    
    def _get_edge_weight_spot(self, pool: Pool, token_in: str) -> float:
        """Edge weight as negative log of spot rate. Aligns with Dijkstra."""
        rate = pool.get_spot_price(token_in)
        if rate <= 0:
            return float('inf')
        return -math.log(rate)

    def _astar_find_path(self, start: str, end: str,
                          amount: float) -> Tuple[Optional[List[str]],
                                                  Optional[List[Pool]]]:
        """
        A* search using spot-price edge weights (same as Dijkstra).
        Path is optimal for marginal rates; actual output may be lower for large trades.
        """
        if start == end:
            return [start], []
        if start not in self.token_graph:
            return None, None

        g_score: Dict[str, float] = {start: 0.0}
        f_score: Dict[str, float] = {start: self._heuristic(start, end, amount)}
        predecessor: Dict[str, Tuple[Optional[str], Optional[Pool]]] = {start: (None, None)}
        counter = 0
        open_set = [(f_score[start], counter, start)]
        open_set_tokens = {start}
        closed_set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current not in open_set_tokens:
                continue
            open_set_tokens.remove(current)
            if current == end:
                path, pools = [], []
                node = end
                while node is not None:
                    path.append(node)
                    prev, pool = predecessor[node]
                    if pool is not None:
                        pools.append(pool)
                    node = prev
                path.reverse()
                pools.reverse()
                return path, pools
            closed_set.add(current)
            if current not in self.token_graph:
                continue
            for next_token, pool in self.token_graph[current]:
                if next_token in closed_set:
                    continue
                edge_weight = self._get_edge_weight_spot(pool, current)
                tentative_g = g_score[current] + edge_weight
                if next_token not in g_score or tentative_g < g_score[next_token]:
                    predecessor[next_token] = (current, pool)
                    g_score[next_token] = tentative_g
                    f_score[next_token] = tentative_g + self._heuristic(next_token, end, amount)
                    if next_token not in open_set_tokens:
                        counter += 1
                        heapq.heappush(open_set, (f_score[next_token], counter, next_token))
                        open_set_tokens.add(next_token)
        return None, None
