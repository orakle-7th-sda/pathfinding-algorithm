"""
BFS (Breadth-First Search) Routing Algorithm.

Finds the shortest path in terms of number of hops.
Does not consider edge weights (prices/slippage).
"""

from collections import deque
from typing import List, Optional, Dict, Tuple
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class BFSRouting(BaseAlgorithm):
    """
    BFS-based routing algorithm.
    
    Finds the path with minimum number of hops (intermediate swaps).
    Fast but doesn't optimize for output amount.
    
    Difficulty: Easy
    Time Complexity: O(V + E) where V = tokens, E = pools
    Space Complexity: O(V) for visited set
    """
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="BFS Routing",
            difficulty=Difficulty.EASY,
            algorithm_type=AlgorithmType.SINGLE,
            time_complexity="O(V + E)",
            space_complexity="O(V)",
            description="Finds the path with minimum number of hops. "
                       "Fast but may not find the best price.",
            pros=[
                "Very fast execution",
                "Guaranteed to find shortest path (by hops)",
                "Simple implementation",
                "Low gas cost routes (fewer hops)"
            ],
            cons=[
                "Ignores price/slippage optimization",
                "May miss better routes with more hops",
                "Not suitable for large swaps"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find the shortest path using BFS.
        """
        start_time = time.perf_counter()
        
        path, pools = self._bfs_find_path(request.token_in, request.token_out)
        
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
        
        # Calculate output
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
    
    def _bfs_find_path(self, start: str, end: str) -> Tuple[Optional[List[str]], 
                                                            Optional[List[Pool]]]:
        """
        BFS to find shortest path.
        
        Returns:
            Tuple of (path, pools) or (None, None) if no path found
        """
        if start == end:
            return [start], []
        
        if start not in self.token_graph:
            return None, None
        
        queue = deque([(start, [start], [])])
        visited = {start}
        
        while queue:
            current, path, pools = queue.popleft()
            
            if current not in self.token_graph:
                continue
            
            for next_token, pool in self.token_graph[current]:
                if next_token == end:
                    return path + [next_token], pools + [pool]
                
                if next_token not in visited:
                    visited.add(next_token)
                    queue.append((
                        next_token,
                        path + [next_token],
                        pools + [pool]
                    ))
        
        return None, None
