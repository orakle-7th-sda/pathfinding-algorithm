"""
Naive Brute Force Algorithm - Baseline implementation.

Explores all possible paths to find the best route.
This serves as a baseline for comparing other algorithms.
"""

from typing import List, Optional
from itertools import permutations
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class NaiveBruteForce(BaseAlgorithm):
    """
    Naive Brute Force routing algorithm.
    
    Explores all possible paths up to a maximum depth to find the best route.
    Very inefficient but guaranteed to find the optimal solution within
    the search space.
    
    Difficulty: Easy
    Time Complexity: O(n!) where n is number of intermediate tokens
    Space Complexity: O(n) for path storage
    """
    
    MAX_PATH_LENGTH = 4  # Limit to prevent explosion
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Naive Brute Force",
            difficulty=Difficulty.EASY,
            algorithm_type=AlgorithmType.SINGLE,
            time_complexity="O(n!)",
            space_complexity="O(n)",
            description="Explores all possible paths to find the best route. "
                       "Serves as a baseline for comparison.",
            pros=[
                "Guaranteed to find optimal solution within search space",
                "Simple to understand and implement",
                "Good for small graphs"
            ],
            cons=[
                "Extremely slow for large graphs",
                "Exponential time complexity",
                "Not practical for production use"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find the best route by exploring all possible paths.
        """
        start_time = time.perf_counter()
        
        all_paths = self._find_all_paths(
            request.token_in, 
            request.token_out,
            max_length=self.MAX_PATH_LENGTH
        )
        
        if not all_paths:
            # No path found
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=0.0,
                algorithm_name=self.metadata.name
            )
        
        best_route: Optional[SwapRoute] = None
        best_output = 0.0
        
        for path, pools in all_paths:
            output = self.calculate_route_output(path, pools, request.amount_in)
            
            if output > best_output:
                best_output = output
                best_route = SwapRoute(
                    path=path,
                    pools=pools,
                    dexes=[p.dex for p in pools],
                    amount_in=request.amount_in,
                    amount_out=output
                )
        
        end_time = time.perf_counter()
        
        if best_route is None:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name
            )
        
        return SwapResult(
            routes=[best_route],
            total_amount_in=request.amount_in,
            total_amount_out=best_output,
            total_price_impact=best_route.price_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name
        )
    
    def _find_all_paths(self, start: str, end: str, 
                        max_length: int) -> List[tuple]:
        """
        Find all possible paths from start to end token.
        
        Returns:
            List of (path, pools) tuples
        """
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str], 
                pools: List[Pool], visited: set):
            if len(path) > max_length:
                return
            
            if current == target and len(path) > 1:
                all_paths.append((path.copy(), pools.copy()))
                return
            
            if current not in self.token_graph:
                return
            
            for next_token, pool in self.token_graph[current]:
                if next_token not in visited:
                    visited.add(next_token)
                    path.append(next_token)
                    pools.append(pool)
                    
                    dfs(next_token, target, path, pools, visited)
                    
                    visited.remove(next_token)
                    path.pop()
                    pools.pop()
        
        visited = {start}
        dfs(start, end, [start], [], visited)
        
        return all_paths
