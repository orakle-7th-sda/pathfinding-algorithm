"""
Multi-Hop Routing Algorithm - Route through intermediate tokens.

Finds the best path that may involve multiple intermediate tokens,
potentially getting better rates than direct swaps.
"""

from typing import List, Optional, Dict, Tuple
import heapq
import math
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class MultiHopRouting(BaseAlgorithm):
    """
    Multi-Hop routing algorithm.
    
    Explores paths through intermediate tokens (e.g., ETH -> WBTC -> USDC)
    to potentially find better rates than direct swaps.
    
    Difficulty: Medium
    Time Complexity: O((V + E) log V) similar to Dijkstra
    Space Complexity: O(V)
    """
    
    MAX_HOPS = 4  # Maximum number of intermediate swaps
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Multi-Hop Routing",
            difficulty=Difficulty.MEDIUM,
            algorithm_type=AlgorithmType.COMPOSITE,
            time_complexity="O((V + E) log V)",
            space_complexity="O(V)",
            description="Routes through intermediate tokens to find better rates. "
                       "A->B->C might be better than A->C directly.",
            pros=[
                "Can find better rates via intermediate tokens",
                "Explores full graph potential",
                "Handles illiquid pairs well",
                "Works when direct swap unavailable"
            ],
            cons=[
                "Higher gas cost for multiple hops",
                "More slippage opportunities",
                "Complexity increases with hops",
                "May not beat direct for liquid pairs"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find the best multi-hop route.
        """
        start_time = time.perf_counter()
        
        # Find all paths up to MAX_HOPS
        all_paths = self._find_all_paths_limited(
            request.token_in,
            request.token_out,
            self.MAX_HOPS
        )
        
        if not all_paths:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=0.0,
                algorithm_name=self.metadata.name
            )
        
        # Evaluate each path and find the best one
        best_path = None
        best_pools = None
        best_output = 0.0
        
        for path, pools in all_paths:
            output = self.calculate_route_output(path, pools, request.amount_in)
            if output > best_output:
                best_output = output
                best_path = path
                best_pools = pools
        
        end_time = time.perf_counter()
        
        if best_path is None:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name
            )
        
        route = SwapRoute(
            path=best_path,
            pools=best_pools,
            dexes=[p.dex for p in best_pools],
            amount_in=request.amount_in,
            amount_out=best_output
        )
        
        # Gas estimate increases with hops
        base_gas = 150000
        gas_per_hop = 80000
        gas_estimate = base_gas + (len(best_pools) - 1) * gas_per_hop
        
        return SwapResult(
            routes=[route],
            total_amount_in=request.amount_in,
            total_amount_out=best_output,
            total_price_impact=route.price_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name,
            gas_estimate=gas_estimate
        )
    
    def _find_all_paths_limited(self, start: str, end: str,
                                 max_hops: int) -> List[Tuple[List[str], List[Pool]]]:
        """
        Find all paths from start to end with limited hops.
        """
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str],
                pools: List[Pool], visited: set):
            if len(pools) > max_hops:
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
    
    def find_best_route_with_splits(self, request: SwapRequest,
                                     num_splits: int = 3) -> SwapResult:
        """
        Find best multi-hop routes and optionally split among them.
        
        Advanced version that combines multi-hop with splitting.
        """
        start_time = time.perf_counter()
        
        all_paths = self._find_all_paths_limited(
            request.token_in,
            request.token_out,
            self.MAX_HOPS
        )
        
        if not all_paths:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=0.0,
                algorithm_name=self.metadata.name
            )
        
        # Evaluate and rank paths
        path_scores = []
        for path, pools in all_paths:
            output = self.calculate_route_output(path, pools, request.amount_in / num_splits)
            effective_rate = output / (request.amount_in / num_splits)
            path_scores.append((effective_rate, path, pools))
        
        # Sort by effective rate (descending)
        path_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Take top N paths for splitting
        top_paths = path_scores[:num_splits]
        
        routes = []
        total_output = 0.0
        split_amount = request.amount_in / len(top_paths)
        
        for _, path, pools in top_paths:
            output = self.calculate_route_output(path, pools, split_amount)
            
            route = SwapRoute(
                path=path,
                pools=pools,
                dexes=[p.dex for p in pools],
                amount_in=split_amount,
                amount_out=output
            )
            routes.append(route)
            total_output += output
        
        end_time = time.perf_counter()
        
        # Calculate total price impact (weighted average)
        total_impact = sum(r.price_impact * r.amount_in / request.amount_in for r in routes)
        
        return SwapResult(
            routes=routes,
            total_amount_in=request.amount_in,
            total_amount_out=total_output,
            total_price_impact=total_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name + " (Split)"
        )
