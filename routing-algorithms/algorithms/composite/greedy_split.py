"""
Greedy Split Algorithm - Liquidity-based distribution.

Distributes the order based on pool liquidity to minimize price impact.
"""

from typing import List, Tuple
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class GreedySplit(BaseAlgorithm):
    """
    Greedy Split routing algorithm.
    
    Distributes the swap amount proportionally to pool liquidity.
    Uses a greedy approach to allocate more to pools with better rates.
    
    Difficulty: Medium
    Time Complexity: O(P * I) where P is pools, I is iterations
    Space Complexity: O(P)
    """
    
    ITERATIONS = 100  # Number of allocation iterations
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Greedy Split",
            difficulty=Difficulty.MEDIUM,
            algorithm_type=AlgorithmType.COMPOSITE,
            time_complexity="O(P * I)",
            space_complexity="O(P)",
            description="Distributes the order based on pool liquidity and rates. "
                       "Greedy allocation to pools with best marginal rates.",
            pros=[
                "Better than simple split",
                "Considers liquidity differences",
                "Adaptive allocation",
                "Good practical performance"
            ],
            cons=[
                "May not find global optimum",
                "Iteration count affects quality",
                "Greedy decisions can be suboptimal",
                "No guarantee of optimality"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Split the order greedily based on marginal rates.
        """
        start_time = time.perf_counter()
        
        # Find all pools for this token pair
        pools = self.get_pools_for_pair(request.token_in, request.token_out)
        
        if not pools:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=0.0,
                algorithm_name=self.metadata.name
            )
        
        # Initialize allocations
        allocations = {pool.pool_id: 0.0 for pool in pools}
        chunk_size = request.amount_in / self.ITERATIONS
        
        # Greedy allocation: assign each chunk to pool with best marginal rate
        for _ in range(self.ITERATIONS):
            best_pool = None
            best_marginal_output = 0.0
            
            for pool in pools:
                current_allocation = allocations[pool.pool_id]
                
                # Calculate marginal output for adding this chunk
                output_before = pool.get_amount_out(current_allocation, request.token_in) if current_allocation > 0 else 0
                output_after = pool.get_amount_out(current_allocation + chunk_size, request.token_in)
                marginal_output = output_after - output_before
                
                if marginal_output > best_marginal_output:
                    best_marginal_output = marginal_output
                    best_pool = pool
            
            if best_pool:
                allocations[best_pool.pool_id] += chunk_size
        
        # Build routes from allocations
        routes = []
        total_output = 0.0
        total_impact = 0.0
        
        for pool in pools:
            amount = allocations[pool.pool_id]
            if amount > 0:
                output = pool.get_amount_out(amount, request.token_in)
                impact = pool.get_price_impact(amount, request.token_in)
                
                route = SwapRoute(
                    path=[request.token_in, request.token_out],
                    pools=[pool],
                    dexes=[pool.dex],
                    amount_in=amount,
                    amount_out=output
                )
                routes.append(route)
                
                total_output += output
                total_impact += impact * (amount / request.amount_in)
        
        end_time = time.perf_counter()
        
        # Calculate gas estimate
        base_gas = 150000
        gas_per_route = 100000
        gas_estimate = base_gas + (len(routes) - 1) * gas_per_route
        
        return SwapResult(
            routes=routes,
            total_amount_in=request.amount_in,
            total_amount_out=total_output,
            total_price_impact=total_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name,
            gas_estimate=gas_estimate
        )
