"""
Simple Split Algorithm - Equal distribution across DEXes.

Splits the order equally among available DEXes for the same token pair.
"""

from typing import List
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)


class SimpleSplit(BaseAlgorithm):
    """
    Simple Split routing algorithm.
    
    Distributes the swap amount equally among all available pools
    for the given token pair. Simple but can reduce price impact
    for large orders.
    
    Difficulty: Easy
    Time Complexity: O(P) where P is number of pools
    Space Complexity: O(P)
    """
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Simple Split",
            difficulty=Difficulty.EASY,
            algorithm_type=AlgorithmType.COMPOSITE,
            time_complexity="O(P)",
            space_complexity="O(P)",
            description="Splits the order equally among all available DEXes. "
                       "Simple approach to reduce price impact.",
            pros=[
                "Simple to implement",
                "Reduces price impact for large orders",
                "Predictable behavior",
                "Fast execution"
            ],
            cons=[
                "Ignores liquidity differences",
                "Not optimal for varying pool sizes",
                "May use pools with bad rates",
                "Fixed split ratio"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Split the order equally among available pools.
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
        
        # Split equally
        num_pools = len(pools)
        amount_per_pool = request.amount_in / num_pools
        
        routes = []
        total_output = 0.0
        total_impact = 0.0
        
        for pool in pools:
            output = pool.get_amount_out(amount_per_pool, request.token_in)
            impact = pool.get_price_impact(amount_per_pool, request.token_in)
            
            route = SwapRoute(
                path=[request.token_in, request.token_out],
                pools=[pool],
                dexes=[pool.dex],
                amount_in=amount_per_pool,
                amount_out=output
            )
            routes.append(route)
            
            total_output += output
            total_impact += impact / num_pools  # Weighted average
        
        end_time = time.perf_counter()
        
        # Calculate gas estimate (more routes = more gas)
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
