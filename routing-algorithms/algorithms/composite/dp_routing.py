"""
Dynamic Programming Split Routing Algorithm.

Uses DP to find optimal allocation across multiple pools.
"""

from typing import List, Dict, Tuple
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)

# Optional numpy import for optimized version
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DPRouting(BaseAlgorithm):
    """
    Dynamic Programming routing algorithm.
    
    Uses dynamic programming to find the optimal split ratios
    across multiple pools, discretizing the allocation space.
    
    Difficulty: Hard
    Time Complexity: O(P * N^2) where P is pools, N is discretization steps
    Space Complexity: O(P * N)
    """
    
    DISCRETIZATION_STEPS = 100  # Number of allocation levels
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="DP Split Routing",
            difficulty=Difficulty.HARD,
            algorithm_type=AlgorithmType.COMPOSITE,
            time_complexity="O(P * N^2)",
            space_complexity="O(P * N)",
            description="Uses dynamic programming to find optimal split ratios. "
                       "Discretizes the allocation space for tractability.",
            pros=[
                "Near-optimal split allocation",
                "Considers all pool combinations",
                "Provably good within discretization",
                "Handles non-linear price impact"
            ],
            cons=[
                "Computationally expensive",
                "Discretization introduces error",
                "Memory intensive",
                "May not scale to many pools"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find optimal split using dynamic programming.
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
        
        if len(pools) == 1:
            # Only one pool, no splitting needed
            output = pools[0].get_amount_out(request.amount_in, request.token_in)
            route = SwapRoute(
                path=[request.token_in, request.token_out],
                pools=[pools[0]],
                dexes=[pools[0].dex],
                amount_in=request.amount_in,
                amount_out=output
            )
            end_time = time.perf_counter()
            return SwapResult(
                routes=[route],
                total_amount_in=request.amount_in,
                total_amount_out=output,
                total_price_impact=route.price_impact,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name
            )
        
        # Run DP to find optimal allocation
        allocations = self._dp_optimal_split(pools, request.amount_in, request.token_in)
        
        # Build routes
        routes = []
        total_output = 0.0
        total_impact = 0.0
        
        for pool, amount in allocations.items():
            if amount > 0:
                pool_obj = next(p for p in pools if p.pool_id == pool)
                output = pool_obj.get_amount_out(amount, request.token_in)
                impact = pool_obj.get_price_impact(amount, request.token_in)
                
                route = SwapRoute(
                    path=[request.token_in, request.token_out],
                    pools=[pool_obj],
                    dexes=[pool_obj.dex],
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
    
    def _dp_optimal_split(self, pools: List[Pool], total_amount: float,
                          token_in: str) -> Dict[str, float]:
        """
        Use DP to find optimal allocation.
        
        State: dp[i][j] = max output using first i pools with j units of input
        
        Returns:
            Dictionary mapping pool_id to allocation amount
        """
        n_pools = len(pools)
        n_steps = self.DISCRETIZATION_STEPS
        unit = total_amount / n_steps
        
        # dp[i][j] = (max_output, allocations_dict)
        # Using j units with first i pools
        dp = [[(-float('inf'), {}) for _ in range(n_steps + 1)] 
              for _ in range(n_pools + 1)]
        
        # Base case: 0 pools, 0 units = 0 output
        dp[0][0] = (0.0, {})
        
        for i in range(1, n_pools + 1):
            pool = pools[i - 1]
            
            for j in range(n_steps + 1):
                # Option 1: Don't use this pool at all
                if dp[i-1][j][0] > dp[i][j][0]:
                    dp[i][j] = dp[i-1][j]
                
                # Option 2: Allocate k units to this pool
                for k in range(j + 1):
                    remaining = j - k
                    
                    if dp[i-1][remaining][0] == -float('inf'):
                        continue
                    
                    # Calculate output from this pool with k units
                    amount = k * unit
                    output = pool.get_amount_out(amount, token_in) if amount > 0 else 0
                    
                    total_output = dp[i-1][remaining][0] + output
                    
                    if total_output > dp[i][j][0]:
                        new_allocs = dp[i-1][remaining][1].copy()
                        if amount > 0:
                            new_allocs[pool.pool_id] = amount
                        dp[i][j] = (total_output, new_allocs)
        
        # Return the optimal allocation for all units
        return dp[n_pools][n_steps][1]
    
    def _dp_optimal_split_optimized(self, pools: List[Pool], total_amount: float,
                                     token_in: str) -> Dict[str, float]:
        """
        Optimized DP using numpy for better performance.
        
        Trade-off: Faster but uses more memory.
        Falls back to basic DP if numpy is not available.
        """
        if not HAS_NUMPY:
            return self._dp_optimal_split(pools, total_amount, token_in)
        
        n_pools = len(pools)
        n_steps = self.DISCRETIZATION_STEPS
        unit = total_amount / n_steps
        
        # Precompute all outputs
        outputs = np.zeros((n_pools, n_steps + 1))
        for i, pool in enumerate(pools):
            for j in range(n_steps + 1):
                amount = j * unit
                outputs[i, j] = pool.get_amount_out(amount, token_in) if amount > 0 else 0
        
        # DP table
        dp = np.full((n_pools + 1, n_steps + 1), -np.inf)
        dp[0, 0] = 0.0
        
        # Track allocations
        alloc = np.zeros((n_pools + 1, n_steps + 1, n_pools), dtype=int)
        
        for i in range(1, n_pools + 1):
            for j in range(n_steps + 1):
                # Don't use this pool
                if dp[i-1, j] > dp[i, j]:
                    dp[i, j] = dp[i-1, j]
                    alloc[i, j] = alloc[i-1, j]
                
                # Use k units for this pool
                for k in range(j + 1):
                    remaining = j - k
                    if dp[i-1, remaining] == -np.inf:
                        continue
                    
                    total_out = dp[i-1, remaining] + outputs[i-1, k]
                    
                    if total_out > dp[i, j]:
                        dp[i, j] = total_out
                        alloc[i, j] = alloc[i-1, remaining].copy()
                        alloc[i, j, i-1] = k
        
        # Convert to dictionary
        result = {}
        for i, pool in enumerate(pools):
            units = alloc[n_pools, n_steps, i]
            if units > 0:
                result[pool.pool_id] = units * unit
        
        return result
