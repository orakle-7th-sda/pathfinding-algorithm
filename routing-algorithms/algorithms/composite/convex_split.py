"""
Convex Optimization Split Algorithm.

Uses convex optimization to find the optimal split that minimizes
price impact while maximizing output.
"""

from typing import List, Dict, Optional
import math
import time

from ..base import (
    BaseAlgorithm, AlgorithmMetadata, Difficulty, AlgorithmType,
    Pool, SwapRequest, SwapResult, SwapRoute
)

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ConvexSplit(BaseAlgorithm):
    """
    Convex Optimization split routing algorithm.
    
    Formulates the split problem as a convex optimization problem
    and uses gradient descent to find the optimal allocation.
    
    For AMM pools, the output function is concave, making this
    a convex maximization problem (or concave minimization of negative).
    
    Difficulty: Hard
    Time Complexity: O(I * P) where I is iterations, P is pools
    Space Complexity: O(P)
    """
    
    MAX_ITERATIONS = 1000
    LEARNING_RATE = 0.01
    TOLERANCE = 1e-8
    
    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Convex Optimization Split",
            difficulty=Difficulty.HARD,
            algorithm_type=AlgorithmType.COMPOSITE,
            time_complexity="O(I * P)",
            space_complexity="O(P)",
            description="Uses convex optimization with gradient descent to find "
                       "optimal split ratios that minimize price impact.",
            pros=[
                "Theoretically optimal for AMM pools",
                "Handles continuous allocation",
                "Converges to global optimum",
                "Elegant mathematical formulation"
            ],
            cons=[
                "Requires differentiable output function",
                "Convergence can be slow",
                "Numerical precision issues",
                "More complex implementation"
            ]
        )
    
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find optimal split using convex optimization.
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
        
        # Run convex optimization
        allocations = self._optimize_allocation(pools, request.amount_in, request.token_in)
        
        # Build routes
        routes = []
        total_output = 0.0
        total_impact = 0.0
        
        for pool, amount in allocations.items():
            if amount > 1e-10:  # Ignore very small allocations
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
    
    def _optimize_allocation(self, pools: List[Pool], total_amount: float,
                              token_in: str) -> Dict[str, float]:
        """
        Optimize allocation using projected gradient descent.
        
        Objective: maximize sum of outputs
        Constraint: sum of allocations = total_amount, all allocations >= 0
        """
        n = len(pools)
        
        # Initialize with equal allocation
        x = [total_amount / n] * n
        
        for iteration in range(self.MAX_ITERATIONS):
            # Compute gradient (numerical)
            grad = self._compute_gradient(pools, x, token_in)
            
            # Gradient ascent step (we want to maximize)
            x_new = [x[i] + self.LEARNING_RATE * grad[i] for i in range(n)]
            
            # Project onto simplex (constraint: sum = total_amount, all >= 0)
            x_new = self._project_simplex(x_new, total_amount)
            
            # Check convergence
            diff_norm = math.sqrt(sum((x_new[i] - x[i])**2 for i in range(n)))
            if diff_norm < self.TOLERANCE:
                break
            
            x = x_new
        
        # Build result dictionary
        result = {}
        for i, pool in enumerate(pools):
            if x[i] > 1e-10:
                result[pool.pool_id] = float(x[i])
        
        return result
    
    def _compute_gradient(self, pools: List[Pool], x: List[float],
                          token_in: str, eps: float = 1e-6) -> List[float]:
        """
        Compute numerical gradient of total output with respect to allocations.
        """
        n = len(pools)
        grad = [0.0] * n
        
        for i in range(n):
            # Partial derivative with respect to x[i]
            x_plus = x.copy()
            x_plus[i] += eps
            
            x_minus = x.copy()
            x_minus[i] -= eps
            
            f_plus = self._total_output(pools, x_plus, token_in)
            f_minus = self._total_output(pools, x_minus, token_in)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def _total_output(self, pools: List[Pool], allocations: List[float],
                      token_in: str) -> float:
        """Calculate total output for given allocations."""
        total = 0.0
        for i, pool in enumerate(pools):
            if allocations[i] > 0:
                total += pool.get_amount_out(allocations[i], token_in)
        return total
    
    def _project_simplex(self, x: List[float], target_sum: float) -> List[float]:
        """
        Project x onto the simplex {y : sum(y) = target_sum, y >= 0}.
        
        Uses the algorithm from "Efficient Projections onto the l1-Ball
        for Learning in High Dimensions" (Duchi et al., 2008).
        """
        n = len(x)
        
        # Sort in descending order
        sorted_x = sorted(x, reverse=True)
        
        # Find the right threshold
        cumsum = []
        s = 0
        for val in sorted_x:
            s += val
            cumsum.append(s)
        
        rho = -1
        for i in range(n):
            if sorted_x[i] > (cumsum[i] - target_sum) / (i + 1):
                rho = i
        
        if rho < 0:
            # All zeros except uniform distribution
            return [target_sum / n] * n
        
        theta = (cumsum[rho] - target_sum) / (rho + 1)
        
        # Project
        return [max(xi - theta, 0) for xi in x]
    
    def _optimize_with_scipy(self, pools: List[Pool], total_amount: float,
                              token_in: str) -> Dict[str, float]:
        """
        Alternative implementation using scipy.optimize.
        
        Requires scipy and numpy to be installed.
        """
        if not HAS_NUMPY:
            return self._optimize_allocation(pools, total_amount, token_in)
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback to manual implementation
            return self._optimize_allocation(pools, total_amount, token_in)
        
        n = len(pools)
        
        def negative_output(x):
            """Negative of total output (we minimize this)."""
            total = 0.0
            for i, pool in enumerate(pools):
                if x[i] > 0:
                    total += pool.get_amount_out(x[i], token_in)
            return -total
        
        # Constraint: sum = total_amount
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_amount}
        
        # Bounds: all >= 0
        bounds = [(0, total_amount) for _ in range(n)]
        
        # Initial guess
        x0 = np.array([total_amount / n] * n)
        
        result = minimize(
            negative_output,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.MAX_ITERATIONS}
        )
        
        # Build result dictionary
        allocations = {}
        for i, pool in enumerate(pools):
            if result.x[i] > 1e-10:
                allocations[pool.pool_id] = float(result.x[i])
        
        return allocations
