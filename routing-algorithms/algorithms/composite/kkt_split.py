"""
KKT-based optimal split for constant-product AMM pools.

Maximizes total output under:
  - sum(allocation_i) = total_amount
  - allocation_i >= 0
for direct pools of the same token pair.
"""

from typing import Dict, List, Tuple
import math
import time

from ..base import (
    BaseAlgorithm,
    AlgorithmMetadata,
    Difficulty,
    AlgorithmType,
    Pool,
    SwapRequest,
    SwapResult,
    SwapRoute,
)


class KKTOptimalSplit(BaseAlgorithm):
    """
    Closed-form + bisection optimal splitter.

    For each pool i:
      out_i(a_i) = y_i * alpha_i * a_i / (x_i + alpha_i * a_i)
      d(out_i)/d(a_i) = c_i / (x_i + alpha_i * a_i)^2
      where c_i = y_i * alpha_i * x_i

    At optimum over active pools, marginal outputs are equal (KKT):
      d(out_i)/d(a_i) = lambda
    """

    BISECTION_ITERATIONS = 100
    ALLOCATION_EPS = 1e-10

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="KKT Optimal Split",
            difficulty=Difficulty.HARD,
            algorithm_type=AlgorithmType.COMPOSITE,
            time_complexity="O(P * log(1/eps))",
            space_complexity="O(P)",
            description=(
                "Solves split allocation with KKT conditions for constant-product "
                "pools, yielding fast and stable near-closed-form optimization."
            ),
            pros=[
                "Fast compared to iterative numerical gradients",
                "Directly grounded in AMM concavity/KKT",
                "Stable allocation with explicit constraints",
            ],
            cons=[
                "Assumes constant-product style pools",
                "Direct pair split only (no multi-hop in this class)",
                "Does not model gas in objective",
            ],
        )

    def find_best_route(self, request: SwapRequest) -> SwapResult:
        start_time = time.perf_counter()

        pools = self.get_pools_for_pair(request.token_in, request.token_out)
        if not pools:
            return SwapResult(
                routes=[],
                total_amount_in=request.amount_in,
                total_amount_out=0.0,
                total_price_impact=1.0,
                execution_time_ms=0.0,
                algorithm_name=self.metadata.name,
            )

        if len(pools) == 1:
            output = pools[0].get_amount_out(request.amount_in, request.token_in)
            route = SwapRoute(
                path=[request.token_in, request.token_out],
                pools=[pools[0]],
                dexes=[pools[0].dex],
                amount_in=request.amount_in,
                amount_out=output,
            )
            end_time = time.perf_counter()
            return SwapResult(
                routes=[route],
                total_amount_in=request.amount_in,
                total_amount_out=output,
                total_price_impact=route.price_impact,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm_name=self.metadata.name,
            )

        allocations = self._solve_kkt_allocations(pools, request.token_in, request.amount_in)

        routes: List[SwapRoute] = []
        total_output = 0.0
        total_impact = 0.0

        pool_map = {pool.pool_id: pool for pool in pools}
        for pool_id, amount in allocations.items():
            if amount <= self.ALLOCATION_EPS:
                continue
            pool = pool_map[pool_id]
            output = pool.get_amount_out(amount, request.token_in)
            impact = pool.get_price_impact(amount, request.token_in)
            route = SwapRoute(
                path=[request.token_in, request.token_out],
                pools=[pool],
                dexes=[pool.dex],
                amount_in=amount,
                amount_out=output,
            )
            routes.append(route)
            total_output += output
            total_impact += impact * (amount / request.amount_in)

        end_time = time.perf_counter()

        base_gas = 150000
        gas_per_route = 100000
        gas_estimate = base_gas + max(0, len(routes) - 1) * gas_per_route

        return SwapResult(
            routes=routes,
            total_amount_in=request.amount_in,
            total_amount_out=total_output,
            total_price_impact=total_impact,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm_name=self.metadata.name,
            gas_estimate=gas_estimate,
        )

    def _solve_kkt_allocations(
        self, pools: List[Pool], token_in: str, total_amount: float
    ) -> Dict[str, float]:
        params: List[Tuple[Pool, float, float, float, float]] = []
        # (pool, reserve_in=x, reserve_out=y, alpha, c=y*alpha*x)
        for pool in pools:
            x, y = self._get_pool_reserves(pool, token_in)
            alpha = 1.0 - pool.fee
            if x <= 0 or y <= 0 or alpha <= 0:
                continue
            c = y * alpha * x
            params.append((pool, x, y, alpha, c))

        if not params:
            return {}

        if len(params) == 1:
            return {params[0][0].pool_id: total_amount}

        # d_i(0) = y_i * alpha_i / x_i
        max_initial_derivative = max(y * alpha / x for _, x, y, alpha, _ in params)
        if max_initial_derivative <= 0:
            return {}

        low = 1e-18
        high = max_initial_derivative

        for _ in range(self.BISECTION_ITERATIONS):
            mid = (low + high) / 2.0
            alloc_sum = self._allocation_sum_for_lambda(params, mid)
            if alloc_sum > total_amount:
                low = mid
            else:
                high = mid

        lambda_star = high
        raw_allocations: Dict[str, float] = {}
        alloc_total = 0.0

        for pool, x, _, alpha, c in params:
            amount = (math.sqrt(c / lambda_star) - x) / alpha
            amount = max(0.0, amount)
            if amount > self.ALLOCATION_EPS:
                raw_allocations[pool.pool_id] = amount
                alloc_total += amount

        if alloc_total <= 0:
            # Fallback: send all to best instantaneous pool.
            best_pool = max(params, key=lambda item: item[2] * item[3] / item[1])[0]
            return {best_pool.pool_id: total_amount}

        # Numerical cleanup to match the exact budget.
        scale = total_amount / alloc_total
        final_allocations = {
            pool_id: max(0.0, amount * scale) for pool_id, amount in raw_allocations.items()
        }
        return final_allocations

    def _allocation_sum_for_lambda(
        self, params: List[Tuple[Pool, float, float, float, float]], lambda_value: float
    ) -> float:
        total = 0.0
        for _, x, _, alpha, c in params:
            amount = (math.sqrt(c / lambda_value) - x) / alpha
            if amount > 0:
                total += amount
        return total

    def _get_pool_reserves(self, pool: Pool, token_in: str) -> Tuple[float, float]:
        if token_in == pool.token0:
            return pool.reserve0, pool.reserve1
        return pool.reserve1, pool.reserve0
