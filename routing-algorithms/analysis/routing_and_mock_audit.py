#!/usr/bin/env python3
"""
Audit script for:
1) routing algorithm logic behavior
2) mock data suitability for DEX aggregator experiments
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.base import Pool, SwapRequest
from algorithms.single.a_star import AStarRouting
from algorithms.single.dijkstra import DijkstraRouting
from algorithms.single.k_best import KBestRouting
from algorithms.single.naive import NaiveBruteForce
from mock_data.data_generator import DataGenerator


def _build_random_case(case_id: int, rng: random.Random) -> List[Pool]:
    tokens = ["A", "B", "C", "D", "E"]
    usd = {"A": 2000.0, "B": 1.0, "C": 40.0, "D": 0.5, "E": 120.0}
    dexes = ["x", "y", "z"]
    pools: List[Pool] = []

    for i, t0 in enumerate(tokens):
        for t1 in tokens[i + 1 :]:
            if rng.random() < 0.85:
                for dex in rng.sample(dexes, rng.randint(1, 2)):
                    liq = rng.uniform(2e5, 2e6)
                    skew = rng.uniform(0.90, 1.10)
                    reserve0 = (liq / 2.0) / usd[t0]
                    reserve1 = (liq / 2.0) / (usd[t1] * skew)
                    fee = rng.choice([0.0005, 0.002, 0.003])
                    pools.append(
                        Pool(
                            pool_id=f"case{case_id}_{dex}_{t0}_{t1}_{rng.randint(1, 99999)}",
                            dex=dex,
                            token0=t0,
                            token1=t1,
                            reserve0=reserve0,
                            reserve1=reserve1,
                            fee=fee,
                        )
                    )
    return pools


def _best_spot_score_exhaustive(
    pools: List[Pool], start: str, end: str, max_hops: int = 4
) -> Tuple[float, Optional[List[str]], Optional[List[Pool]]]:
    graph: Dict[str, List[Tuple[str, Pool]]] = {}
    for pool in pools:
        graph.setdefault(pool.token0, []).append((pool.token1, pool))
        graph.setdefault(pool.token1, []).append((pool.token0, pool))

    best = (float("inf"), None, None)

    def dfs(cur: str, path: List[str], used_pools: List[Pool], visited: set) -> None:
        nonlocal best
        if len(used_pools) > max_hops:
            return
        if cur == end and used_pools:
            score = 0.0
            for i, pool in enumerate(used_pools):
                rate = pool.get_spot_price(path[i])
                if rate <= 0:
                    return
                score += -math.log(rate)
            if score < best[0]:
                best = (score, path.copy(), used_pools.copy())
            return
        for nxt, pool in graph.get(cur, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            used_pools.append(pool)
            dfs(nxt, path, used_pools, visited)
            used_pools.pop()
            path.pop()
            visited.remove(nxt)

    dfs(start, [start], [], {start})
    return best


def audit_algorithm_logic(samples: int = 120, seed: int = 1234) -> Dict[str, float]:
    rng = random.Random(seed)
    mism_spot_dijkstra = 0
    mism_spot_astar = 0
    valid_spot = 0

    lower_amount_dijkstra = 0
    lower_amount_astar = 0
    lower_amount_kbest = 0
    valid_amount = 0

    for i in range(samples):
        pools = _build_random_case(i, rng)
        req = SwapRequest("A", "B", rng.uniform(10.0, 400.0))

        # Spot objective mismatch vs exhaustive
        best_score, _, _ = _best_spot_score_exhaustive(pools, "A", "B", max_hops=4)
        dijkstra = DijkstraRouting(pools)
        astar = AStarRouting(pools)
        path_d, pools_d = dijkstra._dijkstra_find_path("A", "B", req.amount_in)
        path_a, pools_a = astar._astar_find_path("A", "B", req.amount_in)

        if math.isfinite(best_score):
            valid_spot += 1
            if path_d is None:
                mism_spot_dijkstra += 1
            else:
                score_d = sum(-math.log(p.get_spot_price(path_d[j])) for j, p in enumerate(pools_d))
                if abs(score_d - best_score) > 1e-9:
                    mism_spot_dijkstra += 1
            if path_a is None:
                mism_spot_astar += 1
            else:
                score_a = sum(-math.log(p.get_spot_price(path_a[j])) for j, p in enumerate(pools_a))
                if abs(score_a - best_score) > 1e-9:
                    mism_spot_astar += 1

        # Actual amount_out mismatch vs Naive
        naive = NaiveBruteForce(pools).find_best_route(req)
        if naive.total_amount_out > 0:
            valid_amount += 1
            out_d = dijkstra.find_best_route(req).total_amount_out
            out_a = astar.find_best_route(req).total_amount_out
            out_k = KBestRouting(pools, k_paths=10).find_best_route(req).total_amount_out
            eps = 1e-9
            if out_d + eps < naive.total_amount_out:
                lower_amount_dijkstra += 1
            if out_a + eps < naive.total_amount_out:
                lower_amount_astar += 1
            if out_k + eps < naive.total_amount_out:
                lower_amount_kbest += 1

    return {
        "valid_spot_cases": valid_spot,
        "dijkstra_spot_mismatch_ratio": (mism_spot_dijkstra / valid_spot) if valid_spot else 0.0,
        "astar_spot_mismatch_ratio": (mism_spot_astar / valid_spot) if valid_spot else 0.0,
        "valid_amount_cases": valid_amount,
        "dijkstra_lower_than_naive_ratio": (lower_amount_dijkstra / valid_amount) if valid_amount else 0.0,
        "astar_lower_than_naive_ratio": (lower_amount_astar / valid_amount) if valid_amount else 0.0,
        "kbest_lower_than_naive_ratio": (lower_amount_kbest / valid_amount) if valid_amount else 0.0,
    }


def audit_mock_data_suitability(seed: int = 42) -> Dict[str, float]:
    pools = DataGenerator(seed=seed).generate_standard_test_pools()
    by_pair: Dict[Tuple[str, str], List[float]] = {}
    for pool in pools:
        pair = tuple(sorted([pool.token0, pool.token1]))
        rate = pool.get_spot_price(pool.token0) * (1.0 - pool.fee)
        by_pair.setdefault(pair, []).append(rate)

    spreads = []
    stable_spreads = []
    for pair, vals in by_pair.items():
        if len(vals) < 2:
            continue
        mn, mx = min(vals), max(vals)
        spread_pct = (mx - mn) / ((mx + mn) / 2.0) * 100.0 if (mx + mn) > 0 else 0.0
        spreads.append(spread_pct)
        if set(pair).issubset({"USDC", "USDT", "DAI"}):
            stable_spreads.append(spread_pct)

    eth_usdc = [p for p in pools if {p.token0, p.token1} == {"ETH", "USDC"}]
    liq_usd = []
    for pool in eth_usdc:
        if pool.token0 == "ETH":
            liq = 2000.0 * pool.reserve0 + pool.reserve1
        else:
            liq = 2000.0 * pool.reserve1 + pool.reserve0
        liq_usd.append(liq)

    liq_cv = 0.0
    if liq_usd:
        mean = statistics.mean(liq_usd)
        if mean > 0:
            liq_cv = statistics.pstdev(liq_usd) / mean

    return {
        "num_pools": float(len(pools)),
        "num_pairs": float(len(by_pair)),
        "mean_pair_spread_pct": statistics.mean(spreads) if spreads else 0.0,
        "max_pair_spread_pct": max(spreads) if spreads else 0.0,
        "mean_stable_spread_pct": statistics.mean(stable_spreads) if stable_spreads else 0.0,
        "max_stable_spread_pct": max(stable_spreads) if stable_spreads else 0.0,
        "eth_usdc_liquidity_cv": liq_cv,
    }


def main() -> None:
    algo = audit_algorithm_logic(samples=120, seed=1234)
    mock = audit_mock_data_suitability(seed=42)

    print("=== Algorithm Logic Audit ===")
    for k, v in algo.items():
        if "ratio" in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v}")

    print("")
    print("=== Mock Data Suitability Audit ===")
    for k, v in mock.items():
        if "pct" in k:
            print(f"{k}: {v:.6f}%")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

