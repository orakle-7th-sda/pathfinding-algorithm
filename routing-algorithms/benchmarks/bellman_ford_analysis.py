"""
Bellman-Ford baseline efficiency analysis.

This is a compact, dependency-free benchmark used by `main.py --bellman-ford`.
"""

from __future__ import annotations

import heapq
import random
import statistics
import time
from typing import Dict, List, Optional, Tuple

Graph = Dict[int, List[Tuple[int, int]]]


def build_random_graph(
    num_nodes: int, num_edges: int, weight_min: int = 1, weight_max: int = 10, seed: int = 42
) -> Graph:
    """
    Build a connected directed graph with positive weights.
    """
    if num_nodes <= 1:
        return {0: []}

    rng = random.Random(seed + num_nodes + num_edges)
    graph: Graph = {i: [] for i in range(num_nodes)}
    used_edges = set()

    # Force connectivity with a chain 0->1->...->N-1.
    for i in range(num_nodes - 1):
        w = rng.randint(weight_min, weight_max)
        graph[i].append((i + 1, w))
        used_edges.add((i, i + 1))

    max_possible = num_nodes * (num_nodes - 1)
    target_edges = min(max_possible, max(num_edges, num_nodes - 1))

    while len(used_edges) < target_edges:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v or (u, v) in used_edges:
            continue
        w = rng.randint(weight_min, weight_max)
        graph[u].append((v, w))
        used_edges.add((u, v))

    return graph


def dijkstra(graph: Graph, start: int, goal: int) -> Optional[int]:
    dist = {start: 0}
    pq: List[Tuple[int, int]] = [(0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            return d
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return None


def astar_zero_heuristic(graph: Graph, start: int, goal: int) -> Optional[int]:
    # A* with h(n)=0 is equivalent to Dijkstra.
    return dijkstra(graph, start, goal)


def bellman_ford(graph: Graph, start: int, goal: int) -> Optional[int]:
    nodes = list(graph.keys())
    edges: List[Tuple[int, int, int]] = []
    for u, neighbors in graph.items():
        for v, w in neighbors:
            edges.append((u, v, w))

    dist = {node: float("inf") for node in nodes}
    dist[start] = 0

    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Negative-cycle check omitted because this benchmark uses positive weights.
    if dist[goal] == float("inf"):
        return None
    return int(dist[goal])


def benchmark(
    fn, graph: Graph, start: int, goal: int, runs: int = 5
) -> Tuple[Optional[int], float]:
    """
    Return (cost, mean_time_sec) across runs.
    """
    costs = []
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cost = fn(graph, start, goal)
        t1 = time.perf_counter()
        costs.append(cost)
        timings.append(t1 - t0)

    # For deterministic graph/algorithm, all costs should match.
    return costs[0], statistics.mean(timings)


def main() -> None:
    scenarios = [
        ("S1", 50, 150),
        ("S2", 100, 800),
        ("S3", 200, 1600),
    ]

    print("\nBellman-Ford Baseline Analysis (Positive-weight scenarios)")
    print(f"{'Scenario':<8} {'V':>5} {'E':>6} {'Cost':>8} {'BF(s)':>12} {'Dij(s)':>12} {'A*(s)':>12} {'BF/Dij':>10}")
    print("-" * 84)

    for sid, v, e in scenarios:
        graph = build_random_graph(v, e)
        start, goal = 0, v - 1

        bf_cost, bf_time = benchmark(bellman_ford, graph, start, goal)
        dij_cost, dij_time = benchmark(dijkstra, graph, start, goal)
        astar_cost, astar_time = benchmark(astar_zero_heuristic, graph, start, goal)

        # Safety guard to make cost mismatches visible.
        if bf_cost != dij_cost or bf_cost != astar_cost:
            cost_display = f"{bf_cost}*"
        else:
            cost_display = str(bf_cost)

        ratio = (bf_time / dij_time) if dij_time > 0 else 0.0
        print(
            f"{sid:<8} {v:>5} {e:>6} {cost_display:>8} "
            f"{bf_time:>12.6f} {dij_time:>12.6f} {astar_time:>12.6f} {ratio:>10.2f}"
        )

    print("\n* Cost mismatch marker. No marker means same shortest-path cost as Bellman-Ford.")


if __name__ == "__main__":
    main()
