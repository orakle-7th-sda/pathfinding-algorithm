#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
패스파인딩 알고리즘 비교 차트
=============================================================================
- Baseline 1: BFS (가중치 무시, 간선 개수 기준)
- Baseline 2: Naive (모든 경로 열거 후 최소 비용 선택, 지수 시간)
- 비교: Dijkstra, A*, Bellman-Ford
- 지표: 같은 그래프에서 실행 시간 (그래프 크기 증가에 따른 변화)
- 실행: python3 chart_pathfinding.py
=============================================================================
"""

import time
import random
import io
import contextlib
import csv
import os
from typing import Dict, List, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
CHART_DIR = os.path.join(ROOT, "chart")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 예제 모듈의 알고리즘 사용 (출력 억제 후 시간 측정)
from aggregation_pathfinding_examples import (
    pathfinding_bfs,
    pathfinding_dijkstra,
    pathfinding_astar,
    pathfinding_bellman_ford,
)


# =============================================================================
# Baseline 2: Naive — 모든 단순 경로 열거 후 최소 비용 경로 반환 (지수 시간)
# =============================================================================


def pathfinding_naive(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], float]:
    """
    모든 단순 경로를 DFS로 열거한 뒤, 비용 합이 최소인 경로를 반환.
    시간복잡도: O((V-1)! ) 수준 — 그래프가 커지면 극단적으로 느려짐.
    """
    if start == goal:
        return ([start], 0.0)
    all_paths: List[Tuple[List[str], float]] = []

    def dfs(node: str, path: List[str], cost: float, visited: set) -> None:
        if node == goal:
            all_paths.append((path[:], cost))
            return
        for neighbor, w in graph.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            dfs(neighbor, path, cost + w, visited)
            path.pop()
            visited.discard(neighbor)

    visited = {start}
    dfs(start, [start], 0.0, visited)
    if not all_paths:
        return (None, float("inf"))
    best = min(all_paths, key=lambda x: x[1])
    return (best[0], best[1])


# =============================================================================
# 그래프 생성: 노드 수 V에 따른 연결 그래프 (간선 가중치 1~10)
# =============================================================================


def build_random_graph(
    v: int, seed: Optional[int] = None
) -> Dict[str, List[Tuple[str, int]]]:
    """
    노드 0..v-1, 체인 0-1-2-...-(v-1) 보장 + 추가 간선으로 여러 경로 생성.
    """
    if seed is not None:
        random.seed(seed)
    graph = {str(i): [] for i in range(v)}
    for i in range(v - 1):
        w = random.randint(1, 10)
        graph[str(i)].append((str(i + 1), w))
        graph[str(i + 1)].append((str(i), w))
    # 추가 간선: 약 v개 더 넣어서 경로 다양화
    added = 0
    for _ in range(v * 2):
        a, b = random.randint(0, v - 1), random.randint(0, v - 1)
        if a == b or abs(a - b) == 1:
            continue
        w = random.randint(1, 10)
        if (str(b), w) not in graph[str(a)]:
            graph[str(a)].append((str(b), w))
            graph[str(b)].append((str(a), w))
            added += 1
            if added >= v:
                break
    return graph


# =============================================================================
# 실행 시간 측정 (stdout 억제)
# =============================================================================


def measure_time(fn, *args, **kwargs):
    """함수 한 번 실행 시간(초). stdout 억제."""
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
    return t1 - t0


def measure_avg_time(fn, n_runs: int, *args, **kwargs) -> float:
    """n_runs회 실행 후 평균 시간(초)."""
    times = []
    for _ in range(n_runs):
        times.append(measure_time(fn, *args, **kwargs))
    return sum(times) / len(times)


# =============================================================================
# 벤치마크 실행 및 데이터 수집
# =============================================================================


def run_benchmark(
    node_counts: List[int],
    n_runs: int = 3,
    naive_max_nodes: int = 10,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    node_counts에 있는 각 V에 대해 그래프 생성 후,
    BFS, Naive, Dijkstra, A*, Bellman-Ford 실행 시간 수집.
    Naive는 naive_max_nodes 초과 시 스킵(너무 느림).
    """
    results: Dict[str, List[float]] = {
        "BFS": [],
        "Naive": [],
        "Dijkstra": [],
        "A*": [],
        "Bellman-Ford": [],
    }
    for v in node_counts:
        graph = build_random_graph(v, seed=42)
        start, goal = "0", str(v - 1)
        # BFS
        t_bfs = measure_avg_time(pathfinding_bfs, n_runs, graph, start, goal)
        results["BFS"].append(t_bfs)
        # Naive: V가 크면 스킵
        if v <= naive_max_nodes:
            t_naive = measure_avg_time(
                pathfinding_naive, max(1, n_runs - 1), graph, start, goal
            )
            results["Naive"].append(t_naive)
        else:
            results["Naive"].append(float("nan"))
        # Dijkstra
        t_dij = measure_avg_time(pathfinding_dijkstra, n_runs, graph, start, goal)
        results["Dijkstra"].append(t_dij)
        # A*
        t_astar = measure_avg_time(pathfinding_astar, n_runs, graph, start, goal)
        results["A*"].append(t_astar)
        # Bellman-Ford
        t_bf = measure_avg_time(pathfinding_bellman_ford, n_runs, graph, start, goal)
        results["Bellman-Ford"].append(t_bf)
    return (node_counts, results)


# =============================================================================
# 차트 그리기
# =============================================================================


def plot_pathfinding(
    node_counts: List[int], results: Dict[str, List[float]], out_path: str
):
    """x=노드 수, y=실행 시간(초), 알고리즘별 선 그래프. matplotlib 없으면 CSV만 저장."""
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        for name, times in results.items():
            ys = [t if t == t else None for t in times]  # nan -> None so line breaks
            plt.plot(node_counts, times, marker="o", label=name)
        plt.xlabel("Number of nodes (V)")
        plt.ylabel("Time (sec)")
        plt.title("Pathfinding: Baseline (BFS, Naive) vs Dijkstra / A* / Bellman-Ford")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"차트 저장: {out_path}")
    csv_path = os.path.join(DATA_DIR, "chart_pathfinding.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nodes"] + list(results.keys()))
        for i, v in enumerate(node_counts):
            row = [v] + [results[k][i] if k in results else "" for k in results]
            w.writerow(row)
    print(f"데이터 저장: {csv_path}")


# =============================================================================
# 메인
# =============================================================================


def main():
    if not HAS_MATPLOTLIB:
        print(
            "matplotlib 없음. 차트 대신 CSV만 저장합니다. PNG 차트를 보려면: pip install matplotlib"
        )
    # Naive가 너무 느려지기 전까지: 5~10 또는 5~11
    node_counts = [5, 6, 7, 8, 9, 10]
    print("패스파인딩 벤치마크 실행 중 (노드 수:", node_counts, ")...")
    counts, results = run_benchmark(node_counts, n_runs=3, naive_max_nodes=10)
    out = os.path.join(CHART_DIR, "chart_pathfinding.png")
    os.makedirs(CHART_DIR, exist_ok=True)
    plot_pathfinding(counts, results, out)
    print("완료.")


if __name__ == "__main__":
    main()
