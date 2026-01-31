#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Bellman-Ford 대비 효율 분석 벤치마크
=============================================================================
- DESIGN.md의 시나리오(S1~S12)와 조합(C1~C8)에 따라,
  Bellman-Ford를 기준으로 다른 알고리즘·조합의 "결과(비용)" 및 "실행 시간"을 측정.
- 효율 기준: (1) 실행 결과가 제일 좋은 것(경로 비용 최적), (2) 충분히 빠르게(실행 시간).
- 출력: CSV (시나리오 ID, 알고리즘, 비용 일치 여부, 경로 비용, 평균 시간, BF 대비 시간 비율).
- 실행: python3 benchmark_vs_bellman_ford.py
=============================================================================
"""

import io
import contextlib
import csv
import os
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Any

# -----------------------------------------------------------------------------
# 알고리즘 import (aggregation_pathfinding_examples 모듈 사용)
# -----------------------------------------------------------------------------
from aggregation_pathfinding_examples import (
    pathfinding_bfs,
    pathfinding_dijkstra,
    pathfinding_astar,
    pathfinding_bellman_ford,
    pathfinding_floyd_warshall,
    get_path_floyd,
    pathfinding_composite_bfs_with_distances,
    reconstruct_path,
)


# =============================================================================
# 1. 시나리오 정의 (DESIGN.md 3.2 시나리오 테이블)
# =============================================================================
# 각 시나리오: (scenario_id, V, E, weight_type)
# - V: 노드 수
# - E: 무방향 간선 수 (연결 그래프 보장을 위해 E >= V-1 가정)
# - weight_type: "all_positive" | "some_negative"

SCENARIOS: List[Tuple[str, int, int, str]] = [
    ("S1", 50, 150, "all_positive"),   # 소형, sparse, 양수만
    ("S2", 50, 400, "all_positive"),   # 소형, 중간 밀도, 양수만
    ("S3", 50, 1200, "all_positive"),  # 소형, dense, 양수만
    ("S4", 100, 300, "all_positive"),  # 중형, sparse, 양수만
    ("S5", 100, 800, "all_positive"),  # 중형, 중간 밀도, 양수만
    ("S6", 100, 2500, "all_positive"), # 중형, dense, 양수만
    ("S7", 200, 600, "all_positive"),  # 대형, sparse, 양수만
    ("S8", 200, 1600, "all_positive"), # 대형, 중간 밀도, 양수만
    ("S9", 200, 5000, "all_positive"), # 대형, dense, 양수만
    ("S10", 50, 150, "some_negative"),  # 소형, sparse, 일부 음수
    ("S11", 100, 300, "some_negative"), # 중형, sparse, 일부 음수
    ("S12", 200, 600, "some_negative"), # 대형, sparse, 일부 음수
]


# =============================================================================
# 2. 그래프 생성
# =============================================================================

def _build_connected_graph_with_edges(
    v: int,
    target_edges: int,
    weight_fn,
    seed: Optional[int] = None,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    노드 0..v-1, 무방향 간선이 정확히 target_edges개인 연결 그래프를 만든다.
    - 먼저 체인 0-1-2-...-(v-1)을 넣어 연결성 보장 (v-1개 간선).
    - 나머지 target_edges - (v-1)개 간선을 랜덤 (a,b)로 추가 (중복 제외).
    - weight_fn()은 매 간선마다 가중치를 반환하는 함수 (예: lambda: random.randint(1,10)).
    """
    if seed is not None:
        random.seed(seed)
    graph: Dict[str, List[Tuple[str, int]]] = {str(i): [] for i in range(v)}
    edge_set: Set[Tuple[int, int]] = set()

    # 1) 체인 0-1-2-...-(v-1) 필수 (연결성)
    for i in range(v - 1):
        a, b = i, i + 1
        w = weight_fn()
        graph[str(a)].append((str(b), w))
        graph[str(b)].append((str(a), w))
        edge_set.add((min(a, b), max(a, b)))
    need = target_edges - (v - 1)
    if need <= 0:
        return graph

    # 2) 나머지 간선 랜덤 추가 (무방향이므로 (min,max)로 한 번만 저장)
    attempts = 0
    max_attempts = need * 20
    while len(edge_set) < target_edges and attempts < max_attempts:
        a, b = random.randint(0, v - 1), random.randint(0, v - 1)
        if a == b:
            attempts += 1
            continue
        u, w_ = min(a, b), max(a, b)
        if (u, w_) in edge_set:
            attempts += 1
            continue
        weight = weight_fn()
        graph[str(a)].append((str(b), weight))
        graph[str(b)].append((str(a), weight))
        edge_set.add((u, w_))
        attempts = 0
    return graph


def _apply_some_negative(
    graph: Dict[str, List[Tuple[str, int]]],
    ratio: float = 0.15,
    neg_range: Tuple[int, int] = (-5, -1),
    seed: Optional[int] = None,
) -> None:
    """
    그래프의 간선 중 ratio 비율만큼을 음수 가중치로 바꾼다.
    원본 graph를 in-place 수정. 무방향이므로 (u,v)와 (v,u) 동일 가중치로 유지.
    """
    if seed is not None:
        random.seed(seed)
    seen: Set[Tuple[int, int]] = set()
    for u in graph:
        for idx, (v, w) in enumerate(graph[u]):
            ui, vi = int(u), int(v)
            key = (min(ui, vi), max(ui, vi))
            if key in seen:
                continue
            seen.add(key)
            if random.random() < ratio:
                new_w = random.randint(neg_range[0], neg_range[1])
                graph[u][idx] = (v, new_w)
                for j, (v2, _) in enumerate(graph[v]):
                    if v2 == u:
                        graph[v][j] = (u, new_w)
                        break


def build_graph_for_scenario(
    scenario_id: str,
    v: int,
    e: int,
    weight_type: str,
    seed: int = 42,
    max_negative_cycle_retries: int = 5,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    시나리오 ID에 맞는 그래프를 생성한다.
    - all_positive: 모든 간선 가중치 1~10.
    - some_negative: 먼저 1~10으로 생성 후 약 15% 간선을 -5~-1로 변경.
      음수 사이클이 있으면 Bellman-Ford가 None을 반환하므로, 그런 경우 재생성 시도(max_negative_cycle_retries).
    """
    def weight_positive():
        return random.randint(1, 10)

    graph = _build_connected_graph_with_edges(v, e, weight_positive, seed=seed)
    if weight_type == "all_positive":
        return graph

    # some_negative: 음수 적용 후 BF로 사이클 검사, 필요 시 재생성
    for attempt in range(max_negative_cycle_retries):
        g_copy: Dict[str, List[Tuple[str, int]]] = {
            k: [(n, w) for n, w in vv] for k, vv in graph.items()
        }
        _apply_some_negative(g_copy, ratio=0.15, seed=seed + attempt + 1)
        # BF로 음수 사이클 여부만 확인 (start=0, goal=str(v-1))
        with contextlib.redirect_stdout(io.StringIO()):
            path, cost = pathfinding_bellman_ford(g_copy, "0", str(v - 1))
        if path is not None and cost is not None:
            return g_copy
        # 음수 사이클 감지 시 재시도
    # 재시도 후에도 실패하면 양수만 그래프 반환 (해당 시나리오는 "일부 음수" 대신 양수만 사용)
    return graph


# =============================================================================
# 3. 보조 함수: BFS 도달 집합, 부분 그래프
# =============================================================================

def get_reachable_bfs(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
) -> Set[str]:
    """
    start에서 BFS로 도달 가능한 노드 집합을 반환한다.
    (가중치 무시, 간선만 따라 감.)
    """
    reachable: Set[str] = {start}
    q: deque = deque([start])
    while q:
        node = q.popleft()
        for neighbor, _ in graph.get(node, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                q.append(neighbor)
    return reachable


def induced_subgraph(
    graph: Dict[str, List[Tuple[str, int]]],
    nodes: Set[str],
) -> Dict[str, List[Tuple[str, int]]]:
    """
    nodes에 속한 노드만 남기고, 그들 사이의 간선만 포함한 부분 그래프를 반환한다.
    """
    sub = {n: [] for n in nodes}
    for n in nodes:
        for neighbor, w in graph.get(n, []):
            if neighbor in nodes:
                sub[n].append((neighbor, w))
    return sub


def bfs_hop_distance(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Optional[int]:
    """start에서 goal까지의 BFS 홉 수(간선 개수). 도달 불가면 None."""
    if start == goal:
        return 0
    dist = {start: 0}
    q: deque = deque([start])
    while q:
        node = q.popleft()
        d = dist[node]
        for neighbor, _ in graph.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = d + 1
                if neighbor == goal:
                    return d + 1
                q.append(neighbor)
    return None


def get_nodes_within_k_hops(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    k: int,
) -> Set[str]:
    """start에서 k홉 이내의 노드 집합."""
    within: Set[str] = {start}
    q: deque = deque([(start, 0)])
    while q:
        node, hops = q.popleft()
        if hops >= k:
            continue
        for neighbor, _ in graph.get(node, []):
            if neighbor not in within:
                within.add(neighbor)
                q.append((neighbor, hops + 1))
    return within


# =============================================================================
# 4. 조합 알고리즘 (C1, C2, C5, C6)
# =============================================================================

def composite_c1_bfs_then_dijkstra(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], float]:
    """
    C1: BFS로 start에서 도달 가능한 노드만 구한 뒤, 그 부분 그래프에서 Dijkstra.
    양수만 있을 때 최적 경로 비용과 동일. (음수 있으면 Dijkstra는 부적절.)
    반환: (path, cost). 경로 없으면 (None, inf).
    """
    reachable = get_reachable_bfs(graph, start)
    if goal not in reachable:
        return (None, float("inf"))
    sub = induced_subgraph(graph, reachable)
    with contextlib.redirect_stdout(io.StringIO()):
        path, cost = pathfinding_dijkstra(sub, start, goal)
    return (path, cost)


def composite_c2_bfs_then_bellman_ford(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], Optional[float]]:
    """
    C2: BFS로 도달 가능 노드만 구한 뒤, 부분 그래프에서 Bellman-Ford.
    BF와 동일한 최적 비용 (부분 그래프에 음수 사이클 없으면).
    """
    reachable = get_reachable_bfs(graph, start)
    if goal not in reachable:
        return (None, None)
    sub = induced_subgraph(graph, reachable)
    with contextlib.redirect_stdout(io.StringIO()):
        path, cost = pathfinding_bellman_ford(sub, start, goal)
    return (path, cost)


def _dfs_reachable(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
    visited: Set[str],
) -> bool:
    if start == goal:
        return True
    visited.add(start)
    for neighbor, _ in graph.get(start, []):
        if neighbor not in visited and _dfs_reachable(graph, neighbor, goal, visited):
            return True
    return False


def composite_c5_dfs_then_dijkstra(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], float]:
    """
    C5: DFS로 start->goal 도달 여부 확인 후, 도달하면 Dijkstra 실행.
    (연결 그래프에서는 항상 도달하므로, 실질적으로 Dijkstra만 실행.)
    """
    visited: Set[str] = set()
    if not _dfs_reachable(graph, start, goal, visited):
        return (None, float("inf"))
    with contextlib.redirect_stdout(io.StringIO()):
        path, cost = pathfinding_dijkstra(graph, start, goal)
    return (path, cost)


def composite_c6_bfs_khop_then_bellman_ford(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
    k_multiplier: float = 2.0,
) -> Tuple[Optional[List[str]], Optional[float]]:
    """
    C6: BFS로 start->goal 홉 거리 d를 구한 뒤, start에서 k = d * k_multiplier 홉 이내 노드만 부분 그래프로 하고 Bellman-Ford.
    k가 충분히 크면 최적 경로가 부분 그래프에 포함되므로 BF와 동일 결과.
    """
    d = bfs_hop_distance(graph, start, goal)
    if d is None:
        return (None, None)
    k = max(int(d * k_multiplier), d + 1)
    within = get_nodes_within_k_hops(graph, start, k)
    if goal not in within:
        return (None, None)
    sub = induced_subgraph(graph, within)
    with contextlib.redirect_stdout(io.StringIO()):
        path, cost = pathfinding_bellman_ford(sub, start, goal)
    return (path, cost)


# =============================================================================
# 5. 실행 시간 측정 (stdout 억제)
# =============================================================================

def measure_time(fn, *args, **kwargs) -> float:
    """함수 한 번 실행 시간(초). stdout 억제."""
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
    return t1 - t0


def measure_avg_time(fn, n_runs: int, *args, **kwargs) -> float:
    """n_runs회 실행 후 평균 시간(초)."""
    times = [measure_time(fn, *args, **kwargs) for _ in range(n_runs)]
    return sum(times) / len(times)


# =============================================================================
# 6. 벤치마크 실행
# =============================================================================

def run_one_algorithm_once(
    algo_name: str,
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Tuple[Optional[float], Optional[List[str]]]:
    """
    단일 알고리즘을 한 번 실행해 (경로 비용, 경로)를 반환.
    BFS는 비용 없음 → (None, path). Floyd-Warshall은 (dist, path).
    """
    cost: Optional[float] = None
    path: Any = None
    if algo_name == "Bellman-Ford":
        with contextlib.redirect_stdout(io.StringIO()):
            path, cost = pathfinding_bellman_ford(graph, start, goal)
    elif algo_name == "Dijkstra":
        with contextlib.redirect_stdout(io.StringIO()):
            path, cost = pathfinding_dijkstra(graph, start, goal)
    elif algo_name == "A*":
        with contextlib.redirect_stdout(io.StringIO()):
            path, cost = pathfinding_astar(graph, start, goal)
    elif algo_name == "BFS":
        with contextlib.redirect_stdout(io.StringIO()):
            path = pathfinding_bfs(graph, start, goal)
        cost = None
    elif algo_name == "C1_BFS_then_Dijkstra":
        path, cost = composite_c1_bfs_then_dijkstra(graph, start, goal)
    elif algo_name == "C2_BFS_then_BellmanFord":
        path, cost = composite_c2_bfs_then_bellman_ford(graph, start, goal)
    elif algo_name == "C5_DFS_then_Dijkstra":
        path, cost = composite_c5_dfs_then_dijkstra(graph, start, goal)
    elif algo_name == "C6_BFS_khop_then_BellmanFord":
        path, cost = composite_c6_bfs_khop_then_bellman_ford(graph, start, goal)
    elif algo_name == "C8_FloydWarshall":
        with contextlib.redirect_stdout(io.StringIO()):
            dist_map, next_map = pathfinding_floyd_warshall(graph)
        if (start, goal) in dist_map and dist_map[(start, goal)] != float("inf"):
            cost = dist_map[(start, goal)]
            path = get_path_floyd(next_map, start, goal)
        else:
            cost = None
            path = None
    else:
        return (None, None)
    return (cost, path)


def measure_algo_time(algo_name: str, graph: Dict[str, List[Tuple[str, int]]], start: str, goal: str, n_runs: int) -> float:
    """
    알고리즘을 n_runs회 실행한 평균 시간(초).
    stdout 억제 후 실행. C8은 Floyd-Warshall 전체 실행 시간만 측정 (경로 조회 제외).
    """
    def runner():
        with contextlib.redirect_stdout(io.StringIO()):
            if algo_name == "Bellman-Ford":
                pathfinding_bellman_ford(graph, start, goal)
            elif algo_name == "Dijkstra":
                pathfinding_dijkstra(graph, start, goal)
            elif algo_name == "A*":
                pathfinding_astar(graph, start, goal)
            elif algo_name == "BFS":
                pathfinding_bfs(graph, start, goal)
            elif algo_name == "C1_BFS_then_Dijkstra":
                composite_c1_bfs_then_dijkstra(graph, start, goal)
            elif algo_name == "C2_BFS_then_BellmanFord":
                composite_c2_bfs_then_bellman_ford(graph, start, goal)
            elif algo_name == "C5_DFS_then_Dijkstra":
                composite_c5_dfs_then_dijkstra(graph, start, goal)
            elif algo_name == "C6_BFS_khop_then_BellmanFord":
                composite_c6_bfs_khop_then_bellman_ford(graph, start, goal)
            elif algo_name == "C8_FloydWarshall":
                pathfinding_floyd_warshall(graph)
    return measure_avg_time(runner, n_runs)


def run_benchmark(
    n_runs: int = 3,
    scenario_ids: Optional[List[str]] = None,
    algo_list: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    시나리오별·알고리즘별로 벤치마크를 돌리고, 각 행(시나리오, 알고리즘)에 대해
    비용, 비용 일치 여부, 평균 시간, BF 대비 시간 비율을 기록한 딕셔너리 리스트를 반환.
    """
    if scenario_ids is None:
        scenario_ids = [s[0] for s in SCENARIOS]
    if algo_list is None:
        algo_list = [
            "Bellman-Ford",
            "Dijkstra",
            "A*",
            "BFS",
            "C1_BFS_then_Dijkstra",
            "C2_BFS_then_BellmanFord",
            "C5_DFS_then_Dijkstra",
            "C6_BFS_khop_then_BellmanFord",
            "C8_FloydWarshall",
        ]

    rows: List[Dict[str, Any]] = []
    for scenario_id, v, e, weight_type in SCENARIOS:
        if scenario_id not in scenario_ids:
            continue
        graph = build_graph_for_scenario(scenario_id, v, e, weight_type)
        start, goal = "0", str(v - 1)

        # Bellman-Ford 한 번 실행해 기준 비용 확보 (비용 일치 비교용)
        bf_cost, _ = run_one_algorithm_once("Bellman-Ford", graph, start, goal)
        if bf_cost is None:
            bf_cost_ref = None  # 음수 사이클 등
        else:
            bf_cost_ref = bf_cost

        # Bellman-Ford 평균 실행 시간 (ratio 분모) — 한 번만 측정
        bf_time = measure_algo_time("Bellman-Ford", graph, start, goal, n_runs)

        for algo_name in algo_list:
            cost, path = run_one_algorithm_once(algo_name, graph, start, goal)
            # 비용 일치: BF 비용이 있고, 이 알고리즘 비용이 있고, 같으면 True (부동소수 오차 허용)
            cost_match = False
            if bf_cost_ref is not None and cost is not None and path is not None:
                cost_match = abs(bf_cost_ref - cost) < 1e-9
            # BF는 이미 bf_time으로 측정했으므로 재측정 생략
            if algo_name == "Bellman-Ford":
                avg_time = bf_time
            else:
                avg_time = measure_algo_time(algo_name, graph, start, goal, n_runs)
            # BF 대비 시간 비율: 1보다 크면 이 알고리즘이 BF보다 빠름
            ratio = (bf_time / avg_time) if (avg_time and avg_time > 0) else None
            rows.append({
                "scenario_id": scenario_id,
                "V": v,
                "E": e,
                "weight_type": weight_type,
                "algorithm": algo_name,
                "cost": cost,
                "cost_match": cost_match,
                "mean_time_sec": round(avg_time, 6),
                "ratio_vs_BF": round(ratio, 4) if ratio is not None else None,
            })
    return rows


# =============================================================================
# 7. CSV 저장 및 main
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")


def save_results_csv(rows: List[Dict[str, Any]], path: Optional[str] = None) -> str:
    """벤치마크 결과를 CSV로 저장. path 없으면 data/benchmark_vs_bellman_ford.csv"""
    if path is None:
        path = os.path.join(DATA_DIR, "benchmark_vs_bellman_ford.csv")
    if not rows:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["scenario_id", "V", "E", "weight_type", "algorithm", "cost", "cost_match", "mean_time_sec", "ratio_vs_BF"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    print("Bellman-Ford 대비 효율 분석 벤치마크 시작 (시나리오 S1~S12, N=3 runs)...")
    rows = run_benchmark(n_runs=3)
    out = save_results_csv(rows)
    print(f"결과 저장: {out}")
    print("완료. 결과를 토대로 RESULTS.md 등 문서를 추가 작성하면 됩니다.")


if __name__ == "__main__":
    main()
