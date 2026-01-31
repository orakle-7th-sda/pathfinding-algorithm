#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
어그리게이팅(Aggregation) & 패스파인딩(Pathfinding) 알고리즘 예제 모음
=============================================================================
- 각 알고리즘: 이름(한/영), 난이도, 설명, 실행 가능한 코드
- 단일 알고리즘 + 복합(여러 단계/조합) 예제 포함
=============================================================================
"""

from collections import deque, defaultdict
import heapq
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 공통: 알고리즘 메타데이터용 데이터클래스
# =============================================================================


class Difficulty(Enum):
    """난이도: 1=쉬움, 2=보통, 3=어려움"""

    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class AlgoInfo:
    """알고리즘 이름·난이도·설명"""

    name_ko: str
    name_en: str
    difficulty: Difficulty
    description: str


def print_algo_header(info: AlgoInfo, category: str):
    """알고리즘 헤더 출력"""
    diff_str = {
        Difficulty.EASY: "쉬움",
        Difficulty.MEDIUM: "보통",
        Difficulty.HARD: "어려움",
    }[info.difficulty]
    print(f"\n{'='*60}")
    print(f"[{category}] {info.name_ko} ({info.name_en})")
    print(f"  난이도: {diff_str} | {info.description}")
    print("=" * 60)


# =============================================================================
# PART 1: 어그리게이팅 (Aggregation) 알고리즘
# =============================================================================


# -----------------------------------------------------------------------------
# 1-1. 단일 값 어그리게이션: 합계 (Sum)
# -----------------------------------------------------------------------------
def aggregation_sum(data: List[float]) -> float:
    """
    이름: 합계 (Sum)
    난이도: 쉬움
    설명: 리스트 전체 원소의 합을 구함. O(n).
    """
    info = AlgoInfo("합계", "Sum", Difficulty.EASY, "전체 원소의 합")
    print_algo_header(info, "어그리게이션")
    total = 0.0
    for x in data:
        total += x
    return total


# -----------------------------------------------------------------------------
# 1-2. 단일 값 어그리게이션: 평균 (Average / Mean)
# -----------------------------------------------------------------------------
def aggregation_average(data: List[float]) -> float:
    """
    이름: 평균 (Average)
    난이도: 쉬움
    설명: 합계 / 개수. 빈 리스트면 0 반환.
    """
    info = AlgoInfo("평균", "Average", Difficulty.EASY, "산술 평균")
    print_algo_header(info, "어그리게이션")
    if not data:
        return 0.0
    return sum(data) / len(data)


# -----------------------------------------------------------------------------
# 1-3. 단일 값 어그리게이션: 최솟값/최댓값 (Min / Max)
# -----------------------------------------------------------------------------
def aggregation_min_max(data: List[float]) -> Tuple[float, float]:
    """
    이름: 최솟값·최댓값 (Min & Max)
    난이도: 쉬움
    설명: 한 번의 순회로 min, max 동시 계산. O(n).
    """
    info = AlgoInfo(
        "최솟값·최댓값", "Min & Max", Difficulty.EASY, "한 번 순회로 둘 다 계산"
    )
    print_algo_header(info, "어그리게이션")
    if not data:
        return (0.0, 0.0)
    min_val = max_val = data[0]
    for x in data[1:]:
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    return (min_val, max_val)


# -----------------------------------------------------------------------------
# 1-4. 카운트 어그리게이션 (Count)
# -----------------------------------------------------------------------------
def aggregation_count(data: List[Any], predicate=None) -> int:
    """
    이름: 카운트 (Count)
    난이도: 쉬움
    설명: 조건(predicate)을 만족하는 원소 개수. 조건 없으면 전체 개수.
    """
    info = AlgoInfo("카운트", "Count", Difficulty.EASY, "조건 만족 개수")
    print_algo_header(info, "어그리게이션")
    if predicate is None:
        return len(data)
    return sum(1 for x in data if predicate(x))


# -----------------------------------------------------------------------------
# 1-5. 그룹별 집계 (Group By + Aggregate)
# -----------------------------------------------------------------------------
def aggregation_group_by_sum(records: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    이름: 그룹별 합계 (Group By Sum)
    난이도: 보통
    설명: 키(그룹)별로 값들을 묶어 합계. SQL GROUP BY + SUM과 유사.
    """
    info = AlgoInfo("그룹별 합계", "Group By Sum", Difficulty.MEDIUM, "키별 합계")
    print_algo_header(info, "어그리게이션")
    result = defaultdict(float)
    for key, value in records:
        result[key] += value
    return dict(result)


# -----------------------------------------------------------------------------
# 1-6. 복합 어그리게이션: 그룹별 평균 + 최대 그룹 찾기
# -----------------------------------------------------------------------------
def aggregation_composite_group_avg_and_top(
    records: List[Tuple[str, float]],
) -> Tuple[Dict[str, float], str]:
    """
    이름: 그룹별 평균 + 최대 평균 그룹 (복합)
    난이도: 보통
    설명: 그룹별 평균을 구한 뒤, 평균이 가장 큰 그룹 키를 반환.
    """
    info = AlgoInfo(
        "그룹별 평균 & 최대 평균 그룹",
        "Group By Avg + Top Group",
        Difficulty.MEDIUM,
        "그룹별 평균 계산 후 최대 평균 그룹 선택",
    )
    print_algo_header(info, "어그리게이션(복합)")
    # 1단계: 그룹별 합계 & 개수
    sum_by_key = defaultdict(float)
    count_by_key = defaultdict(int)
    for key, value in records:
        sum_by_key[key] += value
        count_by_key[key] += 1
    # 2단계: 그룹별 평균
    avg_by_key = {k: sum_by_key[k] / count_by_key[k] for k in sum_by_key}
    # 3단계: 평균 최대인 그룹
    top_group = max(avg_by_key, key=avg_by_key.get) if avg_by_key else ""
    return (dict(avg_by_key), top_group)


# -----------------------------------------------------------------------------
# 1-7. 누적 집계 (Running / Cumulative Aggregate)
# -----------------------------------------------------------------------------
def aggregation_running_sum(data: List[float]) -> List[float]:
    """
    이름: 누적 합 (Running Sum)
    난이도: 쉬움
    설명: 각 위치까지의 부분 합 리스트. [a,b,c] -> [a, a+b, a+b+c]
    """
    info = AlgoInfo("누적 합", "Running Sum", Difficulty.EASY, "부분 합 리스트")
    print_algo_header(info, "어그리게이션")
    result = []
    acc = 0.0
    for x in data:
        acc += x
        result.append(acc)
    return result


# -----------------------------------------------------------------------------
# 1-8. 분산 (Variance) - 통계 어그리게이션
# -----------------------------------------------------------------------------
def aggregation_variance(data: List[float]) -> float:
    """
    이름: 분산 (Variance)
    난이도: 보통
    설명: 평균과의 편차 제곱의 평균. 두 번 순회(평균 -> 분산).
    """
    info = AlgoInfo("분산", "Variance", Difficulty.MEDIUM, "편차 제곱의 평균")
    print_algo_header(info, "어그리게이션")
    if not data or len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    sq_diff = sum((x - mean) ** 2 for x in data)
    return sq_diff / (len(data) - 1)  # 표본 분산


# =============================================================================
# PART 2: 패스파인딩 (Pathfinding) 알고리즘
# =============================================================================

# 그래프 표현: 인접 리스트 (무방향/방향 모두 사용)
# graph[v] = [(neighbor, weight), ...]


def build_sample_graph() -> Dict[str, List[Tuple[str, int]]]:
    """예제용 가중 그래프 (노드: 문자열, 간선 가중치: 양수)"""
    return {
        "A": [("B", 1), ("C", 4)],
        "B": [("A", 1), ("C", 2), ("D", 5)],
        "C": [("A", 4), ("B", 2), ("D", 1)],
        "D": [("B", 5), ("C", 1), ("E", 3)],
        "E": [("D", 3)],
    }


# -----------------------------------------------------------------------------
# 2-1. BFS (Breadth-First Search) - 최단 경로 (간선 가중치 없음/동일)
# -----------------------------------------------------------------------------
def pathfinding_bfs(
    graph: Dict[str, List[Tuple[str, int]]], start: str, goal: str
) -> Optional[List[str]]:
    """
    이름: 너비 우선 탐색 (BFS)
    난이도: 쉬움
    설명: 가중치 없이 '간선 개수' 기준 최단 경로. 큐 + 방문 체크.
    """
    info = AlgoInfo("너비 우선 탐색", "BFS", Difficulty.EASY, "간선 수 기준 최단 경로")
    print_algo_header(info, "패스파인딩")
    if start == goal:
        return [start]
    # (현재 노드, 경로) 큐에 넣기
    q = deque([(start, [start])])
    visited = {start}
    while q:
        node, path = q.popleft()
        for neighbor, _ in graph.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            new_path = path + [neighbor]
            if neighbor == goal:
                return new_path
            q.append((neighbor, new_path))
    return None


# -----------------------------------------------------------------------------
# 2-2. DFS (Depth-First Search) - 경로 존재 여부 / 임의 경로
# -----------------------------------------------------------------------------
def pathfinding_dfs(
    graph: Dict[str, List[Tuple[str, int]]], start: str, goal: str
) -> Optional[List[str]]:
    """
    이름: 깊이 우선 탐색 (DFS)
    난이도: 쉬움
    설명: 스택 사용. 최단 보장은 아니지만 경로 하나 찾기. 백트래킹.
    """
    info = AlgoInfo(
        "깊이 우선 탐색", "DFS", Difficulty.EASY, "경로 존재 시 한 경로 반환"
    )
    print_algo_header(info, "패스파인딩")
    visited = set()
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor, _ in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None


# -----------------------------------------------------------------------------
# 2-3. 다익스트라 (Dijkstra) - 가중 그래프 단일 출발 최단 경로
# -----------------------------------------------------------------------------
def pathfinding_dijkstra(
    graph: Dict[str, List[Tuple[str, int]]], start: str, goal: str
) -> Tuple[Optional[List[str]], float]:
    """
    이름: 다익스트라 (Dijkstra)
    난이도: 보통
    설명: 음수 간선 없을 때 단일 출발 최단 경로. 우선순위 큐 사용. O((V+E) log V).
    """
    info = AlgoInfo(
        "다익스트라", "Dijkstra", Difficulty.MEDIUM, "양수 가중치 단일 출발 최단 경로"
    )
    print_algo_header(info, "패스파인딩")
    # (거리, 노드) 최소 힙
    pq = [(0, start)]
    dist = {start: 0}
    prev = {start: None}
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        if node == goal:
            break
        for neighbor, w in graph.get(node, []):
            nd = d + w
            if nd < dist.get(neighbor, float("inf")):
                dist[neighbor] = nd
                prev[neighbor] = node
                heapq.heappush(pq, (nd, neighbor))
    if goal not in prev:
        return (None, float("inf"))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return (path, dist[goal])


# -----------------------------------------------------------------------------
# 2-4. A* (A-Star) - 휴리스틱이 있는 최단 경로
# -----------------------------------------------------------------------------
def pathfinding_astar(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
    heuristic: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[List[str]], float]:
    """
    이름: A* (A-Star)
    난이도: 어려움
    설명: 휴리스틱 h(n)으로 탐색 범위 축소. h가 admissible면 최적 경로 보장.
    """
    info = AlgoInfo("A*", "A-Star", Difficulty.HARD, "휴리스틱 기반 최단 경로")
    print_algo_header(info, "패스파인딩")
    if heuristic is None:
        heuristic = {n: 0 for n in graph}
    # f = g + h, (f, g, node) 넣기
    pq = [(0 + heuristic.get(start, 0), 0, start)]
    g_score = {start: 0}
    prev = {start: None}
    while pq:
        f, g, node = heapq.heappop(pq)
        if g > g_score.get(node, float("inf")):
            continue
        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return (path, g_score[goal])
        for neighbor, w in graph.get(node, []):
            ng = g + w
            if ng < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = ng
                prev[neighbor] = node
                h = heuristic.get(neighbor, 0)
                heapq.heappush(pq, (ng + h, ng, neighbor))
    return (None, float("inf"))


# -----------------------------------------------------------------------------
# 2-5. 벨만-포드 (Bellman-Ford) - 음수 간선 허용 단일 출발
# -----------------------------------------------------------------------------
def pathfinding_bellman_ford(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], Optional[float]]:
    """
    이름: 벨만-포드 (Bellman-Ford)
    난이도: 어려움
    설명: 음수 간선 허용. V-1번 완화. 음수 사이클 있으면 None 반환 가능.
    """
    info = AlgoInfo(
        "벨만-포드", "Bellman-Ford", Difficulty.HARD, "음수 간선 허용 단일 출발"
    )
    print_algo_header(info, "패스파인딩")
    nodes = set(graph) | {n for adj in graph.values() for n, _ in adj}
    dist = {n: float("inf") for n in nodes}
    dist[start] = 0
    prev = {n: None for n in nodes}
    edges = [(u, v, w) for u in graph for v, w in graph[u]]
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
    # 음수 사이클 검사 (목표까지 경로에 사이클이 있으면 무한히 감소 가능)
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            print("  [경고] 음수 사이클 감지.")
            return (None, None)
    if dist[goal] == float("inf"):
        return (None, None)
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return (path, dist[goal])


# -----------------------------------------------------------------------------
# 2-6. 플로이드-워셜 (Floyd-Warshall) - 전체 쌍 최단 경로
# -----------------------------------------------------------------------------
def pathfinding_floyd_warshall(
    graph: Dict[str, List[Tuple[str, int]]],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], Optional[str]]]:
    """
    이름: 플로이드-워셜 (Floyd-Warshall)
    난이도: 어려움
    설명: 모든 쌍 최단 거리. O(V^3). 음수 간선 허용(음수 사이클 없을 때).
    """
    info = AlgoInfo(
        "플로이드-워셜", "Floyd-Warshall", Difficulty.HARD, "전체 쌍 최단 경로"
    )
    print_algo_header(info, "패스파인딩")
    nodes = sorted(set(graph) | {n for adj in graph.values() for n, _ in adj})
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u in graph:
        for v, w in graph[u]:
            i, j = idx[u], idx[v]
            dist[i][j] = min(dist[i][j], w)
            next_node[i][j] = v
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    mid = next_node[k][j] if next_node[k][j] is not None else nodes[j]
                    next_node[i][j] = next_node[i][k] or mid
    dist_map = {(nodes[i], nodes[j]): dist[i][j] for i in range(n) for j in range(n)}
    next_map = {
        (nodes[i], nodes[j]): next_node[i][j] for i in range(n) for j in range(n)
    }
    return (dist_map, next_map)


def get_path_floyd(
    next_map: Dict[Tuple[str, str], Optional[str]], start: str, goal: str
) -> List[str]:
    """Floyd-Warshall next 테이블로 start -> goal 경로 복원"""
    path = [start]
    cur = start
    while cur != goal and next_map.get((cur, goal)):
        cur = next_map[(cur, goal)]
        path.append(cur)
    return path if cur == goal else []


# -----------------------------------------------------------------------------
# 2-7. 복합: BFS + 경로 복원 (단계별 거리 + 경로)
# -----------------------------------------------------------------------------
def pathfinding_composite_bfs_with_distances(
    graph: Dict[str, List[Tuple[str, int]]],
    start: str,
) -> Tuple[Dict[str, int], Dict[str, Optional[str]]]:
    """
    이름: BFS + 거리/경로 테이블 (복합)
    난이도: 보통
    설명: BFS로 start에서 모든 노드까지 거리와 직전 노드(경로 복원용) 계산.
    """
    info = AlgoInfo(
        "BFS 거리·경로 테이블",
        "BFS with Distance & Predecessor",
        Difficulty.MEDIUM,
        "시작점에서 모든 노드까지 거리와 경로 정보",
    )
    print_algo_header(info, "패스파인딩(복합)")
    dist = {start: 0}
    prev = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        d = dist[node]
        for neighbor, _ in graph.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = d + 1
                prev[neighbor] = node
                q.append(neighbor)
    return (dist, prev)


def reconstruct_path(
    prev: Dict[str, Optional[str]], start: str, goal: str
) -> List[str]:
    """prev 맵으로 start -> goal 경로 복원"""
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path if path and path[0] == start else []


# =============================================================================
# 실행: 모든 예제 실행
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print(" 어그리게이팅 & 패스파인딩 알고리즘 예제 실행")
    print("=" * 60)

    # ---- 어그리게이션 데이터 ----
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    records = [("가", 100), ("나", 200), ("가", 50), ("나", 50), ("다", 300)]

    # 1) 합계
    s = aggregation_sum(data)
    print(f"  결과(합계): {s}")

    # 2) 평균
    a = aggregation_average(data)
    print(f"  결과(평균): {a}")

    # 3) 최솟값·최댓값
    mn, mx = aggregation_min_max(data)
    print(f"  결과(최소, 최대): {mn}, {mx}")

    # 4) 카운트
    c = aggregation_count(data, predicate=lambda x: x >= 25)
    print(f"  결과(x>=25 개수): {c}")

    # 5) 그룹별 합계
    g_sum = aggregation_group_by_sum(records)
    print(f"  결과(그룹별 합계): {g_sum}")

    # 6) 복합: 그룹별 평균 + 최대 평균 그룹
    avg_map, top = aggregation_composite_group_avg_and_top(records)
    print(f"  결과(그룹별 평균): {avg_map}")
    print(f"  결과(최대 평균 그룹): {top}")

    # 7) 누적 합
    run = aggregation_running_sum(data)
    print(f"  결과(누적 합): {run}")

    # 8) 분산
    var = aggregation_variance(data)
    print(f"  결과(분산): {var}")

    # ---- 패스파인딩: 샘플 그래프 ----
    graph = build_sample_graph()
    start, goal = "A", "E"

    # 1) BFS
    path_bfs = pathfinding_bfs(graph, start, goal)
    print(f"  결과(BFS 경로): {path_bfs}")

    # 2) DFS
    path_dfs = pathfinding_dfs(graph, start, goal)
    print(f"  결과(DFS 경로): {path_dfs}")

    # 3) Dijkstra
    path_dij, cost_dij = pathfinding_dijkstra(graph, start, goal)
    print(f"  결과(Dijkstra 경로): {path_dij}, 비용: {cost_dij}")

    # 4) A* (휴리스틱 0이면 Dijkstra와 동일)
    path_astar, cost_astar = pathfinding_astar(graph, start, goal)
    print(f"  결과(A* 경로): {path_astar}, 비용: {cost_astar}")

    # 5) Bellman-Ford
    path_bf, cost_bf = pathfinding_bellman_ford(graph, start, goal)
    print(f"  결과(Bellman-Ford 경로): {path_bf}, 비용: {cost_bf}")

    # 6) Floyd-Warshall
    dist_fw, next_fw = pathfinding_floyd_warshall(graph)
    path_fw = get_path_floyd(next_fw, start, goal)
    print(f"  결과(Floyd-Warshall A->E 거리): {dist_fw.get(('A','E'))}")
    print(f"  결과(Floyd-Warshall 경로): {path_fw}")

    # 7) 복합: BFS 거리/경로 테이블
    dist_all, prev_all = pathfinding_composite_bfs_with_distances(graph, start)
    path_recon = reconstruct_path(prev_all, start, goal)
    print(f"  결과(BFS 거리 테이블): {dist_all}")
    print(f"  결과(복원 경로 A->E): {path_recon}")

    print("\n" + "=" * 60)
    print(" 모든 예제 실행 완료.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
