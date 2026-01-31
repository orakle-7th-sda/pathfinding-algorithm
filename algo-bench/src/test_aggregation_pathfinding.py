#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
어그리게이팅 & 패스파인딩 알고리즘 — 테스트 (Input / Expected Output 보장)
=============================================================================
- 각 테스트: Given(입력) / When(호출) / Then(기대 출력) 주석 + assert 로 검증
- 실행: python3 test_aggregation_pathfinding.py
- 또는: python3 -m unittest test_aggregation_pathfinding -v
- (pytest 없이 표준 라이브러리 unittest만 사용)
=============================================================================
"""

import unittest

# 테스트 대상 모듈 (같은 디렉터리에 aggregation_pathfinding_examples.py 필요)
from aggregation_pathfinding_examples import (
    aggregation_sum,
    aggregation_average,
    aggregation_min_max,
    aggregation_count,
    aggregation_group_by_sum,
    aggregation_composite_group_avg_and_top,
    aggregation_running_sum,
    aggregation_variance,
    build_sample_graph,
    pathfinding_bfs,
    pathfinding_dfs,
    pathfinding_dijkstra,
    pathfinding_astar,
    pathfinding_bellman_ford,
    pathfinding_floyd_warshall,
    get_path_floyd,
    pathfinding_composite_bfs_with_distances,
    reconstruct_path,
)


# =============================================================================
# 공통 Mock 데이터 (테스트용)
# =============================================================================

SAMPLE_DATA = [10.0, 20.0, 30.0, 40.0, 50.0]
SAMPLE_RECORDS = [("가", 100), ("나", 200), ("가", 50), ("나", 50), ("다", 300)]


def get_sample_graph():
    """패스파인딩용 가중 그래프 (A~E, 예제와 동일)"""
    return build_sample_graph()


# =============================================================================
# PART 1: 어그리게이션 테스트
# =============================================================================


class TestAggregationSum(unittest.TestCase):
    """합계 (Sum) — 입력이 정해지면 출력이 한 값으로 정해짐"""

    def test_basic_sum(self):
        # Given: 5개 원소 [10, 20, 30, 40, 50]
        # When: aggregation_sum(data) 호출
        # Then: 10+20+30+40+50 = 150.0
        result = aggregation_sum(SAMPLE_DATA)
        self.assertEqual(result, 150.0)

    def test_empty_list(self):
        # Given: 빈 리스트 []
        # When: aggregation_sum([]) 호출
        # Then: 0.0 (초기값 유지)
        result = aggregation_sum([])
        self.assertEqual(result, 0.0)

    def test_single_element(self):
        # Given: 원소 하나 [7.0]
        # When: aggregation_sum([7.0]) 호출
        # Then: 7.0
        result = aggregation_sum([7.0])
        self.assertEqual(result, 7.0)


class TestAggregationAverage(unittest.TestCase):
    """평균 (Average) — 입력에 대한 기대 평균값 검증"""

    def test_basic_average(self):
        # Given: [10, 20, 30, 40, 50]
        # When: aggregation_average(data) 호출
        # Then: 150/5 = 30.0
        result = aggregation_average(SAMPLE_DATA)
        self.assertEqual(result, 30.0)

    def test_empty_returns_zero(self):
        # Given: 빈 리스트
        # When: aggregation_average([]) 호출
        # Then: 0.0 (정의상)
        result = aggregation_average([])
        self.assertEqual(result, 0.0)


class TestAggregationMinMax(unittest.TestCase):
    """최솟값·최댓값 (Min & Max) — 한 번에 (min, max) 튜플 반환"""

    def test_basic_min_max(self):
        # Given: [10, 20, 30, 40, 50]
        # When: aggregation_min_max(data) 호출
        # Then: (10.0, 50.0)
        result = aggregation_min_max(SAMPLE_DATA)
        self.assertEqual(result, (10.0, 50.0))

    def test_single_element_min_max(self):
        # Given: [3.0] 하나
        # When: aggregation_min_max([3.0]) 호출
        # Then: (3.0, 3.0)
        result = aggregation_min_max([3.0])
        self.assertEqual(result, (3.0, 3.0))


class TestAggregationCount(unittest.TestCase):
    """카운트 (Count) — 조건 만족 개수 또는 전체 개수"""

    def test_count_with_predicate(self):
        # Given: [10, 20, 30, 40, 50], 조건 x >= 25
        # When: aggregation_count(data, predicate=lambda x: x >= 25) 호출
        # Then: 30, 40, 50 → 3
        result = aggregation_count(SAMPLE_DATA, predicate=lambda x: x >= 25)
        self.assertEqual(result, 3)

    def test_count_without_predicate(self):
        # Given: [10, 20, 30, 40, 50], 조건 없음
        # When: aggregation_count(data) 호출
        # Then: len(data) = 5
        result = aggregation_count(SAMPLE_DATA)
        self.assertEqual(result, 5)


class TestAggregationGroupBySum(unittest.TestCase):
    """그룹별 합계 (Group By Sum) — 키별 합이 기대값과 일치하는지"""

    def test_group_by_sum(self):
        # Given: [("가",100), ("나",200), ("가",50), ("나",50), ("다",300)]
        # When: aggregation_group_by_sum(records) 호출
        # Then: 가=150, 나=250, 다=300
        result = aggregation_group_by_sum(SAMPLE_RECORDS)
        self.assertEqual(result, {"가": 150.0, "나": 250.0, "다": 300.0})


class TestAggregationCompositeGroupAvgAndTop(unittest.TestCase):
    """그룹별 평균 + 최대 평균 그룹 (복합) — (평균 딕셔너리, 최대 평균 그룹 키)"""

    def test_group_avg_and_top(self):
        # Given: [("가",100), ("나",200), ("가",50), ("나",50), ("다",300)]
        # When: aggregation_composite_group_avg_and_top(records) 호출
        # Then: 평균 = 가 75, 나 125, 다 300 / 최대 평균 그룹 = "다"
        avg_map, top = aggregation_composite_group_avg_and_top(SAMPLE_RECORDS)
        self.assertEqual(avg_map, {"가": 75.0, "나": 125.0, "다": 300.0})
        self.assertEqual(top, "다")


class TestAggregationRunningSum(unittest.TestCase):
    """누적 합 (Running Sum) — 각 위치까지의 부분 합 리스트"""

    def test_running_sum(self):
        # Given: [10, 20, 30, 40, 50]
        # When: aggregation_running_sum(data) 호출
        # Then: [10, 30, 60, 100, 150]
        result = aggregation_running_sum(SAMPLE_DATA)
        self.assertEqual(result, [10.0, 30.0, 60.0, 100.0, 150.0])


class TestAggregationVariance(unittest.TestCase):
    """분산 (Variance) — 표본 분산 식으로 기대값 검증"""

    def test_variance_five_elements(self):
        # Given: [10, 20, 30, 40, 50], 평균 30
        # When: aggregation_variance(data) 호출
        # Then: 표본 분산 = (0²+100²+0²+100²+400²)/4 = 60000/4 = 250.0
        result = aggregation_variance(SAMPLE_DATA)
        self.assertEqual(result, 250.0)


# =============================================================================
# PART 2: 패스파인딩 테스트
# =============================================================================


class TestPathfindingBfs(unittest.TestCase):
    """BFS — 간선 개수 기준 최단 경로 (가중치 무시)"""

    def test_bfs_a_to_e(self):
        # Given: 그래프 A~E, start=A, goal=E
        # When: pathfinding_bfs(graph, "A", "E") 호출
        # Then: 간선 수 최소 경로 하나 (예: A-B-D-E 또는 A-C-D-E 등)
        graph = get_sample_graph()
        path = pathfinding_bfs(graph, "A", "E")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")
        # BFS는 간선 수 기준이므로 길이 4 (A->?->?->E)
        self.assertEqual(len(path), 4)

    def test_bfs_same_start_goal(self):
        # Given: start=goal=A
        # When: pathfinding_bfs(graph, "A", "A") 호출
        # Then: [ "A" ]
        graph = get_sample_graph()
        path = pathfinding_bfs(graph, "A", "A")
        self.assertEqual(path, ["A"])


class TestPathfindingDfs(unittest.TestCase):
    """DFS — 경로 존재 시 한 경로 반환 (최단 보장 아님)"""

    def test_dfs_finds_some_path_a_to_e(self):
        # Given: 그래프, start=A, goal=E
        # When: pathfinding_dfs(graph, "A", "E") 호출
        # Then: A로 시작하고 E로 끝나는 유효 경로
        graph = get_sample_graph()
        path = pathfinding_dfs(graph, "A", "E")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")


class TestPathfindingDijkstra(unittest.TestCase):
    """다익스트라 — 가중치 합 기준 최단 경로 및 비용"""

    def test_dijkstra_a_to_e(self):
        # Given: 예제 그래프 (A-B=1, B-C=2, C-D=1, D-E=3 등), A -> E
        # When: pathfinding_dijkstra(graph, "A", "E") 호출
        # Then: 최단 비용 7, 경로는 A-B-C-D-E (비용 1+2+1+3=7)
        graph = get_sample_graph()
        path, cost = pathfinding_dijkstra(graph, "A", "E")
        self.assertIsNotNone(path)
        self.assertEqual(cost, 7)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")
        self.assertEqual(path, ["A", "B", "C", "D", "E"])


class TestPathfindingAstar(unittest.TestCase):
    """A* — 휴리스틱 0이면 다익스트라와 동일한 경로/비용"""

    def test_astar_same_as_dijkstra_when_h_zero(self):
        # Given: 휴리스틱 없음(0), A -> E
        # When: pathfinding_astar(graph, "A", "E") 호출
        # Then: 비용 7, 경로 A-B-C-D-E (다익스트라와 동일)
        graph = get_sample_graph()
        path, cost = pathfinding_astar(graph, "A", "E")
        self.assertIsNotNone(path)
        self.assertEqual(cost, 7)
        self.assertEqual(path, ["A", "B", "C", "D", "E"])


class TestPathfindingBellmanFord(unittest.TestCase):
    """벨만-포드 — 양수만 있어도 다익스트라와 동일한 결과 기대"""

    def test_bellman_ford_positive_weights(self):
        # Given: 양수 가중치만 있는 그래프, A -> E
        # When: pathfinding_bellman_ford(graph, "A", "E") 호출
        # Then: 경로 존재, 비용 7
        graph = get_sample_graph()
        path, cost = pathfinding_bellman_ford(graph, "A", "E")
        self.assertIsNotNone(path)
        self.assertEqual(cost, 7)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")


class TestPathfindingFloydWarshall(unittest.TestCase):
    """플로이드-워셜 — 모든 쌍 최단 거리 및 경로 복원"""

    def test_floyd_warshall_a_to_e(self):
        # Given: 예제 그래프
        # When: pathfinding_floyd_warshall(graph), get_path_floyd(next_map, "A", "E")
        # Then: A->E 거리 7, 경로 복원 시 A-B-C-D-E
        graph = get_sample_graph()
        dist_map, next_map = pathfinding_floyd_warshall(graph)
        d_ae = dist_map[("A", "E")]
        self.assertEqual(d_ae, 7)
        path = get_path_floyd(next_map, "A", "E")
        self.assertEqual(path, ["A", "B", "C", "D", "E"])


class TestPathfindingCompositeBfsWithDistances(unittest.TestCase):
    """BFS 거리·경로 테이블 + 경로 복원 (복합)"""

    def test_bfs_distances_and_path_reconstruction(self):
        # Given: 그래프, start=A
        # When: pathfinding_composite_bfs_with_distances(graph, "A"), reconstruct_path(prev, "A", "E")
        # Then: A로부터 거리 A=0, B=1, C=1, D=2, E=3 / A->E 경로 복원 가능
        graph = get_sample_graph()
        dist, prev = pathfinding_composite_bfs_with_distances(graph, "A")
        self.assertEqual(dist, {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3})
        path = reconstruct_path(prev, "A", "E")
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")
        self.assertEqual(len(path), 4)


# =============================================================================
# 실행: python3 test_aggregation_pathfinding.py 또는 python3 -m unittest ...
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
