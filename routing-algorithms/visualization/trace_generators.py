"""
Trace generators for frame-by-frame replay visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

from algorithms.base import Pool, SwapRequest
from algorithms.single.bellman_ford import BellmanFordRouting
from algorithms.single.dijkstra import DijkstraRouting
from algorithms.single.k_best import KBestRouting
from algorithms.composite.greedy_split import GreedySplit
from algorithms.composite.convex_split import ConvexSplit
from algorithms.composite.kkt_split import KKTOptimalSplit


@dataclass
class TraceMetadata:
    algorithm: str
    trace_kind: str
    created_at_utc: str
    request: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_metadata(algorithm_name: str, trace_kind: str, request: SwapRequest) -> Dict[str, Any]:
    return asdict(
        TraceMetadata(
            algorithm=algorithm_name,
            trace_kind=trace_kind,
            created_at_utc=_utc_now_iso(),
            request={
                "token_in": request.token_in,
                "token_out": request.token_out,
                "amount_in": request.amount_in,
            },
        )
    )


def _serialize_pool(pool: Pool) -> Dict[str, Any]:
    return {
        "pool_id": pool.pool_id,
        "dex": pool.dex,
        "token0": pool.token0,
        "token1": pool.token1,
        "reserve0": pool.reserve0,
        "reserve1": pool.reserve1,
        "fee": pool.fee,
    }


def _compact_float_dict(values: Dict[str, float], digits: int = 8) -> Dict[str, float]:
    return {k: round(v, digits) for k, v in values.items()}


def _graph_payload_from_pools(pools: List[Pool]) -> Dict[str, Any]:
    nodes = sorted({pool.token0 for pool in pools} | {pool.token1 for pool in pools})
    edges = [
        {
            "from": pool.token0,
            "to": pool.token1,
            "pool_id": pool.pool_id,
            "dex": pool.dex,
            "fee": pool.fee,
            "spot_0_to_1": pool.get_spot_price(pool.token0),
            "spot_1_to_0": pool.get_spot_price(pool.token1),
        }
        for pool in pools
    ]
    return {"nodes": nodes, "edges": edges}


def _direct_pair_graph_payload(pools: List[Pool], token_in: str, token_out: str) -> Dict[str, Any]:
    edges = [
        {
            "from": token_in,
            "to": token_out,
            "pool_id": pool.pool_id,
            "dex": pool.dex,
            "fee": pool.fee,
        }
        for pool in pools
    ]
    return {"nodes": [token_in, token_out], "edges": edges}


def _empty_trace(
    algorithm_name: str,
    trace_kind: str,
    request: SwapRequest,
    graph: Dict[str, Any],
    frames: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": _build_metadata(algorithm_name, trace_kind, request),
        "graph": graph,
        "frames": frames,
        "result": result,
    }


def _allocation_rows(
    pools: List[Pool], allocations: Dict[str, float], token_in: str, total_amount_in: float
) -> List[Dict[str, Any]]:
    pool_map = {pool.pool_id: pool for pool in pools}
    rows: List[Dict[str, Any]] = []
    for pool_id, amount_in in allocations.items():
        if amount_in <= 0:
            continue
        pool = pool_map[pool_id]
        amount_out = pool.get_amount_out(amount_in, token_in)
        rows.append(
            {
                "pool_id": pool_id,
                "dex": pool.dex,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "share_pct": (amount_in / total_amount_in * 100.0) if total_amount_in > 0 else 0.0,
            }
        )
    return rows


class DijkstraTraceGenerator(DijkstraRouting):
    """Generate frame trace for Dijkstra routing."""

    def generate_trace(self, request: SwapRequest, max_frames: int = 260) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        truncated = False

        def capture(
            event: str,
            description: str,
            distances: Dict[str, Tuple[float, Optional[str], Optional[Pool]]],
            pq: List[Tuple[float, str]],
            visited: set,
            current: Optional[str] = None,
            edge: Optional[Dict[str, str]] = None,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            nonlocal truncated
            if len(frames) >= max_frames:
                truncated = True
                return

            known_distances = {token: dist_info[0] for token, dist_info in distances.items()}
            implied_rates = {
                token: math.exp(-dist) if math.isfinite(dist) else 0.0
                for token, dist in known_distances.items()
            }
            frame = {
                "step": len(frames),
                "event": event,
                "description": description,
                "current_node": current,
                "visited": sorted(visited),
                "distances": _compact_float_dict(known_distances, digits=10),
                "implied_rates": _compact_float_dict(implied_rates, digits=10),
                "priority_queue": [
                    {"distance": round(item[0], 10), "token": item[1]}
                    for item in sorted(pq, key=lambda x: x[0])
                ],
            }
            if edge is not None:
                frame["edge_highlight"] = edge
            if extra:
                frame.update(extra)
            frames.append(frame)

        start = request.token_in
        end = request.token_out
        graph = _graph_payload_from_pools(self.pools)

        if start == end:
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                [
                    {
                        "step": 0,
                        "event": "done",
                        "description": "token_in and token_out are identical.",
                        "current_node": start,
                        "visited": [start],
                        "distances": {start: 0.0},
                        "implied_rates": {start: 1.0},
                        "priority_queue": [],
                        "best_path": [start],
                    }
                ],
                {
                    "path": [start],
                    "pools": [],
                    "amount_out": request.amount_in,
                    "truncated": False,
                },
            )

        distances: Dict[str, Tuple[float, Optional[str], Optional[Pool]]] = {
            start: (0.0, None, None)
        }
        pq: List[Tuple[float, str]] = [(0.0, start)]
        visited = set()

        capture("init", "Initialized Dijkstra with start token.", distances, pq, visited, current=start)

        while pq:
            dist, current = heapq.heappop(pq)
            if current in visited:
                capture("skip", f"Skip {current}: already finalized.", distances, pq, visited, current=current)
                continue

            visited.add(current)
            capture("pop", f"Popped {current} from queue with best known distance.", distances, pq, visited, current=current)

            if current == end:
                capture("goal", f"Reached target token {end}.", distances, pq, visited, current=current)
                break

            for next_token, pool in self.token_graph.get(current, []):
                if next_token in visited:
                    continue

                weight = self._get_edge_weight_spot(pool, current)
                new_dist = dist + weight
                prev_dist = distances.get(next_token, (float("inf"), None, None))[0]

                if new_dist < prev_dist:
                    distances[next_token] = (new_dist, current, pool)
                    heapq.heappush(pq, (new_dist, next_token))
                    capture(
                        "relax",
                        (
                            f"Relax edge {current}->{next_token}; "
                            f"distance improved {prev_dist:.6f} -> {new_dist:.6f}."
                        ),
                        distances,
                        pq,
                        visited,
                        current=current,
                        edge={
                            "from": current,
                            "to": next_token,
                            "pool_id": pool.pool_id,
                            "dex": pool.dex,
                        },
                    )

        if end not in distances:
            capture("done", f"No route found from {start} to {end}.", distances, pq, visited)
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                frames,
                {
                    "path": None,
                    "pools": [],
                    "amount_out": 0.0,
                    "truncated": truncated,
                },
            )

        path: List[str] = []
        pools_used: List[Pool] = []
        node = end
        while node is not None:
            path.append(node)
            _, prev, prev_pool = distances[node]
            if prev_pool is not None:
                pools_used.append(prev_pool)
            node = prev
        path.reverse()
        pools_used.reverse()

        amount_out = self.calculate_route_output(path, pools_used, request.amount_in)
        capture(
            "done",
            "Route reconstruction complete.",
            distances,
            pq,
            visited,
            current=end,
            extra={
                "best_path": path,
                "best_path_pool_ids": [pool.pool_id for pool in pools_used],
                "estimated_amount_out": round(amount_out, 10),
            },
        )

        return _empty_trace(
            self.metadata.name,
            "graph_search",
            request,
            graph,
            frames,
            {
                "path": path,
                "pools": [_serialize_pool(pool) for pool in pools_used],
                "amount_out": amount_out,
                "truncated": truncated,
            },
        )


class BellmanFordTraceGenerator(BellmanFordRouting):
    """Generate frame trace for Bellman-Ford routing."""

    def generate_trace(self, request: SwapRequest, max_frames: int = 360) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        truncated = False

        def capture(
            event: str,
            description: str,
            distances: Dict[str, float],
            iteration: int,
            edge: Optional[Dict[str, str]] = None,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            nonlocal truncated
            if len(frames) >= max_frames:
                truncated = True
                return

            finite_tokens = [t for t, d in distances.items() if math.isfinite(d)]
            implied_rates = {
                token: math.exp(-dist) if math.isfinite(dist) else 0.0
                for token, dist in distances.items()
            }
            frame = {
                "step": len(frames),
                "event": event,
                "description": description,
                "current_node": edge["from"] if edge else None,
                "visited": sorted(finite_tokens),
                "distances": _compact_float_dict(
                    {k: v for k, v in distances.items() if math.isfinite(v)}, digits=10
                ),
                "implied_rates": _compact_float_dict(
                    {k: v for k, v in implied_rates.items() if v > 0}, digits=10
                ),
                "priority_queue": [],
                "iteration": iteration,
            }
            if edge is not None:
                frame["edge_highlight"] = edge
            if extra:
                frame.update(extra)
            frames.append(frame)

        start = request.token_in
        end = request.token_out
        graph = _graph_payload_from_pools(self.pools)

        if start == end:
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                [
                    {
                        "step": 0,
                        "event": "done",
                        "description": "token_in and token_out are identical.",
                        "current_node": start,
                        "visited": [start],
                        "distances": {start: 0.0},
                        "implied_rates": {start: 1.0},
                        "priority_queue": [],
                        "best_path": [start],
                        "iteration": 0,
                    }
                ],
                {
                    "path": [start],
                    "pools": [],
                    "amount_out": request.amount_in,
                    "truncated": False,
                },
            )

        if start not in self.token_graph:
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                [],
                {"path": None, "pools": [], "amount_out": 0.0, "truncated": False},
            )

        vertices = sorted(self.token_graph.keys())
        distances: Dict[str, float] = {token: float("inf") for token in vertices}
        predecessor: Dict[str, Tuple[Optional[str], Optional[Pool]]] = {
            token: (None, None) for token in vertices
        }
        distances[start] = 0.0

        capture("init", "Initialized Bellman-Ford with start token.", distances, iteration=0)

        for i in range(max(0, len(vertices) - 1)):
            updated = False
            capture("pass_start", f"Start relaxation pass {i + 1}.", distances, iteration=i + 1)
            for token in vertices:
                base = distances[token]
                if not math.isfinite(base):
                    continue
                for next_token, pool in self.token_graph.get(token, []):
                    weight = self._get_edge_weight_spot(pool, token)
                    if not math.isfinite(weight):
                        continue
                    new_dist = base + weight
                    prev_dist = distances[next_token]
                    if new_dist < prev_dist:
                        distances[next_token] = new_dist
                        predecessor[next_token] = (token, pool)
                        updated = True
                        capture(
                            "relax",
                            (
                                f"Relax edge {token}->{next_token}; "
                                f"distance improved {prev_dist:.6f} -> {new_dist:.6f}."
                            ),
                            distances,
                            iteration=i + 1,
                            edge={
                                "from": token,
                                "to": next_token,
                                "pool_id": pool.pool_id,
                                "dex": pool.dex,
                            },
                        )
            if not updated:
                capture(
                    "early_stop",
                    f"No updates in pass {i + 1}; Bellman-Ford converged early.",
                    distances,
                    iteration=i + 1,
                )
                break

        negative_cycle_detected = False
        for token in vertices:
            base = distances[token]
            if not math.isfinite(base):
                continue
            for next_token, pool in self.token_graph.get(token, []):
                weight = self._get_edge_weight_spot(pool, token)
                if not math.isfinite(weight):
                    continue
                if base + weight < distances[next_token]:
                    negative_cycle_detected = True
                    capture(
                        "negative_cycle",
                        "Detected a reachable negative cycle candidate (spot-rate arbitrage signal).",
                        distances,
                        iteration=len(vertices),
                        edge={
                            "from": token,
                            "to": next_token,
                            "pool_id": pool.pool_id,
                            "dex": pool.dex,
                        },
                    )
                    break
            if negative_cycle_detected:
                break

        if not math.isfinite(distances.get(end, float("inf"))):
            capture("done", f"No route found from {start} to {end}.", distances, iteration=len(vertices))
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                frames,
                {
                    "path": None,
                    "pools": [],
                    "amount_out": 0.0,
                    "negative_cycle_detected": negative_cycle_detected,
                    "truncated": truncated,
                },
            )

        path: List[str] = []
        pools_used: List[Pool] = []
        node: Optional[str] = end
        hop_guard = 0
        used_fallback = False
        while node is not None:
            path.append(node)
            prev, prev_pool = predecessor[node]
            if prev_pool is not None:
                pools_used.append(prev_pool)
            node = prev
            hop_guard += 1
            if hop_guard > len(vertices) + 1:
                fallback_path, fallback_pools = DijkstraRouting(self.pools)._dijkstra_find_path(
                    start, end, amount=request.amount_in
                )
                if fallback_path is None:
                    capture(
                        "done",
                        "Route reconstruction failed due to predecessor loop.",
                        distances,
                        iteration=len(vertices),
                    )
                    return _empty_trace(
                        self.metadata.name,
                        "graph_search",
                        request,
                        graph,
                        frames,
                        {
                            "path": None,
                            "pools": [],
                            "amount_out": 0.0,
                            "negative_cycle_detected": negative_cycle_detected,
                            "truncated": truncated,
                        },
                    )
                path = fallback_path
                pools_used = fallback_pools
                used_fallback = True
                capture(
                    "fallback",
                    "Predecessor loop detected; switched to Dijkstra fallback path.",
                    distances,
                    iteration=len(vertices),
                    extra={
                        "best_path": path,
                        "best_path_pool_ids": [pool.pool_id for pool in pools_used],
                    },
                )
                break

        if not used_fallback:
            path.reverse()
            pools_used.reverse()
        if not path or path[0] != start:
            fallback_path, fallback_pools = DijkstraRouting(self.pools)._dijkstra_find_path(
                start, end, amount=request.amount_in
            )
            if fallback_path is None:
                capture(
                    "done",
                    "Route reconstruction produced invalid path.",
                    distances,
                    iteration=len(vertices),
                )
                return _empty_trace(
                    self.metadata.name,
                    "graph_search",
                    request,
                    graph,
                    frames,
                    {
                        "path": None,
                        "pools": [],
                        "amount_out": 0.0,
                        "negative_cycle_detected": negative_cycle_detected,
                        "truncated": truncated,
                    },
                )
            path = fallback_path
            pools_used = fallback_pools
            used_fallback = True
            capture(
                "fallback",
                "Invalid Bellman-Ford predecessor chain; switched to Dijkstra fallback path.",
                distances,
                iteration=len(vertices),
                extra={
                    "best_path": path,
                    "best_path_pool_ids": [pool.pool_id for pool in pools_used],
                },
            )

        amount_out = self.calculate_route_output(path, pools_used, request.amount_in)
        capture(
            "done",
            "Route reconstruction complete.",
            distances,
            iteration=len(vertices),
            extra={
                "best_path": path,
                "best_path_pool_ids": [pool.pool_id for pool in pools_used],
                "estimated_amount_out": round(amount_out, 10),
                "negative_cycle_detected": negative_cycle_detected,
            },
        )

        return _empty_trace(
            self.metadata.name,
            "graph_search",
            request,
            graph,
            frames,
            {
                "path": path,
                "pools": [_serialize_pool(pool) for pool in pools_used],
                "amount_out": amount_out,
                "negative_cycle_detected": negative_cycle_detected,
                "truncated": truncated,
            },
        )


class KBestTraceGenerator(KBestRouting):
    """Generate frame trace for K-Best routing."""

    def generate_trace(self, request: SwapRequest, max_frames: int = 260) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        truncated = False

        def capture(event: str, description: str, payload: Optional[Dict[str, Any]] = None) -> None:
            nonlocal truncated
            if len(frames) >= max_frames:
                truncated = True
                return
            frame = {
                "step": len(frames),
                "event": event,
                "description": description,
            }
            if payload:
                frame.update(payload)
            frames.append(frame)

        graph = _graph_payload_from_pools(self.pools)

        if request.token_in == request.token_out:
            capture("done", "token_in and token_out are identical.", {"best_path": [request.token_in]})
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                frames,
                {
                    "path": [request.token_in],
                    "pools": [],
                    "amount_out": request.amount_in,
                    "truncated": truncated,
                },
            )

        all_candidates: List[Tuple[List[str], List[Pool]]] = []

        def dfs(
            current: str, path: List[str], pools: List[Pool], visited_tokens: set
        ) -> None:
            if len(pools) > self.MAX_HOPS:
                return
            if current == request.token_out and pools:
                all_candidates.append((path.copy(), pools.copy()))
                if len(frames) < max_frames:
                    capture(
                        "candidate_found",
                        "Found candidate path during DFS enumeration.",
                        {
                            "candidate_path": path.copy(),
                            "candidate_pool_ids": [p.pool_id for p in pools],
                            "hop_count": len(pools),
                            "spot_score": self._spot_path_score(path, pools),
                        },
                    )
                return
            if current not in self.token_graph:
                return
            for next_token, pool in self.token_graph[current]:
                if next_token in visited_tokens:
                    continue
                visited_tokens.add(next_token)
                path.append(next_token)
                pools.append(pool)
                dfs(next_token, path, pools, visited_tokens)
                pools.pop()
                path.pop()
                visited_tokens.remove(next_token)

        capture(
            "init",
            "Start K-Best candidate enumeration.",
            {
                "max_hops": self.MAX_HOPS,
                "k_paths": self.k_paths,
                "token_in": request.token_in,
                "token_out": request.token_out,
                "amount_in": request.amount_in,
            },
        )
        dfs(request.token_in, [request.token_in], [], {request.token_in})

        if not all_candidates:
            capture("done", "No candidate paths found.")
            return _empty_trace(
                self.metadata.name,
                "graph_search",
                request,
                graph,
                frames,
                {
                    "path": None,
                    "pools": [],
                    "amount_out": 0.0,
                    "truncated": truncated,
                },
            )

        ranked = sorted(all_candidates, key=lambda item: self._spot_path_score(item[0], item[1]))
        top_k = ranked[: self.k_paths]
        capture(
            "ranked",
            "Ranked candidate paths by fee-adjusted spot score.",
            {
                "candidate_count_total": len(all_candidates),
                "candidate_count_kept": len(top_k),
                "top_candidates": [
                    {
                        "rank": i + 1,
                        "path": path,
                        "pool_ids": [p.pool_id for p in pools],
                        "spot_score": self._spot_path_score(path, pools),
                    }
                    for i, (path, pools) in enumerate(top_k)
                ],
            },
        )

        best_route: Optional[Dict[str, Any]] = None
        for idx, (path, pools) in enumerate(top_k, start=1):
            amount_out = self.calculate_route_output(path, pools, request.amount_in)
            is_best = best_route is None or amount_out > best_route["amount_out"]
            if is_best:
                best_route = {
                    "path": path,
                    "pools": pools,
                    "amount_out": amount_out,
                }
            capture(
                "evaluate_candidate",
                "Re-evaluated candidate with full trade-size simulation.",
                {
                    "rank": idx,
                    "path": path,
                    "pool_ids": [p.pool_id for p in pools],
                    "candidate_amount_out": amount_out,
                    "best_path_so_far": best_route["path"] if best_route else None,
                    "best_amount_out_so_far": best_route["amount_out"] if best_route else None,
                    "best_path": best_route["path"] if best_route else None,
                },
            )

        assert best_route is not None
        capture(
            "done",
            "K-Best re-ranking complete.",
            {
                "best_path": best_route["path"],
                "best_path_pool_ids": [p.pool_id for p in best_route["pools"]],
                "estimated_amount_out": best_route["amount_out"],
            },
        )

        return _empty_trace(
            self.metadata.name,
            "graph_search",
            request,
            graph,
            frames,
            {
                "path": best_route["path"],
                "pools": [_serialize_pool(pool) for pool in best_route["pools"]],
                "amount_out": best_route["amount_out"],
                "truncated": truncated,
            },
        )


class GreedySplitTraceGenerator(GreedySplit):
    """Generate frame trace for greedy split optimization."""

    def generate_trace(self, request: SwapRequest, max_frames: int = 220) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        truncated = False

        def capture(event: str, description: str, payload: Optional[Dict[str, Any]] = None) -> None:
            nonlocal truncated
            if len(frames) >= max_frames:
                truncated = True
                return
            frame = {"step": len(frames), "event": event, "description": description}
            if payload:
                frame.update(payload)
            frames.append(frame)

        pools = self.get_pools_for_pair(request.token_in, request.token_out)
        graph = _direct_pair_graph_payload(pools, request.token_in, request.token_out)
        capture("init", "Initialized Greedy split optimizer.", {"pool_count": len(pools), "iterations": self.ITERATIONS})

        if not pools:
            capture("done", "No direct pools found.")
            return _empty_trace(
                self.metadata.name,
                "split_optimization",
                request,
                graph,
                frames,
                {"allocations": {}, "total_amount_out": 0.0, "truncated": truncated},
            )

        allocations = {pool.pool_id: 0.0 for pool in pools}
        chunk_size = request.amount_in / self.ITERATIONS

        capture(
            "params",
            "Prepared greedy chunking parameters.",
            {
                "chunk_size": chunk_size,
                "pools": [
                    {"pool_id": p.pool_id, "dex": p.dex, "fee": p.fee, "reserve0": p.reserve0, "reserve1": p.reserve1}
                    for p in pools
                ],
            },
        )

        for i in range(self.ITERATIONS):
            best_pool: Optional[Pool] = None
            best_marginal_output = 0.0
            marginal_outputs = {}

            for pool in pools:
                current_allocation = allocations[pool.pool_id]
                out_before = pool.get_amount_out(current_allocation, request.token_in) if current_allocation > 0 else 0.0
                out_after = pool.get_amount_out(current_allocation + chunk_size, request.token_in)
                marginal = out_after - out_before
                marginal_outputs[pool.pool_id] = marginal
                if marginal > best_marginal_output:
                    best_marginal_output = marginal
                    best_pool = pool

            if best_pool is not None:
                allocations[best_pool.pool_id] += chunk_size

            capture(
                "iteration",
                "Assigned one chunk to best marginal pool.",
                {
                    "iteration": i + 1,
                    "winner_pool_id": best_pool.pool_id if best_pool else None,
                    "winner_dex": best_pool.dex if best_pool else None,
                    "best_marginal_output": best_marginal_output,
                    "marginal_outputs": _compact_float_dict(marginal_outputs, digits=10),
                    "allocations": _compact_float_dict(allocations, digits=12),
                    "per_pool_results": _allocation_rows(pools, allocations, request.token_in, request.amount_in),
                },
            )

        per_pool_results = _allocation_rows(pools, allocations, request.token_in, request.amount_in)
        total_amount_out = sum(item["amount_out"] for item in per_pool_results)
        capture(
            "done",
            "Greedy split finished.",
            {
                "final_allocations": _compact_float_dict(allocations, digits=12),
                "per_pool_results": per_pool_results,
                "total_amount_out": total_amount_out,
            },
        )

        return _empty_trace(
            self.metadata.name,
            "split_optimization",
            request,
            graph,
            frames,
            {
                "allocations": _compact_float_dict(allocations, digits=12),
                "total_amount_out": total_amount_out,
                "truncated": truncated,
            },
        )


class ConvexSplitTraceGenerator(ConvexSplit):
    """Generate frame trace for convex split optimization."""

    def generate_trace(self, request: SwapRequest, max_frames: int = 240) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        truncated = False

        def capture(event: str, description: str, payload: Optional[Dict[str, Any]] = None) -> None:
            nonlocal truncated
            if len(frames) >= max_frames:
                truncated = True
                return
            frame = {"step": len(frames), "event": event, "description": description}
            if payload:
                frame.update(payload)
            frames.append(frame)

        pools = self.get_pools_for_pair(request.token_in, request.token_out)
        graph = _direct_pair_graph_payload(pools, request.token_in, request.token_out)
        capture("init", "Initialized Convex split optimizer.", {"pool_count": len(pools), "max_iterations": self.MAX_ITERATIONS})

        if not pools:
            capture("done", "No direct pools found.")
            return _empty_trace(
                self.metadata.name,
                "split_optimization",
                request,
                graph,
                frames,
                {"allocations": {}, "total_amount_out": 0.0, "truncated": truncated},
            )

        if len(pools) == 1:
            allocations = {pools[0].pool_id: request.amount_in}
            per_pool_results = _allocation_rows(pools, allocations, request.token_in, request.amount_in)
            capture(
                "done",
                "Single pool only; no optimization required.",
                {"final_allocations": allocations, "per_pool_results": per_pool_results, "total_amount_out": per_pool_results[0]["amount_out"]},
            )
            return _empty_trace(
                self.metadata.name,
                "split_optimization",
                request,
                graph,
                frames,
                {
                    "allocations": allocations,
                    "total_amount_out": per_pool_results[0]["amount_out"],
                    "truncated": truncated,
                },
            )

        n = len(pools)
        x = [request.amount_in / n] * n
        sample_stride = max(1, self.MAX_ITERATIONS // max(1, max_frames - 10))

        capture(
            "params",
            "Prepared convex optimization parameters.",
            {
                "learning_rate": self.LEARNING_RATE,
                "tolerance": self.TOLERANCE,
                "initial_allocations": {pools[i].pool_id: x[i] for i in range(n)},
            },
        )

        converged = False
        used_iterations = self.MAX_ITERATIONS
        for iteration in range(self.MAX_ITERATIONS):
            grad = self._compute_gradient(pools, x, request.token_in)
            x_new = [x[i] + self.LEARNING_RATE * grad[i] for i in range(n)]
            x_new = self._project_simplex(x_new, request.amount_in)
            diff_norm = math.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n)))

            if iteration == 0 or iteration == self.MAX_ITERATIONS - 1 or iteration % sample_stride == 0:
                capture(
                    "iteration",
                    "Convex gradient-ascent update.",
                    {
                        "iteration": iteration + 1,
                        "diff_norm": diff_norm,
                        "gradient": {pools[i].pool_id: grad[i] for i in range(n)},
                        "allocations": {pools[i].pool_id: x_new[i] for i in range(n)},
                        "per_pool_results": _allocation_rows(
                            pools,
                            {pools[i].pool_id: x_new[i] for i in range(n)},
                            request.token_in,
                            request.amount_in,
                        ),
                    },
                )

            x = x_new
            if diff_norm < self.TOLERANCE:
                converged = True
                used_iterations = iteration + 1
                break

        final_allocations = {
            pools[i].pool_id: x[i]
            for i in range(n)
            if x[i] > 1e-10
        }
        per_pool_results = _allocation_rows(pools, final_allocations, request.token_in, request.amount_in)
        total_amount_out = sum(item["amount_out"] for item in per_pool_results)
        capture(
            "done",
            "Convex optimization finished.",
            {
                "converged": converged,
                "used_iterations": used_iterations,
                "final_allocations": _compact_float_dict(final_allocations, digits=12),
                "per_pool_results": per_pool_results,
                "total_amount_out": total_amount_out,
            },
        )

        return _empty_trace(
            self.metadata.name,
            "split_optimization",
            request,
            graph,
            frames,
            {
                "allocations": _compact_float_dict(final_allocations, digits=12),
                "total_amount_out": total_amount_out,
                "truncated": truncated,
            },
        )


class KKTTraceGenerator(KKTOptimalSplit):
    """Generate frame trace for KKT split optimization."""

    def generate_trace(self, request: SwapRequest, max_frames: int = 180) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        truncated = False

        def capture(event: str, description: str, payload: Optional[Dict[str, Any]] = None) -> None:
            nonlocal truncated
            if len(frames) >= max_frames:
                truncated = True
                return
            frame = {
                "step": len(frames),
                "event": event,
                "description": description,
            }
            if payload:
                frame.update(payload)
            frames.append(frame)

        pools = self.get_pools_for_pair(request.token_in, request.token_out)
        graph = _direct_pair_graph_payload(pools, request.token_in, request.token_out)
        capture(
            "init",
            "Initialized KKT split optimization for direct pair pools.",
            {
                "pool_count": len(pools),
                "token_in": request.token_in,
                "token_out": request.token_out,
                "amount_in": request.amount_in,
            },
        )

        if not pools:
            capture("done", "No direct pools found for requested pair.")
            return _empty_trace(
                self.metadata.name,
                "split_optimization",
                request,
                graph,
                frames,
                {"allocations": {}, "total_amount_out": 0.0, "truncated": truncated},
            )

        params: List[Tuple[Pool, float, float, float, float]] = []
        for pool in pools:
            reserve_in, reserve_out = self._get_pool_reserves(pool, request.token_in)
            alpha = 1.0 - pool.fee
            if reserve_in <= 0 or reserve_out <= 0 or alpha <= 0:
                continue
            c = reserve_out * alpha * reserve_in
            params.append((pool, reserve_in, reserve_out, alpha, c))

        pool_payload = []
        for pool, reserve_in, reserve_out, alpha, c in params:
            pool_payload.append(
                {
                    "pool_id": pool.pool_id,
                    "dex": pool.dex,
                    "reserve_in": reserve_in,
                    "reserve_out": reserve_out,
                    "alpha": alpha,
                    "c_value": c,
                    "initial_marginal_output": reserve_out * alpha / reserve_in,
                }
            )
        capture("params", "Prepared CPMM parameters for all candidate pools.", {"pools": pool_payload})

        if not params:
            capture("done", "All candidate pools were invalid after parameter sanitization.")
            return _empty_trace(
                self.metadata.name,
                "split_optimization",
                request,
                graph,
                frames,
                {"allocations": {}, "total_amount_out": 0.0, "truncated": truncated},
            )

        max_initial_derivative = max(reserve_out * alpha / reserve_in for _, reserve_in, reserve_out, alpha, _ in params)
        low = 1e-18
        high = max_initial_derivative

        sample_stride = max(1, self.BISECTION_ITERATIONS // 45)
        for i in range(self.BISECTION_ITERATIONS):
            mid = (low + high) / 2.0
            alloc_sum = self._allocation_sum_for_lambda(params, mid)
            if alloc_sum > request.amount_in:
                low = mid
            else:
                high = mid
            if i == 0 or i == self.BISECTION_ITERATIONS - 1 or i % sample_stride == 0:
                capture(
                    "bisection",
                    "Updated lambda bounds via bisection.",
                    {
                        "iteration": i + 1,
                        "lambda_low": low,
                        "lambda_high": high,
                        "lambda_mid": mid,
                        "allocation_sum_at_mid": alloc_sum,
                        "target_total_amount": request.amount_in,
                    },
                )

        lambda_star = high
        raw_allocations: Dict[str, float] = {}
        raw_total = 0.0
        for pool, reserve_in, _, alpha, c in params:
            amount = (math.sqrt(c / lambda_star) - reserve_in) / alpha
            amount = max(0.0, amount)
            if amount > self.ALLOCATION_EPS:
                raw_allocations[pool.pool_id] = amount
                raw_total += amount

        capture(
            "raw_alloc",
            "Computed raw allocations from lambda* before budget normalization.",
            {
                "lambda_star": lambda_star,
                "raw_allocations": _compact_float_dict(raw_allocations, digits=12),
                "raw_sum": raw_total,
            },
        )

        if raw_total <= 0:
            best_pool = max(params, key=lambda item: item[2] * item[3] / item[1])[0]
            final_allocations = {best_pool.pool_id: request.amount_in}
        else:
            scale = request.amount_in / raw_total
            final_allocations = {
                pool_id: max(0.0, amount * scale)
                for pool_id, amount in raw_allocations.items()
            }

        per_pool_results = _allocation_rows(pools, final_allocations, request.token_in, request.amount_in)
        total_out = sum(item["amount_out"] for item in per_pool_results)
        capture(
            "done",
            "Final allocation and output calculation completed.",
            {
                "final_allocations": _compact_float_dict(final_allocations, digits=12),
                "per_pool_results": per_pool_results,
                "total_amount_out": total_out,
            },
        )

        return _empty_trace(
            self.metadata.name,
            "split_optimization",
            request,
            graph,
            frames,
            {
                "allocations": _compact_float_dict(final_allocations, digits=12),
                "total_amount_out": total_out,
                "truncated": truncated,
            },
        )
