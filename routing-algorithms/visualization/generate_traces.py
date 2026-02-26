#!/usr/bin/env python3
"""
Generate replay traces and optionally export presentation media.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from algorithms.base import Pool, SwapRequest
from mock_data.aggregator_data_generator import AggregatorDataGenerator
from mock_data.data_generator import DataGenerator
from visualization.export_media import export_trace_media_bundle
from visualization.trace_generators import (
    BellmanFordTraceGenerator,
    ConvexSplitTraceGenerator,
    DijkstraTraceGenerator,
    GreedySplitTraceGenerator,
    KBestTraceGenerator,
    KKTTraceGenerator,
)


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pool_from_dict(raw: Dict) -> Pool:
    return Pool(
        pool_id=raw["pool_id"],
        dex=raw["dex"],
        token0=raw["token0"],
        token1=raw["token1"],
        reserve0=float(raw["reserve0"]),
        reserve1=float(raw["reserve1"]),
        fee=float(raw["fee"]),
    )


def _load_enhanced_scenarios(seed: int) -> List[Dict]:
    payload = AggregatorDataGenerator(seed=seed).generate_dataset()
    return payload.get("scenarios", [])


def _pick_scenario(
    scenarios: List[Dict], scenario_name: Optional[str], scenario_index: int
) -> Tuple[Dict, int]:
    if not scenarios:
        raise ValueError("No scenarios available in enhanced dataset.")

    if scenario_name:
        lowered = scenario_name.strip().lower()
        for idx, scenario in enumerate(scenarios):
            if scenario.get("scenario_name", "").strip().lower() == lowered:
                return scenario, idx
        raise ValueError(f"Scenario not found: {scenario_name!r}")

    if scenario_index < 0 or scenario_index >= len(scenarios):
        raise ValueError(
            f"scenario-index out of range: {scenario_index}. "
            f"Valid range: 0..{len(scenarios)-1}"
        )
    return scenarios[scenario_index], scenario_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate replay traces for visualization.")
    parser.add_argument("--token-in")
    parser.add_argument("--token-out")
    parser.add_argument("--amount", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        choices=["standard", "enhanced"],
        default="standard",
        help="Input dataset profile for trace generation.",
    )
    parser.add_argument(
        "--scenario-name",
        help="Scenario name when --dataset enhanced is used.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Scenario index when --dataset enhanced is used and --scenario-name is omitted.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available enhanced scenarios and exit.",
    )
    parser.add_argument(
        "--amount-multiplier",
        type=float,
        default=1.0,
        help="Multiply input amount to stress slippage/route divergence.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "output", "traces"),
        help="Directory where trace JSON files are written.",
    )
    parser.add_argument("--dijkstra-frames", type=int, default=260)
    parser.add_argument("--bellman-frames", type=int, default=320)
    parser.add_argument("--kbest-frames", type=int, default=260)
    parser.add_argument("--greedy-frames", type=int, default=220)
    parser.add_argument("--convex-frames", type=int, default=240)
    parser.add_argument("--kkt-frames", type=int, default=180)
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Skip automatic GIF/MP4 export.",
    )
    parser.add_argument(
        "--media-fps",
        type=int,
        default=2,
        help="FPS for exported GIF/MP4.",
    )
    parser.add_argument(
        "--media-layout",
        choices=["flow", "circular"],
        default="flow",
        help="Graph layout for exported GIF/MP4.",
    )
    parser.add_argument(
        "--media-dir",
        default=os.path.join(ROOT_DIR, "output", "media"),
        help="Directory where GIF/MP4 files are written.",
    )
    args = parser.parse_args()

    if args.dataset == "enhanced":
        scenarios = _load_enhanced_scenarios(args.seed)
        if args.list_scenarios:
            print("Available enhanced scenarios:")
            for idx, s in enumerate(scenarios):
                req = s.get("swap_request", {})
                print(
                    f"[{idx}] {s.get('scenario_name')} | "
                    f"{req.get('token_in')}->{req.get('token_out')} amount={req.get('amount_in')}"
                )
            return

        scenario, picked_idx = _pick_scenario(
            scenarios, args.scenario_name, args.scenario_index
        )
        pools = [_pool_from_dict(raw) for raw in scenario["pools"]]
        base_req = scenario["swap_request"]

        token_in = args.token_in or base_req["token_in"]
        token_out = args.token_out or base_req["token_out"]
        base_amount = float(args.amount) if args.amount is not None else float(
            base_req["amount_in"]
        )
        request = SwapRequest(token_in, token_out, base_amount * args.amount_multiplier)

        print(
            f"Using enhanced scenario[{picked_idx}]: {scenario.get('scenario_name')} "
            f"(pools={len(pools)})"
        )
    else:
        pools = DataGenerator(seed=args.seed).generate_standard_test_pools()
        token_in = args.token_in or "ETH"
        token_out = args.token_out or "USDC"
        base_amount = float(args.amount) if args.amount is not None else 250.0
        request = SwapRequest(token_in, token_out, base_amount * args.amount_multiplier)

    print(
        f"Trace request: {request.token_in}->{request.token_out} amount={request.amount_in} "
        f"(dataset={args.dataset}, seed={args.seed}, pools={len(pools)})"
    )

    dijkstra_trace = DijkstraTraceGenerator(pools).generate_trace(request, max_frames=args.dijkstra_frames)
    bellman_trace = BellmanFordTraceGenerator(pools).generate_trace(request, max_frames=args.bellman_frames)
    kbest_trace = KBestTraceGenerator(pools).generate_trace(request, max_frames=args.kbest_frames)
    greedy_trace = GreedySplitTraceGenerator(pools).generate_trace(request, max_frames=args.greedy_frames)
    convex_trace = ConvexSplitTraceGenerator(pools).generate_trace(request, max_frames=args.convex_frames)
    kkt_trace = KKTTraceGenerator(pools).generate_trace(request, max_frames=args.kkt_frames)

    dijkstra_path = os.path.join(args.output_dir, "dijkstra_trace.json")
    bellman_path = os.path.join(args.output_dir, "bellman_ford_trace.json")
    kbest_path = os.path.join(args.output_dir, "kbest_trace.json")
    greedy_path = os.path.join(args.output_dir, "greedy_split_trace.json")
    convex_path = os.path.join(args.output_dir, "convex_split_trace.json")
    kkt_path = os.path.join(args.output_dir, "kkt_split_trace.json")
    index_path = os.path.join(args.output_dir, "index.json")

    _write_json(dijkstra_path, dijkstra_trace)
    _write_json(bellman_path, bellman_trace)
    _write_json(kbest_path, kbest_trace)
    _write_json(greedy_path, greedy_trace)
    _write_json(convex_path, convex_trace)
    _write_json(kkt_path, kkt_trace)
    _write_json(
        index_path,
        {
            "traces": [
                {
                    "name": "Dijkstra Routing (Graph Search)",
                    "path": "../output/traces/dijkstra_trace.json",
                    "algorithm": dijkstra_trace["metadata"]["algorithm"],
                    "kind": dijkstra_trace["metadata"]["trace_kind"],
                },
                {
                    "name": "Bellman-Ford Routing (Negative-Weight Safe)",
                    "path": "../output/traces/bellman_ford_trace.json",
                    "algorithm": bellman_trace["metadata"]["algorithm"],
                    "kind": bellman_trace["metadata"]["trace_kind"],
                },
                {
                    "name": "K-Best Routing (Candidate Re-ranking)",
                    "path": "../output/traces/kbest_trace.json",
                    "algorithm": kbest_trace["metadata"]["algorithm"],
                    "kind": kbest_trace["metadata"]["trace_kind"],
                },
                {
                    "name": "Greedy Split (Chunk Allocation)",
                    "path": "../output/traces/greedy_split_trace.json",
                    "algorithm": greedy_trace["metadata"]["algorithm"],
                    "kind": greedy_trace["metadata"]["trace_kind"],
                },
                {
                    "name": "Convex Split (Gradient Optimization)",
                    "path": "../output/traces/convex_split_trace.json",
                    "algorithm": convex_trace["metadata"]["algorithm"],
                    "kind": convex_trace["metadata"]["trace_kind"],
                },
                {
                    "name": "KKT Optimal Split (Optimization)",
                    "path": "../output/traces/kkt_split_trace.json",
                    "algorithm": kkt_trace["metadata"]["algorithm"],
                    "kind": kkt_trace["metadata"]["trace_kind"],
                },
            ]
        },
    )

    print(f"Wrote trace files to: {args.output_dir}")
    print(f"- {dijkstra_path}")
    print(f"- {bellman_path}")
    print(f"- {kbest_path}")
    print(f"- {greedy_path}")
    print(f"- {convex_path}")
    print(f"- {kkt_path}")
    print(f"- {index_path}")

    if not args.skip_media:
        print("")
        print(f"Exporting media to: {args.media_dir}")
        media_result = export_trace_media_bundle(
            traces=[
                {"name": "dijkstra_trace", "trace": dijkstra_trace},
                {"name": "bellman_ford_trace", "trace": bellman_trace},
                {"name": "kbest_trace", "trace": kbest_trace},
                {"name": "greedy_split_trace", "trace": greedy_trace},
                {"name": "convex_split_trace", "trace": convex_trace},
                {"name": "kkt_split_trace", "trace": kkt_trace},
            ],
            output_dir=args.media_dir,
            fps=args.media_fps,
            layout=args.media_layout,
        )
        if media_result["ok"]:
            for item in media_result["files"]:
                print(f"- GIF: {item['gif']}")
                if item.get("mp4"):
                    print(f"  MP4: {item['mp4']}")
                elif item.get("mp4_error"):
                    print(f"  MP4: {item['mp4_error']}")
        else:
            print(f"- Media export skipped: {media_result['reason']}")

    print("")
    print("Next:")
    print("1) python3 -m http.server 8000")
    print("2) Open http://localhost:8000/visualization/replay_viewer.html")


if __name__ == "__main__":
    main()
