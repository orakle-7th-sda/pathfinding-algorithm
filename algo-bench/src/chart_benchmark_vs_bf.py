#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Bellman-Ford 벤치마크 결과 차트
=============================================================================
- benchmark_vs_bellman_ford.csv를 읽어,
  x=시나리오 ID (S1~S12), y=mean_time_sec, 알고리즘별 선 그래프 생성.
- 실행: python3 chart_benchmark_vs_bf.py
- 출력: benchmark_vs_bellman_ford.png (같은 디렉터리)
=============================================================================
"""

import csv
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
CHART_DIR = os.path.join(ROOT, "chart")
CSV_PATH = os.path.join(DATA_DIR, "benchmark_vs_bellman_ford.csv")
OUT_PATH = os.path.join(CHART_DIR, "benchmark_vs_bellman_ford.png")


def load_csv(path: str):
    """CSV에서 (scenario_id, algorithm) -> mean_time_sec 로드."""
    data = defaultdict(dict)  # scenario_id -> { algorithm -> mean_time_sec }
    scenario_order = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["scenario_id"]
            algo = row["algorithm"]
            try:
                t = float(row["mean_time_sec"])
            except ValueError:
                t = 0.0
            if sid not in data:
                scenario_order.append(sid)
            data[sid][algo] = t
    return data, scenario_order


def plot_chart(data, scenario_order, out_path: str):
    """x=scenario_id, y=mean_time_sec, 알고리즘별 선. x는 0..N-1로 균등 간격."""
    algorithms = [
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
    # x를 0, 1, 2, ... 로 두어 균등 간격으로 연결 (선이 덜 울퉁불퉁해 보이도록)
    x_numeric = list(range(len(scenario_order)))
    x_labels = scenario_order
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        ys = [data[sid].get(algo, 0) for sid in x_labels]
        plt.plot(x_numeric, ys, marker="o", markersize=4, label=algo)
    plt.xticks(x_numeric, x_labels, rotation=45)
    plt.xlabel("Scenario ID (S1–S12: each scenario has different V, E, weight type)")
    plt.ylabel("Mean execution time (sec)")
    plt.title("Benchmark vs Bellman-Ford: Mean time by scenario and algorithm")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out_path}")


def main():
    if not HAS_MATPLOTLIB:
        print("matplotlib not found. Install with: pip install matplotlib")
        return
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}. Run benchmark_vs_bellman_ford.py first.")
        return
    data, scenario_order = load_csv(CSV_PATH)
    plot_chart(data, scenario_order, OUT_PATH)


if __name__ == "__main__":
    main()
