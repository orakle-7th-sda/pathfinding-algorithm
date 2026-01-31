#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
어그리게이션 알고리즘 비교 차트 (Basic vs Optimized)
=============================================================================
- 같은 연산에 대해 Basic(2패스/단순) vs Optimized(1패스) 구현 비교
- 입력 크기(n)를 키우면서 실행 시간 측정 → Basic 대비 Optimized 이득 시각화
- 연산: Variance(분산), MinMax(최소·최대), Group By Sum(그룹별 합계)
- 실행: python3 chart_aggregation.py
=============================================================================
"""

import time
import random
import csv
import os
from typing import List, Dict, Tuple, Optional

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


# =============================================================================
# 1. Variance (분산)
# =============================================================================


def variance_basic(data: List[float]) -> float:
    """Basic: 2패스 — 1패스에 평균, 2패스에 편차 제곱 합."""
    if not data or len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    sq_diff = sum((x - mean) ** 2 for x in data)
    return sq_diff / (len(data) - 1)


def variance_optimized(data: List[float]) -> float:
    """Optimized: 1패스 — Welford 온라인 분산 알고리즘."""
    if not data or len(data) < 2:
        return 0.0
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    return m2 / (n - 1)


# =============================================================================
# 2. Min & Max
# =============================================================================


def minmax_basic(data: List[float]) -> Tuple[float, float]:
    """Basic: 2패스 — 1패스에 min, 1패스에 max."""
    if not data:
        return (0.0, 0.0)
    min_val = min(data)
    max_val = max(data)
    return (min_val, max_val)


def minmax_optimized(data: List[float]) -> Tuple[float, float]:
    """Optimized: 1패스 — 한 루프에서 min, max 동시 계산."""
    if not data:
        return (0.0, 0.0)
    min_val = max_val = data[0]
    for x in data[1:]:
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    return (min_val, max_val)


# =============================================================================
# 3. Group By Sum
# =============================================================================


def group_by_sum_basic(records: List[Tuple[str, float]]) -> Dict[str, float]:
    """Basic: 2패스 — 1패스에 키별 리스트 수집, 2패스에 각 리스트 합."""
    from collections import defaultdict

    by_key: Dict[str, List[float]] = defaultdict(list)
    for key, value in records:
        by_key[key].append(value)
    return {k: sum(v) for k, v in by_key.items()}


def group_by_sum_optimized(records: List[Tuple[str, float]]) -> Dict[str, float]:
    """Optimized: 1패스 — 딕셔너리로 한 번에 합계 누적."""
    from collections import defaultdict

    result = defaultdict(float)
    for key, value in records:
        result[key] += value
    return dict(result)


# =============================================================================
# Mock 데이터 생성
# =============================================================================


def make_data(n: int, seed: Optional[int] = None) -> List[float]:
    """길이 n인 실수 리스트 (0~100 균일)."""
    if seed is not None:
        random.seed(seed)
    return [random.uniform(0, 100) for _ in range(n)]


def make_records(
    n: int, num_keys: int = 50, seed: Optional[int] = None
) -> List[Tuple[str, float]]:
    """길이 n인 (키, 값) 리스트. 키는 0..num_keys-1 문자열."""
    if seed is not None:
        random.seed(seed)
    return [
        (str(random.randint(0, num_keys - 1)), random.uniform(0, 100)) for _ in range(n)
    ]


# =============================================================================
# 실행 시간 측정
# =============================================================================


def measure_time(fn, *args, n_runs: int = 5, **kwargs) -> float:
    """fn을 n_runs회 실행 후 평균 시간(초)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


# =============================================================================
# 벤치마크 실행
# =============================================================================


def run_variance_benchmark(
    sizes: List[int],
) -> Tuple[List[int], List[float], List[float]]:
    """입력 크기별 Variance Basic vs Optimized 평균 시간."""
    basic_times = []
    opt_times = []
    for n in sizes:
        data = make_data(n, seed=42)
        basic_times.append(measure_time(variance_basic, data))
        opt_times.append(measure_time(variance_optimized, data))
    return (sizes, basic_times, opt_times)


def run_minmax_benchmark(
    sizes: List[int],
) -> Tuple[List[int], List[float], List[float]]:
    """입력 크기별 MinMax Basic vs Optimized 평균 시간."""
    basic_times = []
    opt_times = []
    for n in sizes:
        data = make_data(n, seed=42)
        basic_times.append(measure_time(minmax_basic, data))
        opt_times.append(measure_time(minmax_optimized, data))
    return (sizes, basic_times, opt_times)


def run_group_by_benchmark(
    sizes: List[int],
) -> Tuple[List[int], List[float], List[float]]:
    """입력 크기별 Group By Sum Basic vs Optimized 평균 시간."""
    basic_times = []
    opt_times = []
    for n in sizes:
        records = make_records(n, num_keys=50, seed=42)
        basic_times.append(measure_time(group_by_sum_basic, records))
        opt_times.append(measure_time(group_by_sum_optimized, records))
    return (sizes, basic_times, opt_times)


# =============================================================================
# 차트 그리기
# =============================================================================


def plot_aggregation(
    sizes: List[int],
    basic_times: List[float],
    opt_times: List[float],
    title: str,
    ax,
):
    """One subplot: Basic vs Optimized two lines."""
    ax.plot(sizes, basic_times, marker="o", label="Basic (2-pass)", color="C0")
    ax.plot(sizes, opt_times, marker="s", label="Optimized (1-pass)", color="C1")
    ax.set_xlabel("Input size (n)")
    ax.set_ylabel("Time (sec)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    if not HAS_MATPLOTLIB:
        print(
            "matplotlib 없음. 차트 대신 CSV만 저장합니다. PNG 차트를 보려면: pip install matplotlib"
        )
    # 입력 크기: 1만 ~ 50만 (너무 크면 오래 걸리므로 적당히)
    sizes = [10_000, 50_000, 100_000, 200_000, 500_000]
    print("어그리게이션 벤치마크 실행 중 (크기:", sizes, ")...")

    # Variance
    xs, basic_v, opt_v = run_variance_benchmark(sizes)
    # MinMax
    _, basic_m, opt_m = run_minmax_benchmark(sizes)
    # Group By Sum
    _, basic_g, opt_g = run_group_by_benchmark(sizes)

    out_csv = os.path.join(DATA_DIR, "chart_aggregation.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "size",
                "Variance_Basic",
                "Variance_Opt",
                "MinMax_Basic",
                "MinMax_Opt",
                "GroupBy_Basic",
                "GroupBy_Opt",
            ]
        )
        for i in range(len(xs)):
            w.writerow(
                [
                    xs[i],
                    basic_v[i],
                    opt_v[i],
                    basic_m[i],
                    opt_m[i],
                    basic_g[i],
                    opt_g[i],
                ]
            )
    print(f"데이터 저장: {out_csv}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        plot_aggregation(xs, basic_v, opt_v, "Variance", axes[0])
        plot_aggregation(xs, basic_m, opt_m, "Min & Max", axes[1])
        plot_aggregation(xs, basic_g, opt_g, "Group By Sum", axes[2])
        fig.suptitle(
            "Aggregation: Basic (2-pass) vs Optimized (1-pass) — Time", fontsize=12
        )
        plt.tight_layout()
        out = os.path.join(CHART_DIR, "chart_aggregation.png")
        os.makedirs(CHART_DIR, exist_ok=True)
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"차트 저장: {out}")
    print("완료.")


if __name__ == "__main__":
    main()
