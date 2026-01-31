#!/usr/bin/env python3
"""
Routing Algorithms - Unified Benchmark Suite

This project combines educational pathfinding algorithms with production-grade
DEX routing algorithms for comprehensive analysis and comparison.

Usage:
    python main.py                    # Run full benchmark suite
    python main.py --quick            # Quick educational demo
    python main.py --bellman-ford     # Bellman-Ford efficiency analysis
    python main.py --extreme          # Extreme DEX scenario testing
    python main.py --help             # Show help

Author: Routing Algorithms Research Team
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_educational_demo():
    """Run educational examples (from algo-bench)."""
    print("\n" + "=" * 70)
    print("EDUCATIONAL ALGORITHM DEMONSTRATION")
    print("=" * 70)

    from examples import educational_examples
    educational_examples.main()


def run_bellman_ford_analysis():
    """Run Bellman-Ford efficiency analysis (from algo-bench)."""
    print("\n" + "=" * 70)
    print("BELLMAN-FORD EFFICIENCY ANALYSIS")
    print("=" * 70)

    from benchmarks import bellman_ford_analysis
    bellman_ford_analysis.main()


def run_dex_benchmark():
    """Run DEX aggregator benchmarks (from dex-aggregator-algorithms)."""
    print("\n" + "=" * 70)
    print("DEX AGGREGATOR ROUTING BENCHMARK")
    print("=" * 70)

    from algorithms.base import Pool, SwapRequest
    from algorithms.single import (
        NaiveBruteForce, BFSRouting, DijkstraRouting, AStarRouting
    )
    from algorithms.composite import (
        SimpleSplit, GreedySplit, MultiHopRouting,
        DPRouting, ConvexSplit
    )
    from mock_data.data_generator import DataGenerator
    from benchmarks.benchmark_runner import BenchmarkRunner

    # Generate test pools
    generator = DataGenerator(seed=42)
    pools = generator.generate_standard_test_pools()

    print(f"\nGenerated {len(pools)} liquidity pools")

    # Define test scenarios
    scenarios = [
        ("Small Swap (1 ETH)", SwapRequest("ETH", "USDC", 1.0)),
        ("Medium Swap (10 ETH)", SwapRequest("ETH", "USDC", 10.0)),
        ("Large Swap (100 ETH)", SwapRequest("ETH", "USDC", 100.0)),
        ("Very Large Swap (500 ETH)", SwapRequest("ETH", "USDC", 500.0)),
    ]

    # Run benchmarks
    runner = BenchmarkRunner(pools)
    results = []

    for name, request in scenarios:
        print(f"\n  Running: {name}...")
        scenario_result = runner.run_scenario_benchmark(name, request)
        results.append(scenario_result)
        runner.print_results(scenario_result)

    # Get summary
    summary = runner.get_summary_statistics(results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Print summary
    sorted_algos = sorted(summary.items(),
                          key=lambda x: x[1]['avg_output'], reverse=True)

    print(f"\n{'Rank':<5} {'Algorithm':<30} {'Avg Output':>15} {'Win Rate':>10}")
    print("-" * 65)

    for rank, (name, stats) in enumerate(sorted_algos, 1):
        print(f"{rank:<5} {name:<30} {stats['avg_output']:>15.2f} "
              f"{stats['win_rate_pct']:>9.1f}%")


def run_extreme_test():
    """Run extreme DEX scenarios (5000 ETH swaps)."""
    print("\n" + "=" * 70)
    print("EXTREME SCENARIO TESTING")
    print("=" * 70)
    print("\nTesting with 5000 ETH swap to maximize algorithm differences...")

    from algorithms.base import Pool, SwapRequest
    from algorithms.composite import ConvexSplit, GreedySplit
    from algorithms.single import BFSRouting
    from mock_data.data_generator import DataGenerator

    # Generate pools
    generator = DataGenerator(seed=42)
    pools = generator.generate_standard_test_pools()

    # Extreme swap
    request = SwapRequest("ETH", "USDC", 5000.0)

    algorithms = [
        ("Convex Optimization", ConvexSplit(pools)),
        ("Greedy Split", GreedySplit(pools)),
        ("BFS Routing", BFSRouting(pools)),
    ]

    print(f"\n{'Algorithm':<30} {'Output (USDC)':>20} {'Price Impact':>15}")
    print("-" * 70)

    results = []
    for name, algo in algorithms:
        result = algo.execute_with_timing(request)
        results.append((name, result))
        print(f"{name:<30} ${result.total_amount_out:>19,.0f} "
              f"{result.total_price_impact*100:>14.2f}%")

    # Calculate differences
    best = max(results, key=lambda x: x[1].total_amount_out)
    worst = min(results, key=lambda x: x[1].total_amount_out)

    diff_amount = best[1].total_amount_out - worst[1].total_amount_out
    diff_pct = (diff_amount / worst[1].total_amount_out) * 100

    print("-" * 70)
    print(f"\nBest: {best[0]} - ${best[1].total_amount_out:,.0f}")
    print(f"Worst: {worst[0]} - ${worst[1].total_amount_out:,.0f}")
    print(f"Difference: ${diff_amount:,.0f} ({diff_pct:.2f}%)")


def print_header():
    """Print welcome header."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    Routing Algorithms Suite                           ║
║                                                                       ║
║  Educational Pathfinding + Production DEX Routing Analysis            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Routing Algorithms - Unified Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick              # Educational demo
  python main.py --bellman-ford       # BF efficiency analysis
  python main.py --extreme            # 5000 ETH extreme test
  python main.py                      # Full DEX benchmark
        """
    )

    parser.add_argument("--quick", action="store_true",
                       help="Run quick educational demonstration")
    parser.add_argument("--bellman-ford", action="store_true",
                       help="Run Bellman-Ford efficiency analysis")
    parser.add_argument("--extreme", action="store_true",
                       help="Run extreme DEX scenario testing")
    parser.add_argument("--all", action="store_true",
                       help="Run all benchmarks")

    args = parser.parse_args()

    print_header()

    if args.quick:
        run_educational_demo()
    elif args.bellman_ford:
        run_bellman_ford_analysis()
    elif args.extreme:
        run_extreme_test()
    elif args.all:
        run_educational_demo()
        run_bellman_ford_analysis()
        run_dex_benchmark()
        run_extreme_test()
    else:
        # Default: DEX benchmark
        run_dex_benchmark()

    print("\n" + "=" * 70)
    print("Thank you for using Routing Algorithms Suite!")
    print("=" * 70)


if __name__ == "__main__":
    main()
