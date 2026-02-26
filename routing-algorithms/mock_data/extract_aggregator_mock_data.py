#!/usr/bin/env python3
"""
Extract an improved mock dataset for DEX aggregator testing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mock_data.aggregator_data_generator import AggregatorDataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract enhanced mock data scenarios.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="output/mock-data/aggregator_enhanced_dataset.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = AggregatorDataGenerator(seed=args.seed)
    generator.export_dataset(str(output_path))

    print(f"Enhanced dataset written: {output_path.resolve()}")
    print("Scenarios included:")
    print("- Baseline Deep Liquidity")
    print("- Fragmented Market")
    print("- Stable Depeg Down")
    print("- Low Liquidity Stress")
    print("- Multi-Hop Emphasis")


if __name__ == "__main__":
    main()
