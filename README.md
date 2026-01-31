# Routing Algorithms

Educational pathfinding algorithms + Production DEX routing benchmarks.

## Quick Start

```bash
cd routing-algorithms

# Educational demo (10s)
python3 main.py --quick

# Bellman-Ford analysis (2-3min)
python3 main.py --bellman-ford

# DEX benchmark (1min, default)
python3 main.py

# Extreme test (30s)
python3 main.py --extreme

# All benchmarks (5min)
python3 main.py --all
```

## Highlights

- **15+ algorithms**: Aggregation, Single-path, Multi-path, DEX-specific
- **17 benchmarks**: Bellman-Ford (12) + DEX (5)
- **Proven results**: Dijkstra 219x faster than Bellman-Ford
- **Real impact**: Convex Optimization yields 66% ($2.4M) better results

## Project Structure

```
routing-algorithms/
├── main.py              # Unified CLI
├── algorithms/          # Algorithm implementations (OOP)
├── benchmarks/          # Benchmark tools
├── examples/            # Educational examples
├── docs/                # Documentation (13 files)
└── output/              # Results & charts
```

## Documentation

All docs in `routing-algorithms/docs/`:

- [Quick Start](routing-algorithms/docs/quick-start.md)
- [Documentation Index](routing-algorithms/docs/README.md)
- [Project Overview](routing-algorithms/docs/01-project-overview.md)

## Key Results

**Bellman-Ford Analysis:**

- Dijkstra: Up to **219x faster** ⚡⚡
- A\*: Up to **187x faster** ⚡

**DEX Extreme Test (5000 ETH):**

- Convex Optimization: **+66% output** ($2.4M difference)

## Requirements

- Python 3.8+
- Optional: `matplotlib`, `numpy`, `scipy`

```bash
pip install -r routing-algorithms/requirements.txt
```

## License

MIT
