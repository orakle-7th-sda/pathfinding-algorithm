"""
Mock Data Generator for DEX Aggregator Testing.

Generates realistic DEX pool data and swap requests for algorithm testing.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.base import Pool, SwapRequest


# Token price data (approximate USD values)
TOKEN_PRICES = {
    "ETH": 2000.0,
    "WBTC": 45000.0,
    "USDC": 1.0,
    "USDT": 1.0,
    "DAI": 1.0,
    "WETH": 2000.0,
    "LINK": 15.0,
    "UNI": 7.0,
    "AAVE": 100.0,
    "SUSHI": 1.5,
    "CRV": 0.5,
    "MKR": 1500.0,
    "COMP": 50.0,
    "SNX": 3.0,
    "YFI": 8000.0,
}

# DEX configurations
DEX_CONFIGS = {
    "uniswap_v2": {"fee": 0.003, "liquidity_multiplier": 1.0},
    "uniswap_v3": {"fee": 0.003, "liquidity_multiplier": 1.2},
    "sushiswap": {"fee": 0.003, "liquidity_multiplier": 0.7},
    "curve": {"fee": 0.0004, "liquidity_multiplier": 1.5},  # Lower fee for stablecoins
    "balancer": {"fee": 0.002, "liquidity_multiplier": 0.5},
    "pancakeswap": {"fee": 0.0025, "liquidity_multiplier": 0.4},
}


class DataGenerator:
    """
    Generates mock DEX pool data and swap requests.
    
    Creates realistic scenarios for testing routing algorithms.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.tokens = list(TOKEN_PRICES.keys())
        self.dexes = list(DEX_CONFIGS.keys())
    
    def generate_pool(self, token0: str, token1: str, dex: str,
                      base_liquidity_usd: float = 1_000_000) -> Pool:
        """
        Generate a single pool with realistic reserves.
        
        Args:
            token0: First token symbol
            token1: Second token symbol
            dex: DEX name
            base_liquidity_usd: Base liquidity in USD
            
        Returns:
            Pool object with calculated reserves
        """
        dex_config = DEX_CONFIGS.get(dex, {"fee": 0.003, "liquidity_multiplier": 1.0})
        
        # Calculate liquidity with some randomness
        liquidity_multiplier = dex_config["liquidity_multiplier"]
        random_factor = random.uniform(0.5, 1.5)
        total_liquidity = base_liquidity_usd * liquidity_multiplier * random_factor
        
        # Calculate reserves based on token prices
        price0 = TOKEN_PRICES.get(token0, 1.0)
        price1 = TOKEN_PRICES.get(token1, 1.0)
        
        # Each side gets half the liquidity in USD terms
        reserve0 = (total_liquidity / 2) / price0
        reserve1 = (total_liquidity / 2) / price1
        
        pool_id = f"{dex}_{token0}_{token1}_{random.randint(1000, 9999)}"
        
        return Pool(
            pool_id=pool_id,
            dex=dex,
            token0=token0,
            token1=token1,
            reserve0=reserve0,
            reserve1=reserve1,
            fee=dex_config["fee"]
        )
    
    def generate_pools_for_pair(self, token0: str, token1: str,
                                 num_pools: int = 3,
                                 base_liquidity_usd: float = 1_000_000) -> List[Pool]:
        """
        Generate multiple pools for a token pair across different DEXes.
        
        Args:
            token0: First token symbol
            token1: Second token symbol
            num_pools: Number of pools to generate
            base_liquidity_usd: Base liquidity in USD
            
        Returns:
            List of Pool objects
        """
        pools = []
        dexes = random.sample(self.dexes, min(num_pools, len(self.dexes)))
        
        for dex in dexes:
            pool = self.generate_pool(token0, token1, dex, base_liquidity_usd)
            pools.append(pool)
        
        return pools
    
    def generate_token_network(self, num_tokens: int = 8,
                                pools_per_pair: int = 2,
                                connection_probability: float = 0.5,
                                base_liquidity_usd: float = 1_000_000) -> List[Pool]:
        """
        Generate a network of token pools.
        
        Args:
            num_tokens: Number of tokens to include
            pools_per_pair: Number of pools per token pair
            connection_probability: Probability of creating a pool for each pair
            base_liquidity_usd: Base liquidity in USD
            
        Returns:
            List of all Pool objects
        """
        tokens = random.sample(self.tokens, min(num_tokens, len(self.tokens)))
        pools = []
        
        # Generate pools for each token pair
        for i, token0 in enumerate(tokens):
            for token1 in tokens[i+1:]:
                if random.random() < connection_probability:
                    pair_pools = self.generate_pools_for_pair(
                        token0, token1, 
                        pools_per_pair, 
                        base_liquidity_usd
                    )
                    pools.extend(pair_pools)
        
        return pools
    
    def generate_standard_test_pools(self) -> List[Pool]:
        """
        Generate a standard set of pools for testing.
        
        Includes common trading pairs with varying liquidity.
        
        Returns:
            List of Pool objects
        """
        pools = []
        
        # Major trading pairs with high liquidity
        major_pairs = [
            ("ETH", "USDC", 10_000_000),
            ("ETH", "USDT", 8_000_000),
            ("WBTC", "USDC", 5_000_000),
            ("WBTC", "ETH", 7_000_000),
            ("ETH", "DAI", 3_000_000),
        ]
        
        for token0, token1, liquidity in major_pairs:
            for dex in ["uniswap_v2", "sushiswap", "uniswap_v3"]:
                pool = self.generate_pool(token0, token1, dex, liquidity)
                pools.append(pool)
        
        # Medium liquidity pairs
        medium_pairs = [
            ("LINK", "ETH", 500_000),
            ("UNI", "ETH", 800_000),
            ("AAVE", "ETH", 400_000),
            ("LINK", "USDC", 300_000),
            ("UNI", "USDC", 600_000),
        ]
        
        for token0, token1, liquidity in medium_pairs:
            for dex in random.sample(self.dexes, 2):
                pool = self.generate_pool(token0, token1, dex, liquidity)
                pools.append(pool)
        
        # Bridge tokens for multi-hop routing
        bridge_pairs = [
            ("WETH", "ETH", 20_000_000),  # High liquidity bridge
            ("USDC", "USDT", 50_000_000),  # Stablecoin pair
            ("USDC", "DAI", 30_000_000),   # Stablecoin pair
        ]
        
        for token0, token1, liquidity in bridge_pairs:
            for dex in ["curve", "uniswap_v2"]:
                pool = self.generate_pool(token0, token1, dex, liquidity)
                pools.append(pool)
        
        return pools
    
    def generate_swap_request(self, token_in: str = "ETH",
                               token_out: str = "USDC",
                               amount_in: float = 10.0) -> SwapRequest:
        """
        Generate a swap request.
        
        Args:
            token_in: Input token symbol
            token_out: Output token symbol
            amount_in: Amount of input token
            
        Returns:
            SwapRequest object
        """
        return SwapRequest(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in
        )
    
    def generate_test_scenarios(self) -> List[Tuple[str, SwapRequest, List[Pool]]]:
        """
        Generate multiple test scenarios with varying characteristics.
        
        Returns:
            List of (scenario_name, swap_request, pools) tuples
        """
        scenarios = []
        
        # Scenario 1: Small swap, high liquidity
        pools1 = self.generate_standard_test_pools()
        request1 = SwapRequest(token_in="ETH", token_out="USDC", amount_in=1.0)
        scenarios.append(("Small Swap (1 ETH)", request1, pools1))
        
        # Scenario 2: Medium swap
        pools2 = self.generate_standard_test_pools()
        request2 = SwapRequest(token_in="ETH", token_out="USDC", amount_in=10.0)
        scenarios.append(("Medium Swap (10 ETH)", request2, pools2))
        
        # Scenario 3: Large swap (high price impact)
        pools3 = self.generate_standard_test_pools()
        request3 = SwapRequest(token_in="ETH", token_out="USDC", amount_in=100.0)
        scenarios.append(("Large Swap (100 ETH)", request3, pools3))
        
        # Scenario 4: Very large swap (extreme price impact)
        pools4 = self.generate_standard_test_pools()
        request4 = SwapRequest(token_in="ETH", token_out="USDC", amount_in=500.0)
        scenarios.append(("Very Large Swap (500 ETH)", request4, pools4))
        
        # Scenario 5: Multi-hop required (no direct pool)
        pools5 = [
            self.generate_pool("ETH", "LINK", "uniswap_v2", 1_000_000),
            self.generate_pool("ETH", "LINK", "sushiswap", 800_000),
            self.generate_pool("LINK", "USDC", "uniswap_v2", 500_000),
            self.generate_pool("LINK", "USDC", "balancer", 400_000),
            self.generate_pool("ETH", "UNI", "uniswap_v2", 600_000),
            self.generate_pool("UNI", "USDC", "uniswap_v2", 400_000),
        ]
        request5 = SwapRequest(token_in="ETH", token_out="USDC", amount_in=10.0)
        scenarios.append(("Multi-hop Required", request5, pools5))
        
        # Scenario 6: Low liquidity
        pools6 = self.generate_token_network(
            num_tokens=5,
            pools_per_pair=1,
            base_liquidity_usd=100_000
        )
        # Add some ETH-USDC pools
        pools6.extend([
            self.generate_pool("ETH", "USDC", "uniswap_v2", 100_000),
            self.generate_pool("ETH", "USDC", "sushiswap", 80_000),
        ])
        request6 = SwapRequest(token_in="ETH", token_out="USDC", amount_in=50.0)
        scenarios.append(("Low Liquidity Environment", request6, pools6))
        
        return scenarios
    
    def get_mock_input_output_example(self) -> Dict:
        """
        Get a complete example with mock input and expected output format.
        
        Returns:
            Dictionary with input/output example
        """
        return {
            "input": {
                "pools": [
                    {
                        "pool_id": "uniswap_v2_ETH_USDC_1234",
                        "dex": "uniswap_v2",
                        "token0": "ETH",
                        "token1": "USDC",
                        "reserve0": 5000.0,
                        "reserve1": 10_000_000.0,
                        "fee": 0.003
                    },
                    {
                        "pool_id": "sushiswap_ETH_USDC_5678",
                        "dex": "sushiswap",
                        "token0": "ETH",
                        "token1": "USDC",
                        "reserve0": 3000.0,
                        "reserve1": 6_000_000.0,
                        "fee": 0.003
                    }
                ],
                "swap_request": {
                    "token_in": "ETH",
                    "token_out": "USDC",
                    "amount_in": 10.0
                }
            },
            "expected_output": {
                "algorithm": "Greedy Split",
                "total_amount_in": 10.0,
                "total_amount_out": 19850.5,
                "price_impact": 0.002,
                "effective_price": 1985.05,
                "execution_time_ms": 0.5,
                "gas_estimate": 250000,
                "routes": [
                    {
                        "path": ["ETH", "USDC"],
                        "dexes": ["uniswap_v2"],
                        "amount_in": 6.0,
                        "amount_out": 11910.3
                    },
                    {
                        "path": ["ETH", "USDC"],
                        "dexes": ["sushiswap"],
                        "amount_in": 4.0,
                        "amount_out": 7940.2
                    }
                ]
            }
        }


def create_sample_data():
    """Create and return sample data for testing."""
    generator = DataGenerator(seed=42)
    
    pools = generator.generate_standard_test_pools()
    request = generator.generate_swap_request(
        token_in="ETH",
        token_out="USDC",
        amount_in=10.0
    )
    
    return pools, request


if __name__ == "__main__":
    # Example usage
    generator = DataGenerator(seed=42)
    
    print("=" * 60)
    print("DEX Aggregator Mock Data Generator")
    print("=" * 60)
    
    # Generate standard pools
    pools = generator.generate_standard_test_pools()
    print(f"\nGenerated {len(pools)} pools")
    
    # Show sample pools
    print("\nSample pools:")
    for pool in pools[:5]:
        print(f"  - {pool.pool_id}")
        print(f"    {pool.token0}/{pool.token1} on {pool.dex}")
        print(f"    Reserves: {pool.reserve0:.2f} / {pool.reserve1:.2f}")
        print(f"    Fee: {pool.fee * 100:.2f}%")
    
    # Generate test scenarios
    scenarios = generator.generate_test_scenarios()
    print(f"\nGenerated {len(scenarios)} test scenarios:")
    for name, request, _ in scenarios:
        print(f"  - {name}: {request.amount_in} {request.token_in} -> {request.token_out}")
    
    # Show input/output example
    example = generator.get_mock_input_output_example()
    print("\nMock Input/Output Example:")
    print(f"  Input: {example['input']['swap_request']}")
    print(f"  Expected Output Amount: {example['expected_output']['total_amount_out']}")
