"""
Enhanced mock data generator for DEX aggregator studies.

Focuses on realistic routing stress:
- fragmented prices across DEXes
- stablecoin depeg events
- mixed liquidity depth (deep + toxic pools)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from algorithms.base import Pool, SwapRequest


TOKEN_PRICES_USD: Dict[str, float] = {
    "ETH": 2000.0,
    "WBTC": 45000.0,
    "USDC": 1.0,
    "USDT": 1.0,
    "DAI": 1.0,
    "USDe": 1.0,
    "USDS": 1.0,
    "USD1": 1.0,
    "WETH": 2000.0,
    "LINK": 15.0,
    "UNI": 7.0,
    "AAVE": 100.0,
}

STABLE_TOKENS = {"USDC", "USDT", "DAI", "USDe", "USDS", "USD1"}

DEX_CONFIGS = {
    "uniswap_v2": {"fee": 0.0030, "liquidity_multiplier": 1.00},
    "uniswap_v3": {"fee": 0.0030, "liquidity_multiplier": 1.25},
    "sushiswap": {"fee": 0.0030, "liquidity_multiplier": 0.70},
    "curve": {"fee": 0.0004, "liquidity_multiplier": 1.60},
    "balancer": {"fee": 0.0020, "liquidity_multiplier": 0.55},
}


@dataclass
class MarketProfile:
    name: str
    base_price_jitter_pct: float
    dex_fragmentation_pct: float
    stable_depeg_token: Optional[str] = None
    stable_depeg_pct: float = 0.0
    liquidity_scale: float = 1.0
    include_toxic_pools: bool = True


class AggregatorDataGenerator:
    """Scenario-driven mock dataset generator for routing algorithms."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.dexes = list(DEX_CONFIGS.keys())

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _pair_mid_price(self, token0: str, token1: str) -> float:
        p0 = TOKEN_PRICES_USD[token0]
        p1 = TOKEN_PRICES_USD[token1]
        return p0 / p1

    def _dex_price_skew(self, profile: MarketProfile, token0: str, token1: str) -> float:
        # Base random jitter across the full market.
        base = self.rng.uniform(-profile.base_price_jitter_pct, profile.base_price_jitter_pct)
        # Additional per-DEX fragmentation.
        fragmented = self.rng.uniform(
            -profile.dex_fragmentation_pct, profile.dex_fragmentation_pct
        )

        depeg = 0.0
        if profile.stable_depeg_token is not None:
            # If one side includes depeg token against other stable,
            # force a stronger directional shift.
            if profile.stable_depeg_token == token0 and token1 in STABLE_TOKENS:
                depeg = profile.stable_depeg_pct
            elif profile.stable_depeg_token == token1 and token0 in STABLE_TOKENS:
                depeg = -profile.stable_depeg_pct

        return base + fragmented + depeg

    def _make_pool(
        self,
        token0: str,
        token1: str,
        dex: str,
        base_liquidity_usd: float,
        profile: MarketProfile,
        fee_override: Optional[float] = None,
        pool_tag: str = "main",
    ) -> Pool:
        dex_cfg = DEX_CONFIGS[dex]
        liquidity = (
            base_liquidity_usd
            * profile.liquidity_scale
            * dex_cfg["liquidity_multiplier"]
            * self.rng.uniform(0.8, 1.2)
        )

        mid = self._pair_mid_price(token0, token1)
        skew = self._dex_price_skew(profile, token0, token1)
        target_price = mid * (1.0 + skew)

        # Set reserves so reserve1/reserve0 ~= target_price then rescale to target USD depth.
        reserve0 = (liquidity / 2.0) / TOKEN_PRICES_USD[token0]
        reserve1 = reserve0 * target_price
        current_usd = reserve0 * TOKEN_PRICES_USD[token0] + reserve1 * TOKEN_PRICES_USD[token1]
        if current_usd > 0:
            scale = liquidity / current_usd
            reserve0 *= scale
            reserve1 *= scale

        fee = dex_cfg["fee"] if fee_override is None else fee_override
        pool_id = (
            f"{dex}_{token0}_{token1}_{pool_tag}_{self.seed}_{self.rng.randint(1000, 9999)}"
        )
        return Pool(
            pool_id=pool_id,
            dex=dex,
            token0=token0,
            token1=token1,
            reserve0=reserve0,
            reserve1=reserve1,
            fee=fee,
        )

    def generate_profile_pools(self, profile: MarketProfile) -> List[Pool]:
        pools: List[Pool] = []

        major_pairs = [
            ("ETH", "USDC", 12_000_000),
            ("ETH", "USDT", 9_000_000),
            ("ETH", "DAI", 4_000_000),
            ("WBTC", "ETH", 7_500_000),
            ("WBTC", "USDC", 6_000_000),
        ]
        stable_pairs = [
            ("USDC", "USDT", 70_000_000),
            ("USDC", "DAI", 45_000_000),
            ("USDe", "USDT", 20_000_000),
            ("USDS", "USDC", 16_000_000),
            ("USD1", "USDC", 12_000_000),
        ]
        bridge_pairs = [
            ("WETH", "ETH", 30_000_000),
            ("LINK", "ETH", 800_000),
            ("UNI", "ETH", 900_000),
            ("AAVE", "ETH", 650_000),
            ("LINK", "USDC", 550_000),
            ("UNI", "USDC", 700_000),
        ]

        for token0, token1, liq in major_pairs:
            for dex in ["uniswap_v2", "sushiswap", "uniswap_v3"]:
                pools.append(self._make_pool(token0, token1, dex, liq, profile))

        for token0, token1, liq in stable_pairs:
            for dex in ["curve", "uniswap_v3", "uniswap_v2"]:
                fee_override = 0.0004 if dex == "curve" else None
                pools.append(self._make_pool(token0, token1, dex, liq, profile, fee_override=fee_override))

        for token0, token1, liq in bridge_pairs:
            for dex in self.rng.sample(self.dexes, 2):
                pools.append(self._make_pool(token0, token1, dex, liq, profile))

        if profile.include_toxic_pools:
            # Add low-liquidity outlier pools with bad pricing.
            toxic_pairs = [("ETH", "USDC"), ("USDe", "USDT"), ("LINK", "USDC")]
            for token0, token1 in toxic_pairs:
                for dex in ["balancer", "sushiswap"]:
                    toxic_profile = MarketProfile(
                        name=f"{profile.name}-toxic",
                        base_price_jitter_pct=profile.base_price_jitter_pct,
                        dex_fragmentation_pct=max(0.03, profile.dex_fragmentation_pct * 2.0),
                        stable_depeg_token=profile.stable_depeg_token,
                        stable_depeg_pct=profile.stable_depeg_pct,
                        liquidity_scale=max(0.05, profile.liquidity_scale * 0.15),
                        include_toxic_pools=False,
                    )
                    pools.append(
                        self._make_pool(
                            token0,
                            token1,
                            dex,
                            base_liquidity_usd=120_000,
                            profile=toxic_profile,
                            pool_tag="toxic",
                        )
                    )

        return pools

    def generate_dataset(self) -> Dict:
        scenarios = [
            {
                "name": "Baseline Deep Liquidity",
                "profile": MarketProfile(
                    name="baseline",
                    base_price_jitter_pct=0.0015,
                    dex_fragmentation_pct=0.0010,
                    liquidity_scale=1.0,
                    include_toxic_pools=True,
                ),
                "request": SwapRequest("ETH", "USDC", 120.0),
            },
            {
                "name": "Fragmented Market",
                "profile": MarketProfile(
                    name="fragmented",
                    base_price_jitter_pct=0.0030,
                    dex_fragmentation_pct=0.0150,
                    liquidity_scale=0.9,
                    include_toxic_pools=True,
                ),
                "request": SwapRequest("ETH", "USDC", 450.0),
            },
            {
                "name": "Stable Depeg Down",
                "profile": MarketProfile(
                    name="depeg-down",
                    base_price_jitter_pct=0.0020,
                    dex_fragmentation_pct=0.0080,
                    stable_depeg_token="USDe",
                    stable_depeg_pct=-0.025,
                    liquidity_scale=0.95,
                    include_toxic_pools=True,
                ),
                "request": SwapRequest("USDe", "USDT", 350_000.0),
            },
            {
                "name": "Low Liquidity Stress",
                "profile": MarketProfile(
                    name="low-liquidity",
                    base_price_jitter_pct=0.0040,
                    dex_fragmentation_pct=0.0200,
                    liquidity_scale=0.35,
                    include_toxic_pools=True,
                ),
                "request": SwapRequest("ETH", "USDC", 500.0),
            },
            {
                "name": "Multi-Hop Emphasis",
                "profile": MarketProfile(
                    name="multihop",
                    base_price_jitter_pct=0.0020,
                    dex_fragmentation_pct=0.0100,
                    liquidity_scale=0.75,
                    include_toxic_pools=True,
                ),
                "request": SwapRequest("AAVE", "USDT", 900.0),
            },
        ]

        payload = {
            "generated_at_utc": self._now_iso(),
            "seed": self.seed,
            "scenarios": [],
        }

        for item in scenarios:
            profile: MarketProfile = item["profile"]
            pools = self.generate_profile_pools(profile)
            request: SwapRequest = item["request"]

            payload["scenarios"].append(
                {
                    "scenario_name": item["name"],
                    "profile": {
                        "name": profile.name,
                        "base_price_jitter_pct": profile.base_price_jitter_pct,
                        "dex_fragmentation_pct": profile.dex_fragmentation_pct,
                        "stable_depeg_token": profile.stable_depeg_token,
                        "stable_depeg_pct": profile.stable_depeg_pct,
                        "liquidity_scale": profile.liquidity_scale,
                        "include_toxic_pools": profile.include_toxic_pools,
                    },
                    "swap_request": {
                        "token_in": request.token_in,
                        "token_out": request.token_out,
                        "amount_in": request.amount_in,
                    },
                    "pools": [
                        {
                            "pool_id": p.pool_id,
                            "dex": p.dex,
                            "token0": p.token0,
                            "token1": p.token1,
                            "reserve0": p.reserve0,
                            "reserve1": p.reserve1,
                            "fee": p.fee,
                            "spot_token0_to_token1": p.get_spot_price(p.token0),
                            "spot_token1_to_token0": p.get_spot_price(p.token1),
                        }
                        for p in pools
                    ],
                }
            )

        return payload

    def export_dataset(self, output_path: str) -> str:
        payload = self.generate_dataset()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path

