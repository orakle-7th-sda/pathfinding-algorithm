"""
Base classes and interfaces for DEX aggregator algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time


class Difficulty(Enum):
    """Algorithm difficulty level"""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class AlgorithmType(Enum):
    """Algorithm type classification"""
    SINGLE = "Single Path"
    COMPOSITE = "Composite Path"


@dataclass
class Pool:
    """Represents a DEX liquidity pool"""
    pool_id: str
    dex: str
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    fee: float = 0.003  # 0.3% default fee
    
    def get_amount_out(self, amount_in: float, token_in: str) -> float:
        """
        Calculate output amount using constant product formula (x * y = k).
        Includes fee deduction.
        """
        if token_in == self.token0:
            reserve_in, reserve_out = self.reserve0, self.reserve1
        elif token_in == self.token1:
            reserve_in, reserve_out = self.reserve1, self.reserve0
        else:
            raise ValueError(f"Token {token_in} not in pool {self.pool_id}")
        
        amount_in_with_fee = amount_in * (1 - self.fee)
        amount_out = (reserve_out * amount_in_with_fee) / (reserve_in + amount_in_with_fee)
        return amount_out
    
    def get_price_impact(self, amount_in: float, token_in: str) -> float:
        """Calculate price impact percentage"""
        if token_in == self.token0:
            reserve_in = self.reserve0
        else:
            reserve_in = self.reserve1
        
        return amount_in / (reserve_in + amount_in)
    
    def get_spot_price(self, token_in: str) -> float:
        """Get spot price without any trade"""
        if token_in == self.token0:
            return self.reserve1 / self.reserve0
        return self.reserve0 / self.reserve1


@dataclass
class SwapRequest:
    """Represents a swap request"""
    token_in: str
    token_out: str
    amount_in: float


@dataclass
class SwapRoute:
    """Represents a single swap route"""
    path: List[str]  # Token path: ["ETH", "USDC"] or ["ETH", "WBTC", "USDC"]
    pools: List[Pool]  # Pools used in this route
    dexes: List[str]  # DEX names for each hop
    amount_in: float
    amount_out: float
    
    @property
    def price_impact(self) -> float:
        """Calculate total price impact for this route"""
        if not self.pools:
            return 0.0
        
        total_impact = 0.0
        current_amount = self.amount_in
        
        for i, pool in enumerate(self.pools):
            token_in = self.path[i]
            impact = pool.get_price_impact(current_amount, token_in)
            total_impact += impact
            current_amount = pool.get_amount_out(current_amount, token_in)
        
        return total_impact


@dataclass
class SwapResult:
    """Result of a swap execution"""
    routes: List[SwapRoute]  # Can be multiple routes for split orders
    total_amount_in: float
    total_amount_out: float
    total_price_impact: float
    execution_time_ms: float
    algorithm_name: str
    gas_estimate: int = 150000  # Base gas estimate
    
    @property
    def effective_price(self) -> float:
        """Calculate effective price (amount_out / amount_in)"""
        if self.total_amount_in == 0:
            return 0.0
        return self.total_amount_out / self.total_amount_in
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            "algorithm": self.algorithm_name,
            "amount_in": self.total_amount_in,
            "amount_out": self.total_amount_out,
            "price_impact": self.total_price_impact,
            "effective_price": self.effective_price,
            "execution_time_ms": self.execution_time_ms,
            "gas_estimate": self.gas_estimate,
            "num_routes": len(self.routes),
            "routes": [
                {
                    "path": route.path,
                    "dexes": route.dexes,
                    "amount_in": route.amount_in,
                    "amount_out": route.amount_out
                }
                for route in self.routes
            ]
        }


@dataclass
class AlgorithmMetadata:
    """Metadata describing an algorithm"""
    name: str
    difficulty: Difficulty
    algorithm_type: AlgorithmType
    time_complexity: str
    space_complexity: str
    description: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


class BaseAlgorithm(ABC):
    """
    Base class for all DEX aggregator routing algorithms.
    
    All algorithms must implement the `find_best_route` method
    and provide metadata about themselves.
    """
    
    def __init__(self, pools: List[Pool]):
        """
        Initialize the algorithm with available pools.
        
        Args:
            pools: List of available liquidity pools
        """
        self.pools = pools
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build token graph from pools for pathfinding"""
        self.token_graph: Dict[str, List[Tuple[str, Pool]]] = {}
        
        for pool in self.pools:
            # Add edges in both directions
            if pool.token0 not in self.token_graph:
                self.token_graph[pool.token0] = []
            if pool.token1 not in self.token_graph:
                self.token_graph[pool.token1] = []
            
            self.token_graph[pool.token0].append((pool.token1, pool))
            self.token_graph[pool.token1].append((pool.token0, pool))
    
    def get_pools_for_pair(self, token_in: str, token_out: str) -> List[Pool]:
        """Get all pools that can swap between two tokens directly"""
        return [
            pool for pool in self.pools
            if (pool.token0 == token_in and pool.token1 == token_out) or
               (pool.token1 == token_in and pool.token0 == token_out)
        ]
    
    @property
    @abstractmethod
    def metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata"""
        pass
    
    @abstractmethod
    def find_best_route(self, request: SwapRequest) -> SwapResult:
        """
        Find the best route(s) for the given swap request.
        
        Args:
            request: The swap request containing token_in, token_out, amount_in
            
        Returns:
            SwapResult containing the best route(s) found
        """
        pass
    
    def execute_with_timing(self, request: SwapRequest) -> SwapResult:
        """Execute the algorithm with timing measurement"""
        start_time = time.perf_counter()
        result = self.find_best_route(request)
        end_time = time.perf_counter()
        
        result.execution_time_ms = (end_time - start_time) * 1000
        return result
    
    def calculate_route_output(self, path: List[str], pools: List[Pool], 
                                amount_in: float) -> float:
        """
        Calculate the output amount for a given route.
        
        Args:
            path: Token path (e.g., ["ETH", "USDC"])
            pools: Pools to use for each hop
            amount_in: Input amount
            
        Returns:
            Output amount after all swaps
        """
        current_amount = amount_in
        
        for i, pool in enumerate(pools):
            token_in = path[i]
            current_amount = pool.get_amount_out(current_amount, token_in)
        
        return current_amount
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pools={len(self.pools)})"
