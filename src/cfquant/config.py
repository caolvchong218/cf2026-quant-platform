from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import pandas as pd
import yaml


@dataclass(frozen=True)
class Config:
    name: str = "momentum_weekly"
    data_path: str = "data/processed/market.csv"
    calendar_path: str = "data/processed/calendar.csv"
    start: str = "2023-01-03"
    end: str = "2025-12-31"
    factor: str = "momentum"
    factor_params: dict = field(default_factory=lambda: {"window": 20})
    rebalance_every: int = 5
    holdings: int = 10
    initial_cash: float = 1_000_000.0
    buy_cost: float = 0.001
    sell_cost: float = 0.0015
    annualization: int = 252
    risk_free_annual: float = 0.0
    diagnostic_horizon: int = 5
    groups: int = 5
    min_assets: int = 10
    respect_price_limits: bool = True
    max_stale_sessions: int = 20

    def __post_init__(self):
        for name in ["rebalance_every", "holdings", "annualization", "diagnostic_horizon", "groups", "min_assets", "max_stale_sessions"]:
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.groups < 2 or self.min_assets < self.groups:
            raise ValueError("Require groups >= 2 and min_assets >= groups")
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive and finite")
        for name in ["buy_cost", "sell_cost"]:
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        if not math.isfinite(self.risk_free_annual) or self.risk_free_annual <= -1:
            raise ValueError("Invalid annual risk-free return")
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError("end precedes start")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
