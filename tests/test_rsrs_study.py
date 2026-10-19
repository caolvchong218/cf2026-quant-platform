"""Research orchestration checks independent of strategy outcome."""
import json

import numpy as np
import pandas as pd
import pytest

from cfquant.rsrs_study import load_data, paired_bootstrap, paired_table


def test_invalid_input_rejected(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "open": [11], "high": [10], "low": [9], "close": [10]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="ordering"):
        load_data(f)


def test_paired_metrics_preserve_losing_improvement():
    rows = []
    for name, x in [("right_ma", .2), ("right_ma_vol15", -.1), ("right_ma_cash75", .1)]:
        rows.append(dict(index_id="sh000300", period_id="post_publication", cost_bps=10,
                         strategy_id=name, total_return=x, cagr=x, sharpe=x,
                         max_drawdown=.5-x, annual_turnover=2., exposure=.75))
    row = paired_table(pd.DataFrame(rows)).iloc[0]
    assert row.delta_cagr == pytest.approx(-.3)
    assert row.delta_max_drawdown == pytest.approx(.3)


def test_bootstrap_is_paired_seeded_and_not_cagr(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    # A deterministic difference should yield a point interval, regardless of blocks.
    for strategy, r in [("right_ma", .001), ("right_ma_vol15", .002), ("right_ma_cash75", .0015)]:
        pd.DataFrame({"nav": np.cumprod(np.full(100, 1+r))}).to_csv(daily / f"sh000300__post_publication__{strategy}__10bps.csv", index=False)
    config = {"bootstrap": {"seed": 1, "repetitions": 100, "block_sessions": 20, "paired": True}}
    a, b = paired_bootstrap(tmp_path, config), paired_bootstrap(tmp_path, config)
    assert json.dumps(a) == json.dumps(b)
    comparison = a["comparisons"][0]
    assert comparison["ci_low"] == pytest.approx(.252)
    assert comparison["ci_high"] == pytest.approx(.252)
    assert "NOT CAGR" in a["metric"]
