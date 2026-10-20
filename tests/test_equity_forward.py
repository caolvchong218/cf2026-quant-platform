from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cfquant.equity_forward import (
    SHANGHAI, append_daily_bars, assess_score_readiness, cog_rank_scores, completed_session,
    latest_features, v3_training_rows,
)
from cfquant.features import FEATURES


def test_current_daily_bar_waits_until_the_data_safety_cutoff():
    dates = pd.bdate_range("2026-10-16", "2026-10-20")
    assert completed_session(dates, "2026-10-20",
        now=datetime(2026, 10, 20, 10, tzinfo=SHANGHAI)) == pd.Timestamp("2026-10-19")
    assert completed_session(dates, "2026-10-20",
        now=datetime(2026, 10, 20, 17, 5, tzinfo=SHANGHAI)) == pd.Timestamp("2026-10-20")
    with pytest.raises(ValueError, match="Future"):
        completed_session(dates, "2026-10-21", now=datetime(2026, 10, 20, 18, tzinfo=SHANGHAI))


def test_score_readiness_requires_dated_complete_rows_and_a_refresh():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-20"] * 30), "asset": [str(n) for n in range(30)],
        "eligible": True, "raw_close": 10., "v3_lightgbm_frozen_reference": .1,
        "cogalpha_731_rank_frozen_reference": .2,
    })
    assert assess_score_readiness(rows, "2026-10-20", refreshed=True)["scores_ready"]
    assert not assess_score_readiness(rows, "2026-10-20", refreshed=False)["scores_ready"]
    rows.loc[0, "cogalpha_731_rank_frozen_reference"] = np.nan
    assert not assess_score_readiness(rows, "2026-10-20", refreshed=True)["scores_ready"]
    rows.loc[0, "date"] = pd.Timestamp("2026-10-21")
    with pytest.raises(ValueError, match="declared signal date"):
        assess_score_readiness(rows, "2026-10-20", refreshed=True)


def test_extension_uses_original_adjustment_anchor_and_lots():
    base = pd.DataFrame([dict(date=pd.Timestamp("2026-09-18"), asset="600000.SH",
        open=10., high=11., low=9., close=10., raw_open=10., raw_high=11., raw_low=9.,
        raw_close=10., volume=100., adj_factor=2., adjustment_reference=2.,
        up_limit=11., down_limit=9.)])
    daily = pd.DataFrame([dict(ts_code="600000.SH", trade_date="20260921",
        open=5., high=6., low=4., close=5., vol=12.)])
    adj = pd.DataFrame([dict(ts_code="600000.SH", trade_date="20260921", adj_factor=4.)])
    limits = pd.DataFrame([dict(ts_code="600000.SH", trade_date="20260921", up_limit=5.5, down_limit=4.5)])
    extended, extension = append_daily_bars(base, daily, adj, limits)
    assert extension.iloc[0].close == 10.
    assert extension.iloc[0].raw_close == 5.
    assert extension.iloc[0].volume == 1200.
    pd.testing.assert_frame_equal(extended.iloc[:1].reset_index(drop=True), base)
    with pytest.raises(ValueError, match="rewrite"):
        append_daily_bars(extended, daily, adj, limits)
    with pytest.raises(ValueError, match="Incomplete"):
        append_daily_bars(base, daily, adj.iloc[:0], limits)


def test_v3_training_excludes_labels_reaching_the_cutoff():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-11-20", "2025-11-21", "2022-12-20"]),
        "label_exit": pd.to_datetime(["2025-12-30", "2025-12-31", "2023-01-20"]),
        "label": [.1, 999., .2], "eligible": [True, True, True],
        **{"n_" + name: [0., 0., 0.] for name in FEATURES},
    })
    selected = v3_training_rows(frame, "2025-12-31")
    assert len(selected) == 1
    assert selected.label.iloc[0] == .1


def _fixture():
    dates = pd.bdate_range("2026-05-01", periods=100)
    assets = [f"{600000+i}.SH" for i in range(12)]
    rows = []
    for j, asset in enumerate(assets):
        for i, day in enumerate(dates):
            price = 10 + j + i * .02 + np.sin(i / 4 + j) * .1
            rows.append(dict(date=day, asset=asset, open=price, close=price,
                high=price + .2, low=price - .2, raw_close=price,
                volume=3e6 + j * 1000, adjustment_reference=1., adj_factor=1.,
                up_limit=price * 1.1, down_limit=price * .9))
    market = pd.DataFrame(rows)
    basic = pd.DataFrame([dict(ts_code=asset, trade_date=dates[80].strftime("%Y%m%d"),
        pe_ttm=10 + j, pb=1 + j * .1, dv_ttm=j + 1, total_mv=10000 + 1000*j,
        turnover_rate=j * .1 + 1) for j, asset in enumerate(assets)])
    fina = pd.DataFrame([dict(ts_code=asset, ann_date="20260401", end_date="20251231", roe=j+1)
                         for j, asset in enumerate(assets)])
    industry = pd.DataFrame([dict(ts_code=asset, l1_code="industry" + str(j//6),
        in_date="20200101", out_date=None) for j, asset in enumerate(assets)])
    return dates, market, basic, fina, industry


def test_current_features_cannot_see_future_prices_or_future_announcements():
    dates, market, basic, fina, industries = _fixture()
    date = dates[80]
    first = latest_features(market, dates, basic, fina, industries, date)
    changed = market.copy()
    changed.loc[changed.date > date, ["close", "high", "low", "open", "volume"]] *= 1000
    revised = pd.concat([fina, fina.assign(ann_date=(date + pd.Timedelta(days=1)).strftime("%Y%m%d"), roe=999.)])
    second = latest_features(changed, dates, basic, revised, industries, date)
    pd.testing.assert_frame_equal(first, second)
    assert "label" not in first
    assert (first.quality_roe < .2).all()


def test_cog_rank_requires_every_selected_factor_and_has_no_future_input():
    dates, market, _, _, _ = _fixture()
    assets = pd.Index(sorted(market.asset.unique()), name="asset")
    eligible = pd.Series(True, index=assets)
    cards = [{"id": "trend", "expression": "returns(close, 5)"},
             {"id": "impossible", "expression": "safe_divide(close, close-close)"}]
    scores, _ = cog_rank_scores(market, dates, eligible, cards, dates[80])
    assert scores.isna().all(), "A partial factor mean is not the frozen CogAlpha strategy"
    cards = cards[:1]
    first, _ = cog_rank_scores(market, dates, eligible, cards, dates[80])
    market.loc[market.date > dates[80], "close"] *= 1000
    second, _ = cog_rank_scores(market, dates, eligible, cards, dates[80])
    pd.testing.assert_series_equal(first, second)
