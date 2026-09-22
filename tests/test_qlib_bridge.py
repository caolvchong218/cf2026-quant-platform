"""Bridge contract tests run without Qlib; real engine check is optional."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cfquant.qlib_bridge import (EXPRESSIONS, bridge_command, compare_frames,
                                pandas_reference, qlib_symbol, unit_checks, validate_request, write_provider)


def example():
    dates = pd.bdate_range('2020-01-02', periods=45)
    close = 10 + np.sin(np.arange(45)/3) + np.arange(45)/20
    table = pd.DataFrame({'date': dates, 'asset': '000001.SZ', 'open': close*.99,
                          'high': close*1.02, 'low': close*.97, 'close': close,
                          'volume': 1000+np.arange(45)*11, 'factor': 1., 'vwap': close*.998})
    # A missing session must be visible to the Qlib calendar, rather than
    # shift subsequent observations and silently change lag definitions.
    return dates, table.drop(index=12)


def test_provider_preserves_calendar_gap_and_float32_header(tmp_path):
    dates, frame = example()
    target = tmp_path/'provider'
    manifest = write_provider(frame, dates, target)
    values = np.fromfile(target/'features/sz000001/close.day.bin', dtype='<f4')
    assert len(values) == 46 and values[0] == 0 and np.isnan(values[13])
    assert values[14] == np.float32(frame.loc[13, 'close'])
    assert manifest['calendar_sessions'] == 45
    assert 'SZ000001' in (target/'instruments/all.txt').read_text()
    with pytest.raises(ValueError, match='must be new'):
        write_provider(frame, dates, target)


def test_request_rejects_shell_like_input_and_invalid_intervals(tmp_path):
    assert qlib_symbol('600000.SH') == 'SH600000'
    for code in ['../600000.SH', '600000.SH;echo', '123.SH']:
        with pytest.raises(ValueError):
            qlib_symbol(code)
    with pytest.raises(ValueError):
        validate_request(20, '2020-01-02; echo bad', '2026-09-18')
    with pytest.raises(ValueError):
        validate_request(20, '2026-09-18', '2020-01-02')
    with pytest.raises(ValueError):
        validate_request(20, '2026-09-01', '2026-09-18')
    command = bridge_command(tmp_path, 20, '2023-01-03', '2024-12-31')
    assert isinstance(command, list) and '--assets' in command


def test_comparison_fails_on_missing_data_and_reference_std_convention():
    dates, frame = example()
    refs, panels = pandas_reference(frame, dates)
    returns = panels['close'].iloc[:3, 0].pct_change(fill_method=None).dropna()
    assert refs['low_volatility_20'].iloc[2, 0] == pytest.approx(-returns.std(ddof=1), abs=1e-7)
    actual = refs['trend_20'].copy()
    actual.iloc[4, 0] = np.nan
    assert compare_frames(actual, refs['trend_20'])['availability_mismatches'] == 1
    assert not compare_frames(actual, refs['trend_20'])['passed']


def test_actual_qlib_expression_engine_matches_independent_reference(tmp_path):
    qlib = pytest.importorskip('qlib', reason='Optional isolated .venv-qlib engine integration')
    from qlib.data import D
    dates, frame = example()
    target = tmp_path/'real_provider'
    write_provider(frame, dates, target)
    qlib.init(provider_uri=str(target), kernels=1, expression_cache=None, dataset_cache=None, logging_level=50)
    output = D.features(['SZ000001'], [x.expression for x in EXPRESSIONS],
                        start_time=str(dates[0].date()), end_time=str(dates[-1].date()))
    reference, _ = pandas_reference(frame, dates)
    for spec in EXPRESSIONS:
        actual = output[spec.expression].unstack('instrument').rename(columns={'SZ000001': '000001.SZ'})
        assert compare_frames(actual, reference[spec.name])['passed'], spec.name


def test_vwap_price_and_volume_use_inverse_adjustment_and_missing_is_not_passed():
    # Raw close 10, adjusted close 20; 100 shares traded for 1.1 thousand CNY.
    # Adjusted VWAP must be 22 and adjusted volume 50, with unchanged amount.
    data = pd.DataFrame({'close': [20.], 'raw_close': [10.], 'factor': [2.],
                         'vwap': [22.], 'volume': [50.], 'amount': [1.1]})
    checked = unit_checks(data)
    assert checked['price_scale_matches_close_over_raw_close']
    assert checked['vwap_times_adjusted_volume_equals_amount']
    assert checked['max_price_volume_amount_error'] == 0
    data['volume'] = 100.
    assert not unit_checks(data)['vwap_times_adjusted_volume_equals_amount']
    data['vwap'] = np.nan
    assert unit_checks(data)['amount_checked_rows'] == 0
    assert unit_checks(data)['vwap_times_adjusted_volume_equals_amount'] is None
