"""Large-panel regression: optional NumExpr must not silently zero returns."""
import numpy as np
import pandas as pd
import cfquant


def test_large_panel_returns_match_explicit_division_across_allocations():
    rng=np.random.default_rng(7)
    # Above pandas' optional-expression threshold, with both common layouts.
    for iteration in range(20):
        values=rng.uniform(10,100,(1690,1000))
        if iteration%2:values=np.asfortranarray(values)
        actual=pd.DataFrame(values).pct_change(fill_method=None).to_numpy()
        expected=np.empty_like(values);expected[0]=np.nan
        expected[1:]=values[1:]/values[:-1]-1
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
