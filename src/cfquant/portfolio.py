import pandas as pd


def top_equal_weights(scores: pd.Series, holdings: int) -> pd.Series:
    """Deterministic ties by asset code; cash when no scores; no shorting."""
    valid = scores.dropna().rename("score").rename_axis("asset").reset_index()
    selected = valid.sort_values(["score", "asset"], ascending=[False, True]).head(holdings)
    weights = pd.Series(0.0, index=scores.index)
    if len(selected):
        weights.loc[selected.asset] = 1.0 / len(selected)
    return weights

