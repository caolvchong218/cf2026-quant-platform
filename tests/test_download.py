"""Offline acquisition tests: date boundaries, output quality and resumable cache."""
import json
from pathlib import Path
import pandas as pd
import pytest
from cfquant import data


class FakeProvider:
    requests = []

    def __init__(self, cache, token_file, delay):
        cache.mkdir(parents=True, exist_ok=True)

    def query(self, api, params, fields):
        self.requests.append((api, params.copy()))
        if api == "trade_cal":
            return pd.DataFrame({"cal_date": ["20191230", "20191231", "20200102", "20200103"],
                                 "is_open": [1, 1, 1, 1]})
        if "trade_date" in params:
            return pd.DataFrame({"ts_code": ["600001.SH", "000001.SZ", "300001.SZ"],
                                 "close": [10., 11., 20.], "vol": [100, 100, 100],
                                 "amount": [100, 200, 999]})
        frame = pd.DataFrame({"ts_code": [params["ts_code"]] * 4,
                              "trade_date": ["20200103", "20200102", "20191231", "20191230"]})
        if api == "adj_factor":
            frame["adj_factor"] = [2., 2., 1., 1.]
        elif api == "stk_limit":
            frame["up_limit"], frame["down_limit"] = 11., 9.
        else:
            frame["open"], frame["high"], frame["low"], frame["close"] = 10., 11., 9., 10.
            frame["pre_close"], frame["vol"], frame["amount"] = 10., 100., 1000.
        return frame


def test_expanded_dates_pool_and_parallel_output(tmp_path, monkeypatch):
    FakeProvider.requests = []
    monkeypatch.setattr(data, "TushareProvider", FakeProvider)
    args = dict(size=2, start="20191230", selection_date="20191231",
                study_start="20200102", end="20200103", workers=2)
    manifest = data.download_project(tmp_path, None, **args)
    market = data.load_market(tmp_path/"data/processed/market.csv")
    assert len(market) == 8
    assert set(market.asset) == {"000001.SZ", "600001.SH"}
    assert market.loc[market.date == "2020-01-03", "close"].tolist() == [20., 20.]
    assert (market.volume == 10000).all()
    assert manifest["actual_period"] == ["2019-12-30", "2020-01-03"]
    assert manifest["study_period"] == ["2020-01-02", "2020-01-03"]
    assert all(p["end_date"] == "20200103" for _, p in FakeProvider.requests if "end_date" in p)
    progress = json.loads((tmp_path/"data/processed/progress.json").read_text())
    assert progress["status"] == "complete" and progress["completed_assets"] == 2
    old_hash = data.digest(tmp_path/"data/processed/market.csv")
    data.download_project(tmp_path, None, **{**args, "workers": 1})
    assert data.digest(tmp_path/"data/processed/market.csv") == old_hash
    with pytest.raises(ValueError, match="new --root"):
        data.download_project(tmp_path, None, **{**args, "end": "20200106"})
    assert data.digest(tmp_path/"data/processed/market.csv") == old_hash


@pytest.mark.parametrize("kwargs", [
    {"selection_date": "20230103"}, {"size": 0}, {"workers": 9},
    {"delay": 0.1}, {"max_gib": 0}, {"start": "20000101"}])
def test_bad_download_plan_fails_before_credentials(tmp_path, kwargs):
    with pytest.raises(ValueError):
        data.download_project(tmp_path, None, **kwargs)


def test_cache_reuses_success_and_excludes_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "offline-test-credential")
    calls = []

    class Response:
        def __enter__(self):
            from io import StringIO
            return StringIO(json.dumps({"code": 0, "data": {"fields": ["close"], "items": [[10]]}}))

        def __exit__(self, *args):
            pass

    def request(req, timeout):
        calls.append(req)
        return Response()

    monkeypatch.setattr(data.urllib.request, "urlopen", request)
    provider = data.TushareProvider(tmp_path)
    for _ in range(2):
        assert provider.query("daily", {"ts_code": "600000.SH"}, "close").close.tolist() == [10]
    assert len(calls) == 1
    assert all("offline-test-credential" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_storage_guard_keeps_cache_and_does_not_mark_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "TushareProvider", FakeProvider)
    raw = tmp_path/"data/raw"
    raw.mkdir(parents=True)
    (raw/"previous.json").write_text('{"retained":true}')
    with pytest.raises(ValueError, match="budget"):
        data.download_project(tmp_path, None, size=2, start="20191230", selection_date="20191231",
                              study_start="20200102", end="20200103", max_gib=1e-9)
    assert (raw/"previous.json").exists()
    assert json.loads((tmp_path/"data/processed/progress.json").read_text())["status"] == "interrupted"
