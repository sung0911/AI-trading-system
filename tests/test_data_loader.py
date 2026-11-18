"""네트워크 없이 가짜 fetcher로 캐시·증분·재시도 로직을 검증한다."""
import pandas as pd
import pytest

from src.data_loader import DataStore, normalize_ohlcv

CFG = {"start": "2024-01-01", "request_delay_sec": 0, "batch_size": 10, "batch_pause_sec": 0,
       "retries": 3, "retry_delay_sec": 0, "overlap_days": 3, "adjust_check_tol": 0.005}


def frame(n, start="2024-01-01", scale=1.0):
    idx = pd.bdate_range(start, periods=n)
    c = pd.Series(range(100, 100 + n), index=idx, dtype=float) * scale
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1.0})


class FakeSource:
    def __init__(self, full):
        self.full = full
        self.calls = []

    def __call__(self, symbol, start):
        self.calls.append(start)
        return normalize_ohlcv(self.full[self.full.index >= pd.Timestamp(start)])


def test_full_then_incremental(tmp_path):
    src = FakeSource(frame(20))
    store = DataStore(CFG, tmp_path, {"us": src}, sleep=lambda s: None)
    src.full = frame(15)
    df1 = store.update("us", "AAA")
    assert len(df1) == 15 and src.calls[-1] == "2024-01-01"

    src.full = frame(20)  # 새 봉 5개 추가
    df2 = store.update("us", "AAA")
    assert len(df2) == 20
    assert src.calls[-1] != "2024-01-01"          # 증분 요청
    pd.testing.assert_frame_equal(df2, normalize_ohlcv(frame(20)), check_freq=False)
    pd.testing.assert_frame_equal(store.load("us", "AAA"), df2, check_freq=False)


def test_adjustment_change_triggers_full_reload(tmp_path):
    src = FakeSource(frame(15))
    store = DataStore(CFG, tmp_path, {"us": src}, sleep=lambda s: None)
    store.update("us", "AAA")
    src.full = frame(20, scale=0.5)  # 분할 등으로 과거 가격 전체가 바뀜
    df = store.update("us", "AAA")
    assert src.calls[-1] == "2024-01-01"
    pd.testing.assert_frame_equal(df, normalize_ohlcv(frame(20, scale=0.5)), check_freq=False)


def test_failed_symbol_is_skipped(tmp_path):
    ok = FakeSource(frame(10))

    def fetch(symbol, start):
        if symbol == "BAD":
            raise ConnectionError("boom")
        return ok(symbol, start)

    store = DataStore(CFG, tmp_path, {"us": fetch}, sleep=lambda s: None)
    data, failed = store.update_many("us", ["AAA", "BAD", "BBB"])
    assert set(data) == {"AAA", "BBB"} and failed == ["BAD"]


def test_retry_then_success(tmp_path):
    src = FakeSource(frame(10))
    attempts = {"n": 0}

    def flaky(symbol, start):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("slow")
        return src(symbol, start)

    store = DataStore(CFG, tmp_path, {"us": flaky}, sleep=lambda s: None)
    assert len(store.update("us", "AAA")) == 10 and attempts["n"] == 3


def test_normalize_drops_bad_rows():
    df = frame(5)
    df.iloc[2, df.columns.get_loc("Open")] = 0
    with pytest.raises(ValueError):
        normalize_ohlcv(df.drop(columns=["Volume"]))
    assert len(normalize_ohlcv(df)) == 4
