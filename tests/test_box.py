"""
=====================================================================
 tests/test_box.py : 박스 인식과 박스 전략이 의도대로 동작하는지 검사
=====================================================================
  - 확정 터치를 제대로 세는가 (반등 전에는 안 세는가)
  - 박스 계산에 오늘 데이터를 안 쓰는가
  - 바닥 2번 → 3번째 접근에서 매수 신호가 나오는가
  - 바닥 1번뿐이거나, 상승 추세이거나, 폭이 너무 넓으면 신호가 안 나오는가
  - ★ 미래 참조 검사: 데이터를 t일까지 잘라서 줬을 때와 전체를 줬을 때 t일 신호가 같은가
  - 손절(-5%)·익절 경계값이 정확한가
"""
import numpy as np
import pandas as pd

from src.indicators import count_confirmed_touches, rolling_box
from src.strategies import BoxStrategy
from tests.conftest import make_ohlc, path_from_points


def test_confirmed_touch_requires_rebound():
    lows = np.array([100, 105, 108, 101, 100, 104])
    highs = lows + 1
    n = count_confirmed_touches(lows, highs, lambda v: v <= 101, lambda v: v >= 107)
    # 1차 터치(100) → 반등(109) 완료 = 1회. 2차 터치(101,100)는 반등이 아직 없음 → 미확정
    assert n == 1


def test_box_values_use_only_previous_bars(double_bottom_df):
    df = double_bottom_df
    box = rolling_box(df, 60, 0.015, 0.5)
    t = 70
    assert box["box_high"].iloc[t] == df["high"].iloc[t - 60:t].max()
    assert box["box_low"].iloc[t] == df["low"].iloc[t - 60:t].min()
    assert box["box_high"].iloc[:60].isna().all()


def test_double_bottom_entry_signal(double_bottom_df, box_params):
    sig = BoxStrategy(box_params).generate_signals(double_bottom_df)
    entries = sig.index[sig["entry"]]
    assert len(entries) > 0
    first = sig.loc[entries[0]]
    assert first["bottom_touches"] >= 2
    assert first["width_pct"] <= 0.15
    # 신호가 나온 날은 세 번째 하단 접근 구간(봉 58~66)이어야 한다
    i = sig.index.get_loc(entries[0])
    assert 58 <= i <= 66
    # 신호일 종가는 하단~매수구간 상단 사이
    assert first["box_low"] <= first["close"] <= first["entry_top"]


def test_single_bottom_no_entry(box_params):
    # 하단을 한 번만 찍고 반등한 뒤 두 번째로 접근 → 확정 터치 1회라 매수 없음
    pts = [(0, 108), (5, 107), (15, 100.5), (30, 109.5), (45, 105.5), (60, 108), (75, 101.0)]
    sig = BoxStrategy(box_params).generate_signals(make_ohlc(path_from_points(pts)))
    assert not sig["entry"].any()
    assert sig["bottom_touches"].max() <= 1


def test_trending_market_no_box(box_params):
    df = make_ohlc(np.linspace(100, 200, 200))
    sig = BoxStrategy(box_params).generate_signals(df)
    assert not sig["is_box"].any()
    assert not sig["entry"].any()


def test_wide_range_not_box(box_params):
    # 100~130 (30%) 횡보는 폭 조건 15% 초과로 박스 아님
    pts = [(0, 115), (12, 100.5), (24, 129), (36, 100.6), (48, 129), (62, 102), (70, 115)]
    sig = BoxStrategy(box_params).generate_signals(make_ohlc(path_from_points(pts)))
    assert not sig["entry"].any()


def test_no_lookahead(double_bottom_df, box_params):
    """t일까지 자른 데이터와 전체 데이터의 t일 신호가 같아야 한다."""
    strat = BoxStrategy(box_params)
    full = strat.generate_signals(double_bottom_df)
    cols = ["box_high", "box_low", "bottom_touches", "top_touches", "entry", "score"]
    for t in range(55, len(double_bottom_df)):
        part = strat.generate_signals(double_bottom_df.iloc[: t + 1])
        pd.testing.assert_series_equal(part[cols].iloc[-1], full[cols].iloc[t], check_names=False)


def test_exit_rules(box_params):
    strat = BoxStrategy(box_params)
    meta = {"box_low": 100.0, "box_high": 110.0}
    bar = lambda c: pd.Series({"open": c, "high": c, "low": c, "close": c})  # noqa: E731
    assert strat.exit_reason(meta, bar(96.0)) is None             # 하단 이탈했지만 -5% 안쪽
    assert strat.exit_reason(meta, bar(94.9)).startswith("stop_box_break")  # 100*0.95=95 미만
    assert strat.exit_reason(meta, bar(105.0)) is None
    assert strat.exit_reason(meta, bar(108.0)).startswith("take_profit")    # 110-2=108 이상
