import numpy as np
import pandas as pd
import pytest

BOX_PARAMS = {
    "lookback": 60,
    "max_width_pct": 0.15,
    "touch_tol_pct": 0.015,
    "rebound_frac": 0.5,
    "min_bottom_touches": 2,
    "min_top_touches": 1,
    "entry_zone_frac": 0.20,
    "exit_zone_frac": 0.20,
    "stop_below_box_pct": 0.05,
}


def path_from_points(points: list[tuple[int, float]]) -> np.ndarray:
    """(봉 번호, 종가) 꼭짓점을 직선으로 이은 종가 배열."""
    xs, ys = zip(*points)
    return np.interp(np.arange(xs[-1] + 1), xs, ys)


def make_ohlc(closes, spread: float = 0.005, start: str = "2020-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]]
    high = np.maximum(opens, closes) * (1 + spread)
    low = np.minimum(opens, closes) * (1 - spread)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": closes,
                         "volume": 1_000_000.0}, index=idx)


# 100~110 박스에서 하단 2회 터치·반등 후 세 번째로 하단에 접근 → 반등해서 상단 도달
DOUBLE_BOTTOM_POINTS = [(0, 106), (12, 100.5), (24, 109.5), (36, 100.6), (48, 109.4),
                        (62, 101.2), (76, 109.0), (85, 105)]
# 위와 같지만 세 번째 접근 후 박스 하단을 크게 이탈
BREAKDOWN_POINTS = [(0, 106), (12, 100.5), (24, 109.5), (36, 100.6), (48, 109.4),
                    (62, 101.2), (64, 101.0), (70, 90.0), (80, 88.0)]


@pytest.fixture
def box_params():
    return dict(BOX_PARAMS)


@pytest.fixture
def double_bottom_df():
    return make_ohlc(path_from_points(DOUBLE_BOTTOM_POINTS))


@pytest.fixture
def breakdown_df():
    return make_ohlc(path_from_points(BREAKDOWN_POINTS))
