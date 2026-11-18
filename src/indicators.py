"""지표 계산. 모든 값은 해당 시점 이전 데이터만 사용한다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def count_confirmed_touches(values_touch: np.ndarray, values_rebound: np.ndarray,
                            is_touch, is_rebound) -> int:
    """터치 → 반등이 완료된 횟수를 센다.

    터치 후 반등이 아직 안 나왔으면(진행 중인 터치) 세지 않는다.
    같은 봉에서 터치와 반등이 같이 나오면 터치만 인정한다.
    """
    confirmed = 0
    in_touch = False
    for vt, vr in zip(values_touch, values_rebound):
        if not in_touch:
            if is_touch(vt):
                in_touch = True
        elif is_rebound(vr):
            confirmed += 1
            in_touch = False
    return confirmed


def rolling_box(df: pd.DataFrame, lookback: int, touch_tol_pct: float,
                rebound_frac: float) -> pd.DataFrame:
    """t일 행에는 t일 직전 lookback봉(t-lookback ~ t-1)으로 만든 박스 정보를 담는다.

    t일 자신의 데이터는 쓰지 않으므로 t일 장중에 이미 알 수 있는 값이다(미래 참조 없음).

    반환 컬럼:
      box_high, box_low : 구간 최고 고가 / 최저 저가
      width_pct         : (box_high - box_low) / box_low
      bottom_touches    : 하단 터치 후 폭의 rebound_frac 이상 반등까지 완료된 횟수
      top_touches       : 상단 터치 후 폭의 rebound_frac 이상 하락까지 완료된 횟수
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    box_high = np.full(n, np.nan)
    box_low = np.full(n, np.nan)
    bottom = np.zeros(n, dtype=int)
    top = np.zeros(n, dtype=int)

    for t in range(lookback, n):
        wh = high[t - lookback:t]
        wl = low[t - lookback:t]
        hi, lo = wh.max(), wl.min()
        box_high[t], box_low[t] = hi, lo
        width = hi - lo
        bottom_zone = lo * (1 + touch_tol_pct)
        top_zone = hi * (1 - touch_tol_pct)
        up_level = lo + width * rebound_frac      # 하단 터치 후 도달해야 할 고가
        down_level = hi - width * rebound_frac    # 상단 터치 후 내려가야 할 저가
        bottom[t] = count_confirmed_touches(
            wl, wh, lambda v: v <= bottom_zone, lambda v: v >= up_level)
        top[t] = count_confirmed_touches(
            wh, wl, lambda v: v >= top_zone, lambda v: v <= down_level)

    out = pd.DataFrame(
        {"box_high": box_high, "box_low": box_low,
         "bottom_touches": bottom, "top_touches": top},
        index=df.index,
    )
    out["width_pct"] = (out["box_high"] - out["box_low"]) / out["box_low"]
    return out
