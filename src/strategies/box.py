"""박스권 하단 매매 전략.

매수: 직전 N봉이 박스(폭 <= max_width_pct)이고, 하단 확정 터치가 2회 이상이며,
      오늘 종가가 하단 ~ 하단+폭*entry_zone_frac 사이로 다가왔을 때.
익절: 종가가 박스 상단-폭*exit_zone_frac 이상.
손절: 종가가 박스 하단*(1-stop_below_box_pct) 미만 (박스 이탈).
박스 상·하단은 매수 신호 시점 값으로 고정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import rolling_box
from .base import Strategy


class BoxStrategy(Strategy):
    name = "box"

    def __init__(self, params: dict):
        self.p = params

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        box = rolling_box(df, int(p["lookback"]), float(p["touch_tol_pct"]),
                          float(p["rebound_frac"]))
        close = df["close"]
        width = box["box_high"] - box["box_low"]
        entry_top = box["box_low"] + width * float(p["entry_zone_frac"])

        is_box = (
            (box["width_pct"] <= float(p["max_width_pct"]))
            & (box["bottom_touches"] >= int(p["min_bottom_touches"]))
            & (box["top_touches"] >= int(p["min_top_touches"]))
        )
        in_zone = (close >= box["box_low"]) & (close <= entry_top)
        entry = (is_box & in_zone).fillna(False).astype(bool)

        out = box.copy()
        out["close"] = close
        out["is_box"] = is_box.fillna(False).astype(bool)
        out["entry_top"] = entry_top
        out["entry"] = entry
        # 하단에 가까울수록 우선 (0 = 하단, -entry_zone_frac = 매수구간 상단)
        out["score"] = -((close - box["box_low"]) / width).where(entry, np.nan)
        out["reason"] = ""
        r = out.loc[entry]
        out.loc[entry, "reason"] = [
            f"box[{lo:.4g}~{hi:.4g}] width={w:.1%} bottom_touches={b} top_touches={tp} close={c:.4g}"
            for lo, hi, w, b, tp, c in zip(r["box_low"], r["box_high"], r["width_pct"],
                                         r["bottom_touches"], r["top_touches"], r["close"])
        ]
        return out

    def entry_meta(self, signal_row: pd.Series) -> dict:
        return {"box_low": float(signal_row["box_low"]), "box_high": float(signal_row["box_high"])}

    def exit_reason(self, meta: dict, bar: pd.Series) -> str | None:
        lo, hi = meta["box_low"], meta["box_high"]
        close = float(bar["close"])
        stop_level = lo * (1 - float(self.p["stop_below_box_pct"]))
        if close < stop_level:
            return f"stop_box_break close={close:.4g} < {stop_level:.4g}"
        tp_level = hi - (hi - lo) * float(self.p["exit_zone_frac"])
        if close >= tp_level:
            return f"take_profit_box_top close={close:.4g} >= {tp_level:.4g}"
        return None
