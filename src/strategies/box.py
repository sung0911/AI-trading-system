"""
=====================================================================
 strategies/box.py : 박스권 하단 매매 전략 — "사? 팔아? 손절?"을 결정하는 파일
=====================================================================

indicators.py가 "측정"(박스 상단·하단·터치 횟수)을 했다면,
이 파일은 그 숫자를 보고 "결정"을 한다.

  ┌ 매수 조건 (4가지 모두 만족) ────────────────────────────
  │  ① 박스 폭이 15% 이하              (횡보 중인가?)
  │  ② 바닥 확정 터치 2회 이상          (바닥이 두 번 받쳐줬나?)
  │  ③ 천장 확정 터치 1회 이상          (천장에 막혀 본 적 있나?)
  │  ④ 오늘 종가가 [하단 ~ 하단+폭×20%] 사이  (바닥 근처로 내려왔나?)
  └──────────────────────────────────────────────────────

  ┌ 매도 조건 (하나라도 걸리면) ────────────────────────────
  │  손절 : 종가 < 하단 × (1 - 5%)        (박스가 깨졌다 → 손실 확정하고 탈출)
  │  익절 : 종가 ≥ 상단 - 폭×20%          (천장 근처 도착 → 수익 실현)
  │  (최대 보유기간 60일은 전략 공통 규칙이라 engine.py에서 처리)
  └──────────────────────────────────────────────────────

  그림 (하단 100, 상단 110, 폭 10 일 때)
     110 ─────────────── 상단
     108 ─ ─ ─ ─ ─ ─ ─ ─ 익절선 (108 이상이면 매도)
          ...
     102 ─ ─ ─ ─ ─ ─ ─ ─ 매수 구간 상단
     100 ─────────────── 하단   (100~102 사이면 매수)
      95 ─ ─ ─ ─ ─ ─ ─ ─ 손절선 (95 미만이면 매도)

  ★ 박스 상·하단은 "산 날 기준으로 고정"한다.
    박스는 매일 새로 계산되므로, 고정하지 않으면 주가가 내려갈 때 하단도 같이 내려가서
    손절선이 계속 밀려 내려가는 문제가 생긴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import rolling_box
from .base import Strategy


class BoxStrategy(Strategy):
    name = "box"

    def __init__(self, params: dict):
        # params = config.yaml의 strategy.box 부분 (lookback, max_width_pct, ...)
        self.p = params

    # -----------------------------------------------------------------
    # 질문 1) 날짜별 매수 신호 만들기
    # -----------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p

        # 1단계: indicators.py로 날짜별 박스 측정 (상단, 하단, 폭, 터치 횟수)
        box = rolling_box(df, int(p["lookback"]), float(p["touch_tol_pct"]),
                          float(p["rebound_frac"]))
        close = df["close"]                                    # 날짜별 종가
        width = box["box_high"] - box["box_low"]               # 박스 폭 (가격 차이)
        entry_top = box["box_low"] + width * float(p["entry_zone_frac"])  # 매수 구간 상단

        # 2단계: "박스로 인정되나?" = 조건 ①②③
        #   (& = 그리고. 날짜마다 참/거짓이 한꺼번에 계산된다)
        is_box = (
            (box["width_pct"] <= float(p["max_width_pct"]))              # ① 폭 15% 이하
            & (box["bottom_touches"] >= int(p["min_bottom_touches"]))    # ② 바닥 2회 이상
            & (box["top_touches"] >= int(p["min_top_touches"]))          # ③ 천장 1회 이상
        )
        # 3단계: "바닥 근처인가?" = 조건 ④ (하단 이상이면서 매수 구간 상단 이하)
        in_zone = (close >= box["box_low"]) & (close <= entry_top)

        # 4단계: 둘 다 참이면 매수 신호 (fillna(False) = 계산 불가한 날은 '신호 없음' 처리)
        entry = (is_box & in_zone).fillna(False).astype(bool)

        # 결과 표 만들기 (박스 정보 + 신호)
        out = box.copy()
        out["close"] = close
        out["is_box"] = is_box.fillna(False).astype(bool)
        out["entry_top"] = entry_top
        out["entry"] = entry

        # 점수: 바닥에 가까울수록 높은 점수 → 같은 날 후보가 많으면 바닥에 더 붙은 종목부터 산다
        #   바닥에 딱 붙음 = 0점, 매수 구간 맨 위 = -0.2점
        out["score"] = -((close - box["box_low"]) / width).where(entry, np.nan)

        # 매수 근거를 글로 남긴다 (로그·리포트에서 "왜 샀는지" 확인용)
        out["reason"] = ""
        r = out.loc[entry]
        out.loc[entry, "reason"] = [
            f"box[{lo:.4g}~{hi:.4g}] width={w:.1%} bottom_touches={b} top_touches={tp} close={c:.4g}"
            for lo, hi, w, b, tp, c in zip(r["box_low"], r["box_high"], r["width_pct"],
                                         r["bottom_touches"], r["top_touches"], r["close"])
        ]
        return out

    # -----------------------------------------------------------------
    # 질문 2) 살 때 기억해 둘 정보: 그날의 박스 하단·상단 (팔 때까지 고정)
    # -----------------------------------------------------------------
    def entry_meta(self, signal_row: pd.Series) -> dict:
        return {"box_low": float(signal_row["box_low"]), "box_high": float(signal_row["box_high"])}

    # -----------------------------------------------------------------
    # 질문 3) 오늘 팔아야 하나? (매일 종가로 판단 → 팔면 엔진이 다음날 시가에 매도)
    # -----------------------------------------------------------------
    def exit_reason(self, meta: dict, bar: pd.Series) -> str | None:
        lo, hi = meta["box_low"], meta["box_high"]    # 산 날 고정해 둔 박스
        close = float(bar["close"])                    # 오늘 종가

        # 손절을 먼저 검사한다 (손실 방어가 우선)
        stop_level = lo * (1 - float(self.p["stop_below_box_pct"]))   # 예) 100 × 0.95 = 95
        if close < stop_level:
            return f"stop_box_break close={close:.4g} < {stop_level:.4g}"

        # 익절 검사
        tp_level = hi - (hi - lo) * float(self.p["exit_zone_frac"])   # 예) 110 - 10×0.2 = 108
        if close >= tp_level:
            return f"take_profit_box_top close={close:.4g} >= {tp_level:.4g}"

        return None   # 둘 다 아니면 계속 보유
