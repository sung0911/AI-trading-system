"""
=====================================================================
 indicators.py : 지표 계산 파일 — 지금은 "박스권"을 찾는 계산만 들어 있다
=====================================================================

지표(indicator)란?
  주가 데이터를 가공해서 "지금 상태가 어떤가"를 숫자로 나타낸 것.
  (예: 이동평균, RSI ... 이 프로젝트는 지금 박스권 계산만 사용)

이 파일이 매일매일 계산하는 것 (날짜 하나당 한 줄)
  box_high       : 최근 60일 동안 가장 높았던 가격  → 박스 "상단(천장)"
  box_low        : 최근 60일 동안 가장 낮았던 가격  → 박스 "하단(바닥)"
  width_pct      : 박스 폭 = (상단 - 하단) ÷ 하단    → 15% 이하면 횡보로 인정
  bottom_touches : 바닥을 찍고 "확실히 반등한" 횟수 → 2회 이상이면 매수 후보
  top_touches    : 천장을 찍고 "확실히 하락한" 횟수

★ 가장 중요한 원칙: 미래 정보 사용 금지 (미래 참조 금지)
  t일(오늘)의 박스는 "어제까지 60일"만으로 만든다. 오늘 데이터는 쓰지 않는다.
  → 실제 매매에서 알 수 없는 정보를 백테스트에서 몰래 쓰면
    결과가 실제보다 훨씬 좋게 나오는 착시가 생긴다.

        |←────────── 박스 계산에 쓰는 60일 ──────────→| 오늘(t)
        t-60  t-59  ...                     t-2   t-1 |  t
                                                        ↑ 오늘 종가는 "박스와 비교"만 한다
"""
from __future__ import annotations

import numpy as np    # 숫자 배열을 빠르게 계산하는 라이브러리
import pandas as pd


# ---------------------------------------------------------------------
# [1] "확정된 터치" 횟수 세기
# ---------------------------------------------------------------------
def count_confirmed_touches(values_touch: np.ndarray, values_rebound: np.ndarray,
                            is_touch, is_rebound) -> int:
    """
    바닥(또는 천장)을 "찍고 → 반등까지 완료한" 횟수를 센다.

    왜 그냥 "바닥 근처에 온 날 수"를 세지 않나?
      바닥 근처에서 5일 동안 머물면 5번이 아니라 "1번" 받친 것이다.
      그리고 찍은 뒤 다시 올라가야 "바닥이 받쳐줬다"고 할 수 있다.
      그래서 [터치 → 반등] 한 세트를 1회로 센다.

    동작 방식 (상태가 두 개인 스위치)
      상태 A "대기 중"   : 바닥 근처에 오면 → 상태 B로
      상태 B "터치 중"   : 박스 중간 이상 반등하면 → 1회 확정(+1), 상태 A로 복귀

      가격 흐름 예시 (하단 터치 = ▼, 반등 완료 = ▲)
        110 ┤      ▲            ▲
        105 ┤    /   \        /   \
        100 ┤ ▼        \ ▼  /       \ ▼ ← 지금 막 3번째로 내려옴
                                         (아직 반등 전이라 세지 않음)
        → 확정 터치 = 2회

    매개변수 (바닥용으로 쓸 때)
      values_touch   : 터치 여부를 볼 값 (저가)
      values_rebound : 반등 여부를 볼 값 (고가)
      is_touch       : "이 값이 바닥 근처인가?"를 판단하는 함수
      is_rebound     : "이 값이 충분히 반등했나?"를 판단하는 함수
      (천장용으로 쓸 땐 고가/저가, 위/아래를 뒤집어서 같은 함수를 재사용)
    """
    confirmed = 0       # 확정된 횟수
    in_touch = False    # 지금 "터치 중" 상태인가?
    for vt, vr in zip(values_touch, values_rebound):   # 하루씩 순서대로 본다
        if not in_touch:
            if is_touch(vt):        # 대기 중인데 바닥 근처에 왔다
                in_touch = True     #  → 터치 시작
        elif is_rebound(vr):        # 터치 중인데 충분히 올라갔다
            confirmed += 1          #  → 1회 확정!
            in_touch = False        #  → 다시 대기
    # 마지막에 "터치 중"으로 끝났다면(아직 반등 전) 그건 세지 않는다
    return confirmed


# ---------------------------------------------------------------------
# [2] 날짜마다 박스 정보 계산
# ---------------------------------------------------------------------
def rolling_box(df: pd.DataFrame, lookback: int, touch_tol_pct: float,
                rebound_frac: float) -> pd.DataFrame:
    """
    모든 날짜에 대해 "그날 직전 lookback(60)일"로 박스를 계산한다.
    (rolling = 창문을 하루씩 밀면서 계산한다는 뜻)

    매개변수 (config.yaml의 strategy.box 값이 들어옴)
      lookback      : 60     → 최근 60일을 본다
      touch_tol_pct : 0.015  → 하단 + 1.5% 이내로 내려오면 "터치"
      rebound_frac  : 0.5    → 박스 폭의 50% 이상 올라가면 "반등 완료"

    예) 하단 100, 상단 110 (폭 10)
        터치 기준  : 저가 ≤ 101.5   (100 × 1.015)
        반등 기준  : 고가 ≥ 105     (100 + 10 × 0.5)
    """
    # 표에서 고가·저가 열을 숫자 배열로 꺼낸다 (계산을 빠르게 하려고)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)

    # 결과를 담을 빈 배열 (NaN = 아직 값 없음)
    # 처음 60일은 과거 데이터가 부족해서 계산 불가 → NaN으로 남는다
    box_high = np.full(n, np.nan)
    box_low = np.full(n, np.nan)
    bottom = np.zeros(n, dtype=int)
    top = np.zeros(n, dtype=int)

    for t in range(lookback, n):           # t = 오늘. 60번째 날부터 마지막 날까지
        wh = high[t - lookback:t]          # 어제까지 60일의 고가들 (오늘 t는 미포함!)
        wl = low[t - lookback:t]           # 어제까지 60일의 저가들
        hi, lo = wh.max(), wl.min()        # 60일 최고가 = 상단, 최저가 = 하단
        box_high[t], box_low[t] = hi, lo
        width = hi - lo                    # 박스 폭 (가격 차이)

        bottom_zone = lo * (1 + touch_tol_pct)   # 이 값 이하로 내려오면 "하단 터치"
        top_zone = hi * (1 - touch_tol_pct)      # 이 값 이상으로 올라가면 "상단 터치"
        up_level = lo + width * rebound_frac     # 하단 터치 후 고가가 여기까지 오면 "반등 완료"
        down_level = hi - width * rebound_frac   # 상단 터치 후 저가가 여기까지 오면 "하락 완료"

        # 바닥: 저가로 터치를 보고, 고가로 반등을 본다
        bottom[t] = count_confirmed_touches(
            wl, wh, lambda v: v <= bottom_zone, lambda v: v >= up_level)
        # 천장: 고가로 터치를 보고, 저가로 하락을 본다 (바닥과 거울처럼 반대)
        top[t] = count_confirmed_touches(
            wh, wl, lambda v: v >= top_zone, lambda v: v <= down_level)
        # (lambda = 이름 없는 짧은 함수. "v를 받아서 v <= bottom_zone 인지 돌려줘"라는 뜻)

    # 결과를 날짜별 표로 만든다
    out = pd.DataFrame(
        {"box_high": box_high, "box_low": box_low,
         "bottom_touches": bottom, "top_touches": top},
        index=df.index,
    )
    # 박스 폭 비율: 예) (110 - 100) ÷ 100 = 0.10 = 10%
    out["width_pct"] = (out["box_high"] - out["box_low"]) / out["box_low"]
    return out
