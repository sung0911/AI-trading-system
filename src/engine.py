"""
=====================================================================
 engine.py : 백테스트 엔진 — 과거를 하루씩 넘기며 가짜 돈으로 실제처럼 사고판다
=====================================================================

백테스트란?
  "이 규칙대로 2016년부터 매매했다면 지금 얼마가 됐을까?"를 컴퓨터로 재현하는 것.
  엔진은 달력을 하루씩 넘기면서, 매일 장이 열리고 닫힐 때 할 일을 순서대로 한다.

★ 체결 규칙 (가장 중요)
  - 판단은 "오늘 종가"로 한다.   (장이 끝나야 종가를 알 수 있으니까)
  - 실제 사고파는 건 "내일 시가"로 한다.
  - 같은 날 종가를 보고 그 종가에 사는 건 불가능하다 (이미 장이 끝났으니까).
    → 이걸 어기면 백테스트 결과가 실제보다 좋게 나온다.

         월요일                         화요일
   ┌──── 장중 ────┐ 종가          ┌─ 시가 ────────┐
   │              │  ↓            │  ↓            │
   │              │ "사자!" 판단 ──→ 실제 매수 체결 │
   └──────────────┘               └───────────────┘

★ 하루 처리 순서 (매일 반복)
  1) [아침 시가] 어제 "팔자"고 정한 종목 → 매도 체결 (현금 확보 먼저)
  2) [아침 시가] 어제 "사자"고 정한 종목 → 매수 체결
  3) [저녁 종가] 오늘 종가로 내 계좌 총액 계산 (= 에쿼티)
  4) [저녁 종가] 들고 있는 종목: 오늘 팔아야 하나? → 내일 매도 예약
  5) [저녁 종가] 새 종목: 오늘 사야 하나?        → 내일 매수 예약

용어
  - 포지션(position) : 지금 들고 있는 종목 1개의 보유 정보 (몇 주, 얼마에 샀나)
  - 에쿼티(equity)   : 계좌 총액 = 현금 + 보유 주식의 현재 가치
  - 체결(fill)       : 주문이 실제로 거래된 것
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from .strategies.base import Strategy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# [1] Position : 보유 종목 1개의 정보 (매수 영수증 같은 것)
#     (@dataclass = 값만 담아두는 간단한 "상자"를 만드는 파이썬 문법)
# ---------------------------------------------------------------------
@dataclass
class Position:
    symbol: str                 # 종목 코드
    shares: int                 # 보유 주식 수
    entry_date: pd.Timestamp    # 산 날 (체결일)
    entry_price: float          # 산 가격 (슬리피지 포함 실제 체결가)
    entry_fee: float            # 살 때 낸 수수료
    signal_date: pd.Timestamp   # "사자"고 판단한 날 (체결 전날)
    meta: dict                  # 전략이 기억해 둔 정보 (박스 하단·상단)
    bars_held: int = 0          # 지금까지 보유한 거래일 수


# ---------------------------------------------------------------------
# [2] Account : 가짜 증권 계좌 (현금, 보유 종목, 거래 내역 관리)
#     한 계좌 = 한 통화. 미국 계좌(달러)와 한국 계좌(원)는 따로 만든다.
# ---------------------------------------------------------------------
@dataclass
class Account:
    cash: float                 # 현금
    commission: float           # 수수료율       (예: 0.0005 = 0.05%)
    slippage: float             # 슬리피지율     (예: 0.0005 = 0.05%)
    sell_tax: float             # 매도 세율      (예: 0.002 = 0.2%, 한국만)
    positions: dict[str, Position] = field(default_factory=dict)  # 보유 종목들 {종목코드: Position}
    trades: list[dict] = field(default_factory=list)   # 완료된 거래 (산 것 + 판 것 한 세트)
    orders: list[dict] = field(default_factory=list)   # 모든 주문 기록 (매수·매도 각각)

    def max_buy_shares(self, open_price: float, budget: float) -> int:
        """예산(budget)과 현금 중 작은 금액으로 살 수 있는 최대 주식 수.
        1주 가격 = 시가 × (1 + 슬리피지) × (1 + 수수료). 소수점 주식은 안 산다(내림)."""
        unit = open_price * (1 + self.slippage) * (1 + self.commission)
        return max(0, math.floor(min(budget, self.cash) / unit))

    def buy(self, symbol: str, date, open_price: float, shares: int, signal_date, meta: dict,
            reason: str = "") -> Position:
        """매수: 현금 감소, 보유 종목 추가.
          체결가 = 시가 × (1 + 슬리피지)      ← 살 때는 조금 비싸게 사진다고 가정
          수수료 = 체결가 × 주식 수 × 수수료율
          현금  -= 체결가 × 주식 수 + 수수료
        """
        price = open_price * (1 + self.slippage)
        value = price * shares
        fee = value * self.commission
        if value + fee > self.cash + 1e-9:
            raise ValueError(f"현금 부족: 필요 {value + fee:.2f}, 보유 {self.cash:.2f}")
        self.cash -= value + fee
        pos = Position(symbol, shares, pd.Timestamp(date), price, fee, pd.Timestamp(signal_date), meta)
        self.positions[symbol] = pos
        self.orders.append(dict(date=date, symbol=symbol, side="BUY", shares=shares, price=price,
                                fee=fee, tax=0.0, signal_date=signal_date, reason=reason, **meta))
        log.info("BUY  %s %s x%d @ %.4f fee=%.4f | %s", date.date() if hasattr(date, "date") else date,
                 symbol, shares, price, fee, reason)
        return pos

    def sell(self, symbol: str, date, open_price: float, signal_date, reason: str = "") -> dict:
        """매도: 보유 종목 제거, 현금 증가, 손익 계산.
          체결가   = 시가 × (1 - 슬리피지)    ← 팔 때는 조금 싸게 팔린다고 가정
          수수료   = 매도금액 × 수수료율
          세금     = 매도금액 × 세율 (한국만)
          받는 돈  = 매도금액 - 수수료 - 세금
          손익     = 받는 돈 - (산 금액 + 살 때 수수료)
        """
        pos = self.positions.pop(symbol)
        price = open_price * (1 - self.slippage)
        value = price * pos.shares
        fee = value * self.commission
        tax = value * self.sell_tax
        proceeds = value - fee - tax
        self.cash += proceeds
        cost = pos.entry_price * pos.shares + pos.entry_fee   # 이 종목에 들어간 총 돈
        pnl = proceeds - cost                                 # 손익 (PnL = Profit and Loss)
        # 거래 1건 기록 (리포트의 승률·손익비 계산에 쓰인다)
        trade = dict(symbol=symbol, entry_signal_date=pos.signal_date, entry_date=pos.entry_date,
                     entry_price=pos.entry_price, exit_signal_date=signal_date, exit_date=date,
                     exit_price=price, shares=pos.shares, fees=pos.entry_fee + fee, tax=tax,
                     pnl=pnl, return_pct=pnl / cost, bars_held=pos.bars_held, exit_reason=reason,
                     **pos.meta)
        self.trades.append(trade)
        self.orders.append(dict(date=date, symbol=symbol, side="SELL", shares=pos.shares, price=price,
                                fee=fee, tax=tax, signal_date=signal_date, reason=reason, **pos.meta))
        log.info("SELL %s %s x%d @ %.4f pnl=%.2f (%.2f%%) | %s",
                 date.date() if hasattr(date, "date") else date, symbol, pos.shares, price, pnl,
                 trade["return_pct"] * 100, reason)
        return trade


# ---------------------------------------------------------------------
# [3] BacktestResult : 백테스트가 끝나고 돌려주는 결과 묶음 (report.py가 사용)
# ---------------------------------------------------------------------
@dataclass
class BacktestResult:
    equity: pd.Series                 # 날짜별 계좌 총액 (에쿼티 커브 그래프의 재료)
    trades: pd.DataFrame              # 완료된 거래 목록
    orders: pd.DataFrame              # 모든 주문 기록
    open_positions: pd.DataFrame      # 끝날 때까지 안 팔고 들고 있는 종목
    signals: dict[str, pd.DataFrame]  # 종목별 신호표
    last_scan: pd.DataFrame           # 마지막 날 매수 조건에 맞은 종목 (오늘의 후보)


# ---------------------------------------------------------------------
# [4] run_backtest : 백테스트 본체 — 하루씩 넘기며 매매
# ---------------------------------------------------------------------
def run_backtest(data: dict[str, pd.DataFrame], strategy: Strategy, market_cfg: dict,
                 risk_cfg: dict, start=None, end=None) -> BacktestResult:
    """
    data       : {종목코드: 주가표}   (data_loader.py가 받아온 것)
    strategy   : 사용할 전략         (BoxStrategy)
    market_cfg : config.yaml의 markets.us 또는 markets.kr (시작 자금, 비용)
    risk_cfg   : config.yaml의 risk (종목당 10%, 최대 10종목, 최대 60일)
    """
    # ---- 준비: 계좌 만들기 ----
    costs = market_cfg["costs"]
    acct = Account(cash=float(market_cfg["initial_cash"]), commission=float(costs["commission"]),
                   slippage=float(costs["slippage"]), sell_tax=float(costs["sell_tax"]))
    max_weight = float(risk_cfg["max_weight"])         # 종목당 최대 비중
    max_positions = int(risk_cfg["max_positions"])     # 최대 보유 종목 수
    max_hold = risk_cfg.get("max_hold_days")           # 최대 보유 기간

    # ---- 준비: 모든 종목의 신호표를 미리 계산 ----
    # 전략은 각 날짜에 "그날까지의 정보"만 쓰도록 만들어져 있다 (tests에서 검증).
    # 그래서 10년치를 한 번에 계산해도 미래 정보가 섞이지 않는다.
    signals = {s: strategy.generate_signals(df) for s, df in data.items() if len(df)}

    # ---- 준비: 달력 만들기 (모든 종목의 거래일을 합쳐서 정렬, 기간 자르기) ----
    all_dates = sorted(set().union(*[df.index for df in data.values()])) if data else []
    dates = [d for d in all_dates
             if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]

    # ---- 준비: 예약 주문함 (오늘 저녁에 정하고, 내일 아침에 체결) ----
    pending_buys: dict[str, tuple] = {}   # 내일 살 종목  {종목: (판단일, 박스정보, 매수근거)}
    pending_sells: dict[str, tuple] = {}  # 내일 팔 종목  {종목: (판단일, 매도사유)}
    last_close: dict[str, float] = {}     # 종목별 가장 최근 종가 (거래정지일 대비)
    equity_curve: dict[pd.Timestamp, float] = {}   # 날짜별 계좌 총액
    prev_equity = acct.cash               # 어제 저녁 계좌 총액 (오늘 매수 금액 계산용)

    # ================= 하루씩 넘기기 =================
    for d in dates:
        # 오늘 거래된 종목들의 일봉 (거래정지 등으로 오늘 데이터가 없는 종목은 빠짐)
        bars = {s: df.loc[d] for s, df in data.items() if d in df.index}

        # ---- 1) 아침 시가: 매도 먼저 체결 (현금을 먼저 확보) ----
        for sym in list(pending_sells):
            if sym in bars:                     # 오늘 거래가 되는 종목이면
                sig_date, reason = pending_sells.pop(sym)
                acct.sell(sym, d, float(bars[sym]["open"]), sig_date, reason)
            # 오늘 거래정지면 예약을 유지 → 거래 재개일에 판다

        # ---- 2) 아침 시가: 매수 체결 ----
        for sym, (sig_date, meta, reason) in list(pending_buys.items()):
            pending_buys.pop(sym)
            if sym not in bars or sym in acct.positions:   # 오늘 거래 안 되거나 이미 보유 중이면 취소
                continue
            if len(acct.positions) >= max_positions:       # 보유 종목 수 꽉 찼으면 취소
                continue
            open_px = float(bars[sym]["open"])
            # 살 금액 = 어제 계좌 총액 × 10% (단, 현금이 그보다 적으면 현금만큼)
            shares = acct.max_buy_shares(open_px, prev_equity * max_weight)
            if shares < 1:
                log.info("SKIP %s %s 수량 0 (현금 %.2f)", d.date(), sym, acct.cash)
                continue
            acct.buy(sym, d, open_px, shares, sig_date, meta, reason)

        # ---- 3) 저녁 종가: 계좌 총액 계산 ----
        for sym, bar in bars.items():
            last_close[sym] = float(bar["close"])
        # 총액 = 현금 + Σ(보유 주식 수 × 오늘 종가)
        equity = acct.cash + sum(p.shares * last_close[p.symbol] for p in acct.positions.values())
        equity_curve[d] = equity
        prev_equity = equity

        # ---- 4) 저녁 종가: 보유 종목 — 팔아야 하나? ----
        for sym, pos in acct.positions.items():
            if sym not in bars or sym in pending_sells:
                continue
            pos.bars_held += 1                                    # 보유일 +1
            reason = strategy.exit_reason(pos.meta, bars[sym])    # 전략에게 질문 (손절/익절?)
            if reason is None and max_hold is not None and pos.bars_held >= int(max_hold):
                reason = f"max_hold {pos.bars_held}bars"          # 60일 넘게 보유 → 정리
            if reason:
                pending_sells[sym] = (d, reason)                  # 내일 아침 매도 예약

        # ---- 5) 저녁 종가: 새로 살 종목 고르기 ----
        # 빈 자리 = 최대 종목 수 - (보유 종목 수 - 내일 팔 종목 수)
        slots = max_positions - (len(acct.positions) - len(pending_sells))
        if slots > 0:
            cands = []
            for sym in bars:
                if sym in acct.positions or sym in pending_buys:  # 이미 보유 중이면 패스
                    continue
                row = signals[sym].loc[d]                         # 이 종목의 오늘 신호
                if bool(row["entry"]):
                    cands.append((float(row["score"]), sym, row))
            cands.sort(key=lambda x: x[0], reverse=True)          # 점수 높은 순 (바닥에 가까운 순)
            for score, sym, row in cands[:slots]:                 # 빈 자리만큼만
                pending_buys[sym] = (d, strategy.entry_meta(row), row["reason"])  # 내일 아침 매수 예약
                log.info("SIGNAL %s %s score=%.3f | %s", d.date(), sym, score, row["reason"])
    # ================= 하루 끝 =================

    # ---- 결과 정리 ----
    equity = pd.Series(equity_curve, name="equity", dtype=float)

    # 마지막 날 매수 조건에 맞은 종목 목록 (= "오늘의 스캔 결과", CSV로 저장됨)
    last_scan = pd.DataFrame()
    if dates:
        d = dates[-1]
        rows = [signals[s].loc[d].rename(s) for s in signals if d in signals[s].index
                and bool(signals[s].loc[d, "entry"])]
        if rows:
            last_scan = pd.DataFrame(rows).sort_values("score", ascending=False)
            last_scan.index.name = "symbol"

    # 끝까지 안 팔린 보유 종목 (평가 손익 = 마지막 종가 ÷ 산 가격 - 1)
    open_pos = pd.DataFrame([
        dict(symbol=p.symbol, shares=p.shares, entry_date=p.entry_date, entry_price=p.entry_price,
             last_close=last_close.get(p.symbol), bars_held=p.bars_held,
             unrealized_pct=last_close.get(p.symbol, p.entry_price) / p.entry_price - 1, **p.meta)
        for p in acct.positions.values()])
    return BacktestResult(equity, pd.DataFrame(acct.trades), pd.DataFrame(acct.orders), open_pos,
                          signals, last_scan)
