"""백테스트 엔진.

체결 규칙: t일 종가로 신호 판단 → t+1일 시가로 체결. 같은 봉에서 판단과 체결을 하지 않는다.
하루 처리 순서:
  1) 전날 예약된 매도 → 오늘 시가 체결
  2) 전날 예약된 매수 → 오늘 시가 체결 (전날 종가 기준 평가금액으로 수량 계산)
  3) 오늘 종가로 평가
  4) 보유 종목 청산 판단(전략 규칙 + 최대 보유기간) → 내일 매도 예약
  5) 신규 매수 후보 판단 → 내일 매수 예약
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from .strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float          # 슬리피지 반영 체결가
    entry_fee: float
    signal_date: pd.Timestamp
    meta: dict
    bars_held: int = 0


@dataclass
class Account:
    """단일 통화 계좌. 현금·보유·체결 내역을 관리한다."""
    cash: float
    commission: float
    slippage: float
    sell_tax: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)

    def max_buy_shares(self, open_price: float, budget: float) -> int:
        unit = open_price * (1 + self.slippage) * (1 + self.commission)
        return max(0, math.floor(min(budget, self.cash) / unit))

    def buy(self, symbol: str, date, open_price: float, shares: int, signal_date, meta: dict,
            reason: str = "") -> Position:
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
        pos = self.positions.pop(symbol)
        price = open_price * (1 - self.slippage)
        value = price * pos.shares
        fee = value * self.commission
        tax = value * self.sell_tax
        proceeds = value - fee - tax
        self.cash += proceeds
        cost = pos.entry_price * pos.shares + pos.entry_fee
        pnl = proceeds - cost
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


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    orders: pd.DataFrame
    open_positions: pd.DataFrame
    signals: dict[str, pd.DataFrame]
    last_scan: pd.DataFrame  # 마지막 날 매수 조건 충족 종목


def run_backtest(data: dict[str, pd.DataFrame], strategy: Strategy, market_cfg: dict,
                 risk_cfg: dict, start=None, end=None) -> BacktestResult:
    costs = market_cfg["costs"]
    acct = Account(cash=float(market_cfg["initial_cash"]), commission=float(costs["commission"]),
                   slippage=float(costs["slippage"]), sell_tax=float(costs["sell_tax"]))
    max_weight = float(risk_cfg["max_weight"])
    max_positions = int(risk_cfg["max_positions"])
    max_hold = risk_cfg.get("max_hold_days")

    # 전략은 미래 참조가 없도록 만들어져 있으므로(테스트로 검증) 전체 데이터로 한 번에 계산한다.
    signals = {s: strategy.generate_signals(df) for s, df in data.items() if len(df)}

    all_dates = sorted(set().union(*[df.index for df in data.values()])) if data else []
    dates = [d for d in all_dates
             if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]

    pending_buys: dict[str, tuple] = {}   # symbol -> (signal_date, meta, reason)
    pending_sells: dict[str, tuple] = {}  # symbol -> (signal_date, reason)
    last_close: dict[str, float] = {}
    equity_curve: dict[pd.Timestamp, float] = {}
    prev_equity = acct.cash

    for d in dates:
        bars = {s: df.loc[d] for s, df in data.items() if d in df.index}

        # 1) 매도 체결
        for sym in list(pending_sells):
            if sym in bars:
                sig_date, reason = pending_sells.pop(sym)
                acct.sell(sym, d, float(bars[sym]["open"]), sig_date, reason)

        # 2) 매수 체결 (오늘 봉이 없으면 주문 취소)
        for sym, (sig_date, meta, reason) in list(pending_buys.items()):
            pending_buys.pop(sym)
            if sym not in bars or sym in acct.positions:
                continue
            if len(acct.positions) >= max_positions:
                continue
            open_px = float(bars[sym]["open"])
            shares = acct.max_buy_shares(open_px, prev_equity * max_weight)
            if shares < 1:
                log.info("SKIP %s %s 수량 0 (현금 %.2f)", d.date(), sym, acct.cash)
                continue
            acct.buy(sym, d, open_px, shares, sig_date, meta, reason)

        # 3) 평가
        for sym, bar in bars.items():
            last_close[sym] = float(bar["close"])
        equity = acct.cash + sum(p.shares * last_close[p.symbol] for p in acct.positions.values())
        equity_curve[d] = equity
        prev_equity = equity

        # 4) 청산 판단
        for sym, pos in acct.positions.items():
            if sym not in bars or sym in pending_sells:
                continue
            pos.bars_held += 1
            reason = strategy.exit_reason(pos.meta, bars[sym])
            if reason is None and max_hold is not None and pos.bars_held >= int(max_hold):
                reason = f"max_hold {pos.bars_held}bars"
            if reason:
                pending_sells[sym] = (d, reason)

        # 5) 매수 판단
        slots = max_positions - (len(acct.positions) - len(pending_sells))
        if slots > 0:
            cands = []
            for sym in bars:
                if sym in acct.positions or sym in pending_buys:
                    continue
                row = signals[sym].loc[d]
                if bool(row["entry"]):
                    cands.append((float(row["score"]), sym, row))
            cands.sort(key=lambda x: x[0], reverse=True)
            for score, sym, row in cands[:slots]:
                pending_buys[sym] = (d, strategy.entry_meta(row), row["reason"])
                log.info("SIGNAL %s %s score=%.3f | %s", d.date(), sym, score, row["reason"])

    equity = pd.Series(equity_curve, name="equity", dtype=float)
    last_scan = pd.DataFrame()
    if dates:
        d = dates[-1]
        rows = [signals[s].loc[d].rename(s) for s in signals if d in signals[s].index
                and bool(signals[s].loc[d, "entry"])]
        if rows:
            last_scan = pd.DataFrame(rows).sort_values("score", ascending=False)
            last_scan.index.name = "symbol"
    open_pos = pd.DataFrame([
        dict(symbol=p.symbol, shares=p.shares, entry_date=p.entry_date, entry_price=p.entry_price,
             last_close=last_close.get(p.symbol), bars_held=p.bars_held,
             unrealized_pct=last_close.get(p.symbol, p.entry_price) / p.entry_price - 1, **p.meta)
        for p in acct.positions.values()])
    return BacktestResult(equity, pd.DataFrame(acct.trades), pd.DataFrame(acct.orders), open_pos,
                          signals, last_scan)
