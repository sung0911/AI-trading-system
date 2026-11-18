import pandas as pd
import pytest

from src.engine import Account, run_backtest
from src.report import compute_metrics
from src.strategies import BoxStrategy

MARKET = {"initial_cash": 1_000_000, "costs": {"commission": 0.001, "slippage": 0.002,
                                               "sell_tax": 0.003}}
RISK = {"max_weight": 0.10, "max_positions": 10, "max_hold_days": 60}


def test_account_buy_sell_accounting():
    acct = Account(cash=10_000.0, commission=0.001, slippage=0.002, sell_tax=0.003)
    d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-10")
    acct.buy("X", d1, 100.0, 10, d1, {"box_low": 1, "box_high": 2})
    buy_px = 100.0 * 1.002                       # 100.2
    buy_fee = buy_px * 10 * 0.001                # 1.002
    assert acct.cash == pytest.approx(10_000 - 1002.0 - buy_fee)

    tr = acct.sell("X", d2, 110.0, d2, "test")
    sell_px = 110.0 * 0.998                      # 109.78
    gross = sell_px * 10                         # 1097.8
    sell_fee, tax = gross * 0.001, gross * 0.003
    expected_cash = 10_000 - 1002.0 - buy_fee + gross - sell_fee - tax
    assert acct.cash == pytest.approx(expected_cash)
    assert tr["pnl"] == pytest.approx(gross - sell_fee - tax - (1002.0 + buy_fee))
    assert tr["fees"] == pytest.approx(buy_fee + sell_fee)
    assert tr["tax"] == pytest.approx(tax)
    assert acct.cash - 10_000 == pytest.approx(tr["pnl"])
    assert not acct.positions


def test_buy_rejects_insufficient_cash():
    acct = Account(cash=100.0, commission=0.0, slippage=0.0, sell_tax=0.0)
    with pytest.raises(ValueError):
        acct.buy("X", pd.Timestamp("2024-01-02"), 50.0, 3, pd.Timestamp("2024-01-02"), {})
    assert acct.max_buy_shares(50.0, 1e9) == 2


def test_entry_fills_next_open_and_take_profit(double_bottom_df, box_params):
    df = double_bottom_df
    res = run_backtest({"X": df}, BoxStrategy(box_params), MARKET, RISK)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    i_sig = df.index.get_loc(tr["entry_signal_date"])
    # 체결은 신호 다음 봉 시가 (+슬리피지)
    assert tr["entry_date"] == df.index[i_sig + 1]
    assert tr["entry_price"] == pytest.approx(df["open"].iloc[i_sig + 1] * 1.002)
    assert tr["exit_reason"].startswith("take_profit")
    i_exit_sig = df.index.get_loc(tr["exit_signal_date"])
    assert tr["exit_date"] == df.index[i_exit_sig + 1]
    assert tr["exit_price"] == pytest.approx(df["open"].iloc[i_exit_sig + 1] * 0.998)
    assert tr["pnl"] > 0
    # 비중 상한 10%
    assert tr["entry_price"] * tr["shares"] <= 1_000_000 * 0.10
    # 에쿼티 마지막 값 = 현금 (모두 청산됨)
    assert res.equity.iloc[-1] == pytest.approx(1_000_000 + tr["pnl"])


def test_stop_loss_on_box_breakdown(breakdown_df, box_params):
    df = breakdown_df
    res = run_backtest({"X": df}, BoxStrategy(box_params), MARKET, RISK)
    assert len(res.trades) >= 1
    tr = res.trades.iloc[0]
    assert tr["exit_reason"].startswith("stop_box_break")
    i_exit_sig = df.index.get_loc(tr["exit_signal_date"])
    # 손절 판단일 종가 < 박스하단*0.95, 체결은 다음날 시가
    assert df["close"].iloc[i_exit_sig] < tr["box_low"] * 0.95
    assert df["close"].iloc[i_exit_sig - 1] >= tr["box_low"] * 0.95
    assert tr["exit_date"] == df.index[i_exit_sig + 1]
    assert tr["pnl"] < 0


def test_max_positions_limit(double_bottom_df, box_params):
    data = {f"S{i}": double_bottom_df for i in range(5)}
    risk = dict(RISK, max_positions=2)
    res = run_backtest(data, BoxStrategy(box_params), MARKET, risk)
    assert len(res.trades) == 2


def test_metrics(double_bottom_df, breakdown_df, box_params):
    res = run_backtest({"A": double_bottom_df, "B": breakdown_df}, BoxStrategy(box_params),
                       MARKET, RISK)
    m = compute_metrics(res, bench=double_bottom_df["close"])
    assert m["num_trades"] == len(res.trades)
    assert 0 <= m["win_rate"] <= 1
    assert m["mdd"] <= 0
    assert "bench_total_return" in m
