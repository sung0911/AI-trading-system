"""백테스트 성과 지표, 리포트, 차트 저장."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BacktestResult

log = logging.getLogger(__name__)

# 참조 팔레트 (dataviz 기본 팔레트 1·2번 슬롯)
C_STRATEGY = "#2a78d6"
C_BENCH = "#eb6834"
C_UP = "#1baf7a"
C_DOWN = "#e34948"
C_INK = "#52514e"

LIMITATIONS = """\
## 한계와 주의사항
- **생존 편향**: 현재 시점의 종목 목록만 사용했다. 상장폐지·지수 편출 종목이 빠져 있어 결과가 실제보다 좋게 나올 수 있다.
- **수정주가**: 미국은 yfinance 수정주가(배당·분할 반영)를 쓴다. 국내 FinanceDataReader 가격은 수정 방식이 소스에 따라 다를 수 있어 분할·배당 시점에 왜곡될 수 있다.
- **체결 가정**: t일 종가에 신호를 내고 t+1일 시가에 전량 체결한다고 가정했다. 슬리피지는 고정 비율이며 호가 단위, 거래량 한계, 갭 상·하한가 미체결은 반영하지 않았다.
- **손절 지연**: 손절도 종가 확인 후 다음 날 시가에 체결하므로 갭하락 시 설정한 손절폭(-5%)보다 손실이 커질 수 있다.
- **미래 참조**: 박스는 t일 직전 N봉만으로 계산하며 테스트로 검증했다. 그래도 결과가 지나치게 좋으면 위 항목을 먼저 의심할 것.
- 세율·수수료는 가정치이며 실제 증권사·세법 기준과 다를 수 있다.
"""


def _cagr(eq: pd.Series) -> float:
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return float("nan")
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def _mdd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def compute_metrics(res: BacktestResult, bench: pd.Series | None = None) -> dict:
    eq = res.equity
    t = res.trades
    m: dict = {
        "start": str(eq.index[0].date()) if len(eq) else None,
        "end": str(eq.index[-1].date()) if len(eq) else None,
        "initial_equity": float(eq.iloc[0]) if len(eq) else None,
        "final_equity": float(eq.iloc[-1]) if len(eq) else None,
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else float("nan"),
        "cagr": _cagr(eq),
        "mdd": _mdd(eq),
        "num_trades": int(len(t)),
        "open_positions": int(len(res.open_positions)),
    }
    if len(t):
        wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
        avg_win = wins["return_pct"].mean() if len(wins) else 0.0
        avg_loss = losses["return_pct"].mean() if len(losses) else 0.0
        m.update(
            win_rate=float(len(wins) / len(t)),
            avg_win_pct=float(avg_win),
            avg_loss_pct=float(avg_loss),
            payoff_ratio=float(avg_win / abs(avg_loss)) if avg_loss else float("inf"),
            profit_factor=float(wins["pnl"].sum() / abs(losses["pnl"].sum()))
            if len(losses) and losses["pnl"].sum() != 0 else float("inf"),
            avg_bars_held=float(t["bars_held"].mean()),
            exit_reasons=t["exit_reason"].str.split(" ").str[0].value_counts().to_dict(),
        )
    if bench is not None and len(eq):
        b = bench.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            m.update(bench_total_return=float(b.iloc[-1] / b.iloc[0] - 1),
                     bench_cagr=_cagr(b), bench_mdd=_mdd(b))
            m["excess_return"] = m["total_return"] - m["bench_total_return"]
    return m


def _fmt_pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2%}"


def metrics_markdown(name: str, m: dict, currency: str, failed: list[str]) -> str:
    rows = [
        ("기간", f"{m.get('start')} ~ {m.get('end')}"),
        ("최종 자산", f"{m.get('final_equity', 0):,.0f} {currency}"),
        ("누적수익률", _fmt_pct(m.get("total_return"))),
        ("CAGR", _fmt_pct(m.get("cagr"))),
        ("MDD", _fmt_pct(m.get("mdd"))),
        ("거래 횟수", str(m.get("num_trades"))),
        ("승률", _fmt_pct(m.get("win_rate"))),
        ("평균 수익 / 평균 손실", f"{_fmt_pct(m.get('avg_win_pct'))} / {_fmt_pct(m.get('avg_loss_pct'))}"),
        ("손익비 (평균수익/평균손실)", f"{m.get('payoff_ratio', float('nan')):.2f}"),
        ("Profit Factor", f"{m.get('profit_factor', float('nan')):.2f}"),
        ("평균 보유 기간(거래일)", f"{m.get('avg_bars_held', float('nan')):.1f}"),
        ("미청산 포지션", str(m.get("open_positions"))),
        ("벤치마크 누적수익률", _fmt_pct(m.get("bench_total_return"))),
        ("벤치마크 CAGR / MDD", f"{_fmt_pct(m.get('bench_cagr'))} / {_fmt_pct(m.get('bench_mdd'))}"),
        ("초과수익 (전략-벤치마크)", _fmt_pct(m.get("excess_return"))),
    ]
    lines = [f"# 백테스트 리포트: {name}", "", "| 항목 | 값 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    if m.get("exit_reasons"):
        lines += ["", "청산 사유: " + ", ".join(f"{k} {v}회" for k, v in m["exit_reasons"].items())]
    if failed:
        lines += ["", f"데이터 수집 실패로 제외된 종목: {', '.join(failed)}"]
    lines += ["", LIMITATIONS]
    return "\n".join(lines)


def plot_equity(eq: pd.Series, bench: pd.Series | None, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    norm = eq / eq.iloc[0] * 100
    ax1.plot(norm.index, norm, color=C_STRATEGY, lw=2, label="Strategy")
    if bench is not None:
        b = bench.reindex(eq.index).ffill().dropna()
        if len(b):
            ax1.plot(b.index, b / b.iloc[0] * 100, color=C_BENCH, lw=2, label="Benchmark")
    ax1.set_title(title, loc="left", color="#0b0b0b")
    ax1.set_ylabel("Indexed (start = 100)", color=C_INK)
    ax1.legend(frameon=False, loc="upper left")
    dd = (eq / eq.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd, 0, color=C_STRATEGY, alpha=0.35, lw=0)
    ax2.set_ylabel("Drawdown %", color=C_INK)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_symbol(df: pd.DataFrame, trades: pd.DataFrame, symbol: str, lookback: int,
                stop_pct: float, path: Path) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name=symbol,
                                 increasing_line_color=C_UP, decreasing_line_color=C_DOWN))
    pos = {d: i for i, d in enumerate(df.index)}
    for _, tr in trades.iterrows():
        i_sig = pos.get(tr["entry_signal_date"])
        x0 = df.index[max(0, i_sig - lookback)] if i_sig is not None else tr["entry_signal_date"]
        x1 = tr["exit_date"]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=tr["box_low"], y1=tr["box_high"],
                      line=dict(color=C_STRATEGY, width=1), fillcolor=C_STRATEGY, opacity=0.08)
        fig.add_shape(type="line", x0=tr["entry_date"], x1=x1,
                      y0=tr["box_low"] * (1 - stop_pct), y1=tr["box_low"] * (1 - stop_pct),
                      line=dict(color=C_DOWN, width=1, dash="dot"))
    if len(trades):
        fig.add_trace(go.Scatter(
            x=trades["entry_date"], y=trades["entry_price"], mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=12, color=C_STRATEGY,
                        line=dict(color="#fcfcfb", width=2)),
            text=trades["entry_signal_date"].astype(str),
            hovertemplate="매수 %{x|%Y-%m-%d}<br>%{y:.2f}<br>신호일 %{text}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=trades["exit_date"], y=trades["exit_price"], mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=12, color=C_BENCH,
                        line=dict(color="#fcfcfb", width=2)),
            text=[f"{r:.2%} / {e.split(' ')[0]}" for r, e in
                  zip(trades["return_pct"], trades["exit_reason"])],
            hovertemplate="매도 %{x|%Y-%m-%d}<br>%{y:.2f}<br>%{text}<extra></extra>"))
    fig.update_layout(title=f"{symbol} - 박스권 매매 (사각형=매수 시점 박스, 점선=손절선)",
                      xaxis_rangeslider_visible=False, template="plotly_white",
                      hovermode="x", height=650, legend=dict(orientation="h", y=1.02, x=1,
                                                              xanchor="right", yanchor="bottom"))
    fig.write_html(path, include_plotlyjs="cdn")


def write_report(out_dir: Path, name: str, res: BacktestResult, data: dict[str, pd.DataFrame],
                 bench: pd.Series | None, currency: str, strategy_params: dict,
                 max_charts: int, failed: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(res, bench)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    (out_dir / "summary.md").write_text(metrics_markdown(name, m, currency, failed), encoding="utf-8")
    res.trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    res.orders.to_csv(out_dir / "orders.csv", index=False, encoding="utf-8-sig")
    res.open_positions.to_csv(out_dir / "open_positions.csv", index=False, encoding="utf-8-sig")
    if len(res.equity):
        res.equity.to_csv(out_dir / "equity.csv", encoding="utf-8-sig")
        plot_equity(res.equity, bench, out_dir / "equity.png", f"{name} equity vs benchmark")
        scan_date = res.equity.index[-1].strftime("%Y%m%d")
        res.last_scan.to_csv(out_dir / f"scan_{scan_date}.csv", encoding="utf-8-sig")

    if len(res.trades):
        chart_dir = out_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        top = res.trades["symbol"].value_counts().head(max_charts).index
        for sym in top:
            plot_symbol(data[sym], res.trades[res.trades["symbol"] == sym], sym,
                        int(strategy_params["lookback"]), float(strategy_params["stop_below_box_pct"]),
                        chart_dir / f"{sym.replace('^', '')}.html")
    log.info("리포트 저장: %s", out_dir)
    return m
