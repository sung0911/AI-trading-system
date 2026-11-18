"""
=====================================================================
 report.py : 백테스트 "성적표" 만들기 — 수익률·위험 지표 계산, 그래프·차트·CSV 저장
=====================================================================

engine.py가 넘겨준 결과(날짜별 계좌 총액, 거래 목록)를 가공해서
사람이 볼 수 있는 성적표로 만든다.

주요 성적 지표 (초보자용 설명)
  누적수익률   : 처음 돈 대비 얼마나 늘었나.            (최종 ÷ 처음 - 1)
  CAGR        : 연평균 복리 수익률. "매년 평균 몇 %씩 불었나"
                기간이 다른 전략끼리 비교할 때 쓴다.
  MDD         : 최대 낙폭. 고점에서 가장 크게 떨어졌을 때 몇 % 잃었나.
                "최악의 순간에 얼마나 아팠나" → 버틸 수 있는 전략인지 판단
  승률        : 전체 거래 중 수익 낸 거래의 비율
  손익비      : 평균 수익 ÷ 평균 손실. 1보다 크면 "벌 때 크게, 잃을 때 작게"
  Profit Factor: 총 수익 금액 ÷ 총 손실 금액. 1보다 크면 전체적으로 돈을 번 전략
  벤치마크     : 같은 기간 지수(S&P 500, 코스피)를 그냥 사서 들고 있었을 때의 성적

만드는 파일 (reports/backtest_시장_box/ 폴더)
  summary.md         : 성적표 요약 + 한계·주의사항
  metrics.json       : 성적 지표 원본 숫자
  trades.csv         : 완료된 거래 목록 (엑셀로 열 수 있음)
  orders.csv         : 모든 주문 기록
  open_positions.csv : 아직 들고 있는 종목
  equity.csv / .png  : 날짜별 계좌 총액 + 그래프 (전략 vs 벤치마크, 낙폭)
  scan_날짜.csv       : 마지막 날 매수 조건에 맞은 종목 (오늘의 후보)
  charts/종목.html    : 종목별 캔들 차트 (박스·손절선·매수·매도 표시, 브라우저로 열기)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BacktestResult

log = logging.getLogger(__name__)

# 그래프 색상 (색약자도 구분되도록 검증된 팔레트에서 선택)
C_STRATEGY = "#2a78d6"   # 파랑 : 전략, 매수
C_BENCH = "#eb6834"      # 주황 : 벤치마크, 매도
C_UP = "#1baf7a"         # 초록 : 상승 캔들
C_DOWN = "#e34948"       # 빨강 : 하락 캔들, 손절선
C_INK = "#52514e"        # 회색 : 글자

# 리포트 맨 아래 붙는 "이 결과를 그대로 믿으면 안 되는 이유" 안내문
LIMITATIONS = """\
## 한계와 주의사항
- **생존 편향**: 현재 시점의 종목 목록만 사용했다. 상장폐지·지수 편출 종목이 빠져 있어 결과가 실제보다 좋게 나올 수 있다.
- **수정주가**: 미국은 yfinance 수정주가(배당·분할 반영)를 쓴다. 국내 FinanceDataReader 가격은 수정 방식이 소스에 따라 다를 수 있어 분할·배당 시점에 왜곡될 수 있다.
- **체결 가정**: t일 종가에 신호를 내고 t+1일 시가에 전량 체결한다고 가정했다. 슬리피지는 고정 비율이며 호가 단위, 거래량 한계, 갭 상·하한가 미체결은 반영하지 않았다.
- **손절 지연**: 손절도 종가 확인 후 다음 날 시가에 체결하므로 갭하락 시 설정한 손절폭(-5%)보다 손실이 커질 수 있다.
- **미래 참조**: 박스는 t일 직전 N봉만으로 계산하며 테스트로 검증했다. 그래도 결과가 지나치게 좋으면 위 항목을 먼저 의심할 것.
- 세율·수수료는 가정치이며 실제 증권사·세법 기준과 다를 수 있다.
"""


# ---------------------------------------------------------------------
# [1] 지표 계산 함수들
# ---------------------------------------------------------------------
def _cagr(eq: pd.Series) -> float:
    """연평균 복리 수익률. (최종 ÷ 처음) ^ (1 ÷ 년수) - 1
    예) 10년 동안 2배 → 2^(1/10) - 1 = 약 7.2%/년"""
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return float("nan")
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def _mdd(eq: pd.Series) -> float:
    """최대 낙폭. 매일 "지금까지의 최고점 대비 몇 % 아래인가"를 구하고 그중 가장 나쁜 값.
    예) 1억 → 1.2억(최고점) → 9천만 이면 MDD = 9천 ÷ 1.2억 - 1 = -25%"""
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def compute_metrics(res: BacktestResult, bench: pd.Series | None = None) -> dict:
    """모든 성적 지표를 계산해서 사전(dict)으로 돌려준다."""
    eq = res.equity      # 날짜별 계좌 총액
    t = res.trades       # 완료된 거래들

    # 계좌 전체 성적
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

    # 거래별 성적 (거래가 1건 이상일 때만)
    if len(t):
        wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]          # 수익 거래 / 손실 거래
        avg_win = wins["return_pct"].mean() if len(wins) else 0.0     # 평균 수익률
        avg_loss = losses["return_pct"].mean() if len(losses) else 0.0  # 평균 손실률
        m.update(
            win_rate=float(len(wins) / len(t)),                       # 승률
            avg_win_pct=float(avg_win),
            avg_loss_pct=float(avg_loss),
            payoff_ratio=float(avg_win / abs(avg_loss)) if avg_loss else float("inf"),   # 손익비
            profit_factor=float(wins["pnl"].sum() / abs(losses["pnl"].sum()))         # 총수익÷총손실
            if len(losses) and losses["pnl"].sum() != 0 else float("inf"),
            avg_bars_held=float(t["bars_held"].mean()),               # 평균 보유일
            # 청산 사유별 횟수 (익절 몇 번, 손절 몇 번, 기간만료 몇 번)
            exit_reasons=t["exit_reason"].str.split(" ").str[0].value_counts().to_dict(),
        )

    # 벤치마크(지수) 성적 — 같은 기간으로 맞춰서 비교
    if bench is not None and len(eq):
        b = bench.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            m.update(bench_total_return=float(b.iloc[-1] / b.iloc[0] - 1),
                     bench_cagr=_cagr(b), bench_mdd=_mdd(b))
            m["excess_return"] = m["total_return"] - m["bench_total_return"]   # 초과수익
    return m


# ---------------------------------------------------------------------
# [2] 성적표 글(summary.md) 만들기
# ---------------------------------------------------------------------
def _fmt_pct(v) -> str:
    """숫자를 % 글자로. 0.1234 → '12.34%', 값이 없으면 '-'"""
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2%}"


def metrics_markdown(name: str, m: dict, currency: str, failed: list[str]) -> str:
    """성적 지표를 마크다운 표로 만든다 (.md = 메모장·GitHub에서 표로 보이는 글 형식)."""
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


# ---------------------------------------------------------------------
# [3] 그래프: 에쿼티 커브 (계좌 총액 변화) — PNG 그림 파일
# ---------------------------------------------------------------------
def plot_equity(eq: pd.Series, bench: pd.Series | None, path: Path, title: str) -> None:
    """
    위쪽 그래프: 전략(파랑) vs 벤치마크(주황). 둘 다 시작을 100으로 맞춰서 비교.
                 (예: 150이면 50% 수익)
    아래 그래프: 낙폭. 최고점 대비 몇 % 아래에 있는지 (0에 가까울수록 좋음)
    """
    import matplotlib
    matplotlib.use("Agg")                # 화면 창 없이 파일로만 저장
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    norm = eq / eq.iloc[0] * 100         # 시작 = 100 으로 환산
    ax1.plot(norm.index, norm, color=C_STRATEGY, lw=2, label="Strategy")
    if bench is not None:
        b = bench.reindex(eq.index).ffill().dropna()
        if len(b):
            ax1.plot(b.index, b / b.iloc[0] * 100, color=C_BENCH, lw=2, label="Benchmark")
    ax1.set_title(title, loc="left", color="#0b0b0b")
    ax1.set_ylabel("Indexed (start = 100)", color=C_INK)
    ax1.legend(frameon=False, loc="upper left")
    dd = (eq / eq.cummax() - 1) * 100    # 낙폭(%)
    ax2.fill_between(dd.index, dd, 0, color=C_STRATEGY, alpha=0.35, lw=0)
    ax2.set_ylabel("Drawdown %", color=C_INK)
    for ax in (ax1, ax2):                # 테두리·격자를 연하게 정리
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------
# [4] 종목별 차트 — HTML 파일 (브라우저로 열면 확대·마우스 올리기 가능)
# ---------------------------------------------------------------------
def plot_symbol(df: pd.DataFrame, trades: pd.DataFrame, symbol: str, lookback: int,
                stop_pct: float, path: Path) -> None:
    """
    캔들 차트 위에 표시하는 것
      파란 사각형 : 매수 시점에 인식한 박스 (박스 계산 60일 ~ 매도일)
      빨간 점선   : 손절선 (박스 하단 × 0.95)
      ▲ 파랑     : 매수 지점
      ▼ 주황     : 매도 지점 (마우스 올리면 수익률·청산 사유 표시)

    캔들(봉) 읽는 법: 몸통 = 시가~종가, 위아래 꼬리 = 고가·저가
                     초록 = 종가가 시가보다 높음(상승), 빨강 = 하락
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    # 캔들 차트
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name=symbol,
                                 increasing_line_color=C_UP, decreasing_line_color=C_DOWN))
    pos = {d: i for i, d in enumerate(df.index)}   # 날짜 → 몇 번째 봉인지
    for _, tr in trades.iterrows():
        # 박스 사각형: 신호일 기준 60봉 전부터 매도일까지
        i_sig = pos.get(tr["entry_signal_date"])
        x0 = df.index[max(0, i_sig - lookback)] if i_sig is not None else tr["entry_signal_date"]
        x1 = tr["exit_date"]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=tr["box_low"], y1=tr["box_high"],
                      line=dict(color=C_STRATEGY, width=1), fillcolor=C_STRATEGY, opacity=0.08)
        # 손절선
        fig.add_shape(type="line", x0=tr["entry_date"], x1=x1,
                      y0=tr["box_low"] * (1 - stop_pct), y1=tr["box_low"] * (1 - stop_pct),
                      line=dict(color=C_DOWN, width=1, dash="dot"))
    if len(trades):
        # 매수 표시 ▲
        fig.add_trace(go.Scatter(
            x=trades["entry_date"], y=trades["entry_price"], mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=12, color=C_STRATEGY,
                        line=dict(color="#fcfcfb", width=2)),
            text=trades["entry_signal_date"].astype(str),
            hovertemplate="매수 %{x|%Y-%m-%d}<br>%{y:.2f}<br>신호일 %{text}<extra></extra>"))
        # 매도 표시 ▼
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


# ---------------------------------------------------------------------
# [5] 전체 리포트 저장 (위 함수들을 차례로 호출)
# ---------------------------------------------------------------------
def write_report(out_dir: Path, name: str, res: BacktestResult, data: dict[str, pd.DataFrame],
                 bench: pd.Series | None, currency: str, strategy_params: dict,
                 max_charts: int, failed: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 성적 지표 계산 → JSON(숫자 원본), MD(사람이 읽는 표) 저장
    m = compute_metrics(res, bench)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    (out_dir / "summary.md").write_text(metrics_markdown(name, m, currency, failed), encoding="utf-8")

    # 거래·주문·보유 목록 CSV 저장 (utf-8-sig = 엑셀에서 한글 안 깨지게)
    res.trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    res.orders.to_csv(out_dir / "orders.csv", index=False, encoding="utf-8-sig")
    res.open_positions.to_csv(out_dir / "open_positions.csv", index=False, encoding="utf-8-sig")

    # 계좌 총액 그래프 + 마지막 날 스캔 결과
    if len(res.equity):
        res.equity.to_csv(out_dir / "equity.csv", encoding="utf-8-sig")
        plot_equity(res.equity, bench, out_dir / "equity.png", f"{name} equity vs benchmark")
        scan_date = res.equity.index[-1].strftime("%Y%m%d")
        res.last_scan.to_csv(out_dir / f"scan_{scan_date}.csv", encoding="utf-8-sig")

    # 거래가 많은 종목부터 최대 max_charts개 차트 저장
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
