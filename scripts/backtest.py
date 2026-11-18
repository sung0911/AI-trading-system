"""박스권 전략 백테스트 실행.

사용 예:
  python scripts/backtest.py                 # 미국·국내 모두, 데이터 업데이트 후 실행
  python scripts/backtest.py --market us
  python scripts/backtest.py --no-update     # 캐시만 사용
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve, setup_logging  # noqa: E402
from src.data_loader import DataStore  # noqa: E402
from src.engine import run_backtest  # noqa: E402
from src.report import write_report  # noqa: E402
from src.strategies import BoxStrategy  # noqa: E402

log = logging.getLogger("backtest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--market", choices=["us", "kr", "all"], default="all")
    ap.add_argument("--no-update", action="store_true", help="다운로드 없이 캐시만 사용")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_file = setup_logging(cfg, "backtest")
    store = DataStore(cfg["data"], resolve(cfg["data"]["cache_dir"]))
    params = cfg["strategy"]["box"]
    strategy = BoxStrategy(params)
    markets = ["us", "kr"] if args.market == "all" else [args.market]

    for mk in markets:
        mcfg = cfg["markets"][mk]
        symbols = [str(s) for s in mcfg["symbols"]]
        log.info("=== [%s] 종목 %d개 ===", mk, len(symbols))
        if args.no_update:
            data, failed = store.load_many(mk, symbols)
            bench_df = store.load(mk, mcfg["benchmark"])
        else:
            data, failed = store.update_many(mk, symbols)
            try:
                bench_df = store.update(mk, mcfg["benchmark"])
            except Exception as e:  # noqa: BLE001
                log.error("벤치마크 %s 수집 실패: %s", mcfg["benchmark"], e)
                bench_df = None
        if not data:
            log.error("[%s] 데이터가 없어 건너뜀", mk)
            continue
        res = run_backtest(data, strategy, mcfg, cfg["risk"],
                           cfg["backtest"].get("start"), cfg["backtest"].get("end"))
        out = resolve(cfg["report"]["out_dir"]) / f"backtest_{mk}_box"
        m = write_report(out, f"{mk.upper()} / box", res, data,
                         bench_df["close"] if bench_df is not None else None, mcfg["currency"],
                         params, int(cfg["report"]["max_symbol_charts"]), failed)
        log.info("[%s] 누적 %.2f%% | CAGR %.2f%% | MDD %.2f%% | 거래 %d회 | 승률 %s",
                 mk, m["total_return"] * 100, m["cagr"] * 100, m["mdd"] * 100, m["num_trades"],
                 f"{m['win_rate']:.1%}" if "win_rate" in m else "-")
    log.info("로그 파일: %s", log_file)


if __name__ == "__main__":
    main()
