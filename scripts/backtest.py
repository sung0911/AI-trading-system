"""
=====================================================================
 scripts/backtest.py : 실행 버튼 — 지금까지의 부품을 순서대로 연결해서 백테스트를 돌린다
=====================================================================

이 파일 자체에는 계산 로직이 없다. 각 부품을 "순서대로 불러오는" 역할만 한다.

  ① config.yaml 읽기         (src/config.py)
  ② 주가 받기/불러오기        (src/data_loader.py)
  ③ 박스 전략 준비            (src/strategies/box.py  ← 안에서 indicators.py 사용)
  ④ 백테스트 실행             (src/engine.py)
  ⑤ 성적표 저장               (src/report.py)
  → 미국(us), 한국(kr) 시장마다 ②~⑤를 반복

사용법 (터미널에서 프로젝트 폴더 기준으로 입력)
  python scripts/backtest.py                 # 미국·한국 모두, 주가 업데이트 후 실행
  python scripts/backtest.py --market us     # 미국만
  python scripts/backtest.py --market kr     # 한국만
  python scripts/backtest.py --no-update     # 인터넷 다운로드 없이 저장된 주가만 사용 (빠름)
  python scripts/backtest.py --config 다른설정.yaml   # 다른 설정 파일로 실행
"""
from __future__ import annotations

import argparse   # 터미널 옵션(--market 등)을 읽는 도구
import logging
import sys
from pathlib import Path

# 이 파일은 scripts/ 폴더 안에 있어서, 위 폴더의 src/를 찾을 수 있도록 경로를 알려준다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve, setup_logging  # noqa: E402
from src.data_loader import DataStore  # noqa: E402
from src.engine import run_backtest  # noqa: E402
from src.report import write_report  # noqa: E402
from src.strategies import BoxStrategy  # noqa: E402

log = logging.getLogger("backtest")


def main() -> None:
    # ---- 터미널 옵션 읽기 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)                                # 설정 파일 경로
    ap.add_argument("--market", choices=["us", "kr", "all"], default="all")  # 시장 선택
    ap.add_argument("--no-update", action="store_true", help="다운로드 없이 캐시만 사용")
    args = ap.parse_args()

    # ---- ① 설정 읽기, 로그 파일 준비 (logs/backtest_날짜_시간.log) ----
    cfg = load_config(args.config)
    log_file = setup_logging(cfg, "backtest")

    # ---- 부품 준비: 주가 창고 관리인, 전략 ----
    store = DataStore(cfg["data"], resolve(cfg["data"]["cache_dir"]))
    params = cfg["strategy"]["box"]
    strategy = BoxStrategy(params)
    markets = ["us", "kr"] if args.market == "all" else [args.market]

    # ---- 시장마다 반복 ----
    for mk in markets:
        mcfg = cfg["markets"][mk]                       # 이 시장의 설정 (자금, 비용, 종목)
        symbols = [str(s) for s in mcfg["symbols"]]
        log.info("=== [%s] 종목 %d개 ===", mk, len(symbols))

        # ② 주가 준비 (+ 비교용 지수)
        if args.no_update:
            data, failed = store.load_many(mk, symbols)          # 저장본만 읽기
            bench_df = store.load(mk, mcfg["benchmark"])
        else:
            data, failed = store.update_many(mk, symbols)        # 최신으로 업데이트
            try:
                bench_df = store.update(mk, mcfg["benchmark"])
            except Exception as e:  # noqa: BLE001
                log.error("벤치마크 %s 수집 실패: %s", mcfg["benchmark"], e)
                bench_df = None                                  # 지수 없이도 백테스트는 진행
        if not data:
            log.error("[%s] 데이터가 없어 건너뜀", mk)
            continue

        # ④ 백테스트 실행
        res = run_backtest(data, strategy, mcfg, cfg["risk"],
                           cfg["backtest"].get("start"), cfg["backtest"].get("end"))

        # ⑤ 성적표 저장 → reports/backtest_us_box/ 또는 reports/backtest_kr_box/
        out = resolve(cfg["report"]["out_dir"]) / f"backtest_{mk}_box"
        m = write_report(out, f"{mk.upper()} / box", res, data,
                         bench_df["close"] if bench_df is not None else None, mcfg["currency"],
                         params, int(cfg["report"]["max_symbol_charts"]), failed)

        # 터미널에 한 줄 요약 출력
        log.info("[%s] 누적 %.2f%% | CAGR %.2f%% | MDD %.2f%% | 거래 %d회 | 승률 %s",
                 mk, m["total_return"] * 100, m["cagr"] * 100, m["mdd"] * 100, m["num_trades"],
                 f"{m['win_rate']:.1%}" if "win_rate" in m else "-")
    log.info("로그 파일: %s", log_file)


# "python scripts/backtest.py"로 직접 실행했을 때만 main()을 돌린다
if __name__ == "__main__":
    main()
