"""일봉 OHLCV 수집과 parquet 캐시 (증분 업데이트).

- 미국: yfinance, 수정주가(auto_adjust=True)
- 국내: FinanceDataReader
- 실패한 종목은 로그를 남기고 건너뛴다.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]
Fetcher = Callable[[str, str], pd.DataFrame]  # (symbol, start) -> df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 누락: {missing}")
    df = df[OHLCV].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    # 거래정지일 등 가격 0인 행 제거
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df.astype(float)


def fetch_us(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start, auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise ValueError("빈 데이터")
    return normalize_ohlcv(df)


def fetch_kr(symbol: str, start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.DataReader(symbol, start)
    if df is None or df.empty:
        raise ValueError("빈 데이터")
    return normalize_ohlcv(df)


FETCHERS: dict[str, Fetcher] = {"us": fetch_us, "kr": fetch_kr}


class DataStore:
    def __init__(self, data_cfg: dict, cache_dir: Path, fetchers: dict[str, Fetcher] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = data_cfg
        self.cache_dir = Path(cache_dir)
        self.fetchers = fetchers or FETCHERS
        self.sleep = sleep

    # ---------- 캐시 ----------
    def path(self, market: str, symbol: str) -> Path:
        safe = symbol.replace("^", "IDX_").replace("/", "_")
        return self.cache_dir / market / f"{safe}.parquet"

    def load(self, market: str, symbol: str) -> pd.DataFrame | None:
        p = self.path(market, symbol)
        return pd.read_parquet(p) if p.exists() else None

    def _save(self, market: str, symbol: str, df: pd.DataFrame) -> None:
        p = self.path(market, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)

    # ---------- 수집 ----------
    def _fetch_with_retry(self, market: str, symbol: str, start: str) -> pd.DataFrame:
        fetch = self.fetchers[market]
        retries = int(self.cfg["retries"])
        delay = float(self.cfg["retry_delay_sec"])
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return fetch(symbol, start)
            except Exception as e:  # noqa: BLE001 - 외부 API 오류 형태가 다양함
                last_err = e
                log.warning("%s/%s 수집 실패 (%d/%d): %s", market, symbol, attempt, retries, e)
                if attempt < retries:
                    self.sleep(delay * attempt)
        raise RuntimeError(f"{market}/{symbol} 수집 최종 실패: {last_err}")

    def update(self, market: str, symbol: str) -> pd.DataFrame:
        """캐시가 있으면 증분, 없으면 전체 다운로드. 수정주가가 바뀌었으면 전체 재다운로드."""
        cached = self.load(market, symbol)
        start = self.cfg["start"]
        if cached is None or cached.empty:
            df = self._fetch_with_retry(market, symbol, start)
            log.info("%s/%s 전체 다운로드: %d봉", market, symbol, len(df))
        else:
            overlap = int(self.cfg["overlap_days"])
            since = cached.index[-min(overlap, len(cached))]
            new = self._fetch_with_retry(market, symbol, since.strftime("%Y-%m-%d"))
            common = cached.index.intersection(new.index)
            tol = float(self.cfg["adjust_check_tol"])
            diff = (new.loc[common, "close"] / cached.loc[common, "close"] - 1).abs()
            if len(common) == 0 or diff.max() > tol:
                log.info("%s/%s 겹친 구간 불일치(수정주가 변경 추정) → 전체 재다운로드", market, symbol)
                df = self._fetch_with_retry(market, symbol, start)
            else:
                df = pd.concat([cached[cached.index < since], new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                log.info("%s/%s 증분 업데이트: +%d봉", market, symbol,
                         len(df) - len(cached))
        self._save(market, symbol, df)
        return df

    def update_many(self, market: str, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
        data: dict[str, pd.DataFrame] = {}
        failed: list[str] = []
        batch = int(self.cfg["batch_size"])
        for i, sym in enumerate(symbols):
            if i and i % batch == 0:
                self.sleep(float(self.cfg["batch_pause_sec"]))
            try:
                data[sym] = self.update(market, sym)
            except Exception as e:  # noqa: BLE001
                log.error("%s/%s 건너뜀: %s", market, sym, e)
                failed.append(sym)
            self.sleep(float(self.cfg["request_delay_sec"]))
        return data, failed

    def load_many(self, market: str, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
        data, missing = {}, []
        for s in symbols:
            df = self.load(market, s)
            if df is None:
                missing.append(s)
            else:
                data[s] = df
        return data, missing
