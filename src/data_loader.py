"""
=====================================================================
 data_loader.py : 주가 데이터를 인터넷에서 받아오고, 컴퓨터에 저장해 두는 파일
=====================================================================

이 파일이 하는 일 (3가지)
  1) 다운로드 : 인터넷에서 종목별 "일봉" 데이터를 받는다.
       - 미국 주식 → yfinance (야후 파이낸스 무료 데이터)
       - 한국 주식 → FinanceDataReader (한국 주가 무료 데이터)
  2) 정리     : 데이터 제공처마다 모양이 달라서, 같은 모양으로 맞춘다.
  3) 저장     : data/cache 폴더에 파일로 저장해 두고,
                다음 실행 때는 "새로 생긴 날짜만" 추가로 받는다 (증분 업데이트).

용어 정리
  - 일봉(OHLCV) : 하루 동안의 가격 요약. 하루 = 한 줄(행).
        open   = 시가   (그날 처음 거래된 가격)
        high   = 고가   (그날 가장 높았던 가격)
        low    = 저가   (그날 가장 낮았던 가격)
        close  = 종가   (그날 마지막 거래 가격)
        volume = 거래량 (그날 거래된 주식 수)
  - 수정주가 : 액면분할·배당이 있으면 과거 가격을 지금 기준으로 고쳐 놓은 가격.
        예) 1주를 2주로 쪼개면(분할) 주가가 반토막 나는데, 이건 손해가 아니다.
            과거 가격도 반으로 고쳐 놔야 차트가 "폭락"처럼 보이지 않는다.
  - 캐시 : 한 번 받은 데이터를 저장해 두고 다시 쓰는 것 (매번 다 받으면 느리니까).
  - DataFrame : pandas 라이브러리의 "엑셀 표" 같은 자료형. 행=날짜, 열=시가/고가/...
"""
from __future__ import annotations

import logging           # 로그(실행 기록) 남기기
import time              # 잠깐 쉬기(sleep)
from pathlib import Path # 파일·폴더 경로 다루기
from typing import Callable

import pandas as pd      # 표(DataFrame) 데이터를 다루는 라이브러리

log = logging.getLogger(__name__)

# 우리 프로그램이 쓰는 표준 열 이름 5개 (시가, 고가, 저가, 종가, 거래량)
OHLCV = ["open", "high", "low", "close", "volume"]

# "다운로드 함수"의 모양: (종목코드, 시작날짜)를 넣으면 주가 표를 돌려준다
Fetcher = Callable[[str, str], pd.DataFrame]


# ---------------------------------------------------------------------
# [정리] 데이터 제공처마다 다른 모양을 우리 표준 모양으로 맞춘다
# ---------------------------------------------------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # 열 이름을 소문자로 통일 ("Close" → "close")
    df = df.rename(columns=str.lower)

    # 필요한 5개 열이 없으면 잘못된 데이터 → 에러
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 누락: {missing}")
    df = df[OHLCV].copy()   # 필요한 5개 열만 남긴다

    # 날짜 정리: 시간대(뉴욕 시간 등) 정보를 빼고 "날짜"만 남긴다
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"

    # 같은 날짜가 두 번 있으면 마지막 것만 남기고, 날짜순으로 정렬
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 가격이 비어 있는 날 제거
    df = df.dropna(subset=["open", "high", "low", "close"])

    # 가격이 0인 날 제거 (거래정지일에 0으로 찍히는 경우가 있음)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df.astype(float)  # 모든 값을 숫자(실수)로


# ---------------------------------------------------------------------
# [다운로드] 시장별 다운로드 함수
# ---------------------------------------------------------------------
def fetch_us(symbol: str, start: str) -> pd.DataFrame:
    """미국 주식: 야후 파이낸스에서 받는다. auto_adjust=True → 수정주가로 받기."""
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start, auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise ValueError("빈 데이터")
    return normalize_ohlcv(df)


def fetch_kr(symbol: str, start: str) -> pd.DataFrame:
    """한국 주식: FinanceDataReader로 받는다. 종목코드는 6자리 숫자 (예: 005930 = 삼성전자)."""
    import FinanceDataReader as fdr

    df = fdr.DataReader(symbol, start)
    if df is None or df.empty:
        raise ValueError("빈 데이터")
    return normalize_ohlcv(df)


# 시장 이름 → 다운로드 함수 연결표 ("us"면 fetch_us를 쓴다)
FETCHERS: dict[str, Fetcher] = {"us": fetch_us, "kr": fetch_kr}


# ---------------------------------------------------------------------
# [저장소] DataStore : 다운로드 + 캐시 저장 + 증분 업데이트를 담당하는 "창고 관리인"
# ---------------------------------------------------------------------
class DataStore:
    def __init__(self, data_cfg: dict, cache_dir: Path, fetchers: dict[str, Fetcher] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = data_cfg                  # config.yaml의 data: 부분 (재시도 횟수, 쉬는 시간 등)
        self.cache_dir = Path(cache_dir)     # 저장 폴더 (data/cache)
        self.fetchers = fetchers or FETCHERS # 다운로드 함수 (테스트할 땐 가짜 함수로 바꿔 끼움)
        self.sleep = sleep                   # 쉬기 함수 (테스트할 땐 안 쉬도록 바꿔 끼움)

    # ---------- 캐시(저장 파일) 다루기 ----------
    def path(self, market: str, symbol: str) -> Path:
        """종목별 저장 파일 위치. 예) data/cache/us/AAPL.parquet
        '^GSPC'처럼 파일 이름에 쓰기 곤란한 문자는 바꿔서 저장한다."""
        safe = symbol.replace("^", "IDX_").replace("/", "_")
        return self.cache_dir / market / f"{safe}.parquet"
        # parquet(파케이) = 표 데이터를 빠르고 작게 저장하는 파일 형식 (엑셀보다 훨씬 빠름)

    def load(self, market: str, symbol: str) -> pd.DataFrame | None:
        """저장된 파일이 있으면 읽어오고, 없으면 None(없음)을 돌려준다."""
        p = self.path(market, symbol)
        return pd.read_parquet(p) if p.exists() else None

    def _save(self, market: str, symbol: str, df: pd.DataFrame) -> None:
        """파일로 저장 (폴더가 없으면 만든다)."""
        p = self.path(market, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)

    # ---------- 다운로드 (실패하면 다시 시도) ----------
    def _fetch_with_retry(self, market: str, symbol: str, start: str) -> pd.DataFrame:
        """인터넷은 가끔 실패한다. 실패하면 잠깐 쉬었다가 최대 retries번까지 다시 시도한다.
        쉬는 시간은 점점 늘린다 (2초 → 4초 → ...)."""
        fetch = self.fetchers[market]
        retries = int(self.cfg["retries"])
        delay = float(self.cfg["retry_delay_sec"])
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return fetch(symbol, start)          # 성공하면 바로 결과 반환
            except Exception as e:  # noqa: BLE001 - 외부 API 오류 형태가 다양함
                last_err = e
                log.warning("%s/%s 수집 실패 (%d/%d): %s", market, symbol, attempt, retries, e)
                if attempt < retries:
                    self.sleep(delay * attempt)      # 쉬었다가 다시
        raise RuntimeError(f"{market}/{symbol} 수집 최종 실패: {last_err}")

    # ---------- 핵심: 한 종목 업데이트 ----------
    def update(self, market: str, symbol: str) -> pd.DataFrame:
        """
        한 종목의 주가를 최신으로 만든다.

          경우 1) 저장된 파일이 없다 → 처음부터 전부 받는다 (전체 다운로드)
          경우 2) 저장된 파일이 있다 → 마지막 며칠만 다시 받아서 이어 붙인다 (증분 업데이트)
                  단, 겹치는 날의 가격이 저장본과 다르면
                  (= 분할·배당으로 과거 수정주가가 통째로 바뀐 것)
                  → 이어 붙이면 앞뒤가 안 맞으니 처음부터 다시 받는다
        """
        cached = self.load(market, symbol)
        start = self.cfg["start"]

        if cached is None or cached.empty:
            # 경우 1: 처음 받는 종목
            df = self._fetch_with_retry(market, symbol, start)
            log.info("%s/%s 전체 다운로드: %d봉", market, symbol, len(df))
        else:
            # 경우 2: 저장본의 마지막 5일(overlap_days)부터 다시 받는다
            #   → 겹치는 구간을 비교해서 데이터가 바뀌었는지 확인하기 위함
            overlap = int(self.cfg["overlap_days"])
            since = cached.index[-min(overlap, len(cached))]
            new = self._fetch_with_retry(market, symbol, since.strftime("%Y-%m-%d"))

            # 겹치는 날짜들의 종가를 비교: (새 값 ÷ 옛 값 - 1)의 크기 = 몇 % 차이나나
            common = cached.index.intersection(new.index)
            tol = float(self.cfg["adjust_check_tol"])
            diff = (new.loc[common, "close"] / cached.loc[common, "close"] - 1).abs()

            if len(common) == 0 or diff.max() > tol:
                # 0.5% 넘게 다르다 → 과거 가격이 수정됨 → 전부 다시 받기
                log.info("%s/%s 겹친 구간 불일치(수정주가 변경 추정) → 전체 재다운로드", market, symbol)
                df = self._fetch_with_retry(market, symbol, start)
            else:
                # 일치한다 → [저장본의 앞부분] + [새로 받은 부분]을 이어 붙인다
                df = pd.concat([cached[cached.index < since], new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                log.info("%s/%s 증분 업데이트: +%d봉", market, symbol,
                         len(df) - len(cached))
        self._save(market, symbol, df)
        return df

    # ---------- 여러 종목 한꺼번에 ----------
    def update_many(self, market: str, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """
        종목 목록 전체를 업데이트한다.
          - 종목 사이마다 잠깐 쉬고, 10종목(batch_size)마다 조금 더 쉰다
            → 너무 빠르게 요청하면 데이터 제공처가 차단할 수 있어서
          - 한 종목이 실패해도 멈추지 않고 기록만 남기고 다음 종목으로 넘어간다
        돌려주는 값: (성공한 종목들의 주가 표, 실패한 종목 목록)
        """
        data: dict[str, pd.DataFrame] = {}
        failed: list[str] = []
        batch = int(self.cfg["batch_size"])
        for i, sym in enumerate(symbols):
            if i and i % batch == 0:
                self.sleep(float(self.cfg["batch_pause_sec"]))   # 배치마다 쉬기
            try:
                data[sym] = self.update(market, sym)
            except Exception as e:  # noqa: BLE001
                log.error("%s/%s 건너뜀: %s", market, sym, e)      # 실패 기록 후 건너뛰기
                failed.append(sym)
            self.sleep(float(self.cfg["request_delay_sec"]))     # 종목마다 쉬기
        return data, failed

    def load_many(self, market: str, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """다운로드 없이 저장된 파일만 읽어온다 (--no-update 옵션으로 실행할 때 사용)."""
        data, missing = {}, []
        for s in symbols:
            df = self.load(market, s)
            if df is None:
                missing.append(s)
            else:
                data[s] = df
        return data, missing
