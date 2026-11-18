"""전략 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """t일 종가까지의 데이터로 t일 신호를 만든다.

        반환 DataFrame(df와 같은 index)에는 최소 다음 컬럼이 있어야 한다.
          entry (bool) : t일 종가 기준 매수 신호 → 엔진이 t+1일 시가에 체결
          score (float): 같은 날 후보가 여러 개일 때 높은 순으로 우선 매수
          reason (str) : 판단 근거 (로그용)
        """

    @abstractmethod
    def entry_meta(self, signal_row: pd.Series) -> dict:
        """매수 신호 시점의 정보(예: 박스 상·하단)를 포지션에 고정해 둔다."""

    @abstractmethod
    def exit_reason(self, meta: dict, bar: pd.Series) -> str | None:
        """보유 중인 포지션을 t일 종가 기준으로 청산할지 판단한다. 청산 사유 또는 None."""
