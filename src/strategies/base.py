"""
=====================================================================
 strategies/base.py : 모든 전략이 지켜야 하는 "공통 틀(약속)"
=====================================================================

왜 틀이 필요한가?
  나중에 이동평균 전략, RSI 전략 등을 추가할 때
  엔진(engine.py)은 전략 내용을 몰라도 똑같은 방식으로 물어볼 수 있어야 한다.
  → "모든 전략은 아래 3가지 질문에 답할 수 있어야 한다"는 약속을 정해 둔 것.

  엔진이 전략에게 묻는 3가지 질문
    1) generate_signals : "이 종목, 날짜별로 매수 신호가 있어?"
    2) entry_meta       : "샀을 때 기억해 둘 정보는 뭐야?" (예: 박스 상·하단)
    3) exit_reason      : "지금 들고 있는 거, 오늘 팔아야 해? 왜?"

  (ABC / abstractmethod = "이 틀을 쓰는 전략은 이 함수들을 반드시 만들어야 한다"는 파이썬 문법)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"   # 전략 이름 (리포트 표시용)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """질문 1) 날짜별 매수 신호표를 만든다.

        t일 "종가까지"의 정보만 보고 판단한다. 실제 매수는 엔진이 다음날(t+1) 시가에 한다.

        돌려줄 표에 반드시 있어야 하는 열
          entry  (참/거짓) : 이 날 매수 신호가 있나?
          score  (숫자)    : 같은 날 후보가 여러 개면 점수 높은 종목부터 산다
          reason (글자)    : 왜 샀는지 설명 (로그에 남김)
        """

    @abstractmethod
    def entry_meta(self, signal_row: pd.Series) -> dict:
        """질문 2) 매수할 때의 정보를 기억해 둔다.
        예) 박스 하단 100, 상단 110 → 팔 때까지 이 기준으로 손절·익절을 판단한다."""

    @abstractmethod
    def exit_reason(self, meta: dict, bar: pd.Series) -> str | None:
        """질문 3) 들고 있는 종목을 오늘 종가 기준으로 팔지 판단한다.
        팔아야 하면 이유(글자)를, 계속 들고 있으면 None(없음)을 돌려준다."""
