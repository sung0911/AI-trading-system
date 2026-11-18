"""
=====================================================================
 config.py : config.yaml을 읽어오고, 로그(실행 기록) 저장을 준비하는 작은 도우미
=====================================================================
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml   # YAML 파일을 읽는 라이브러리

# 프로젝트 최상위 폴더 (이 파일 위치 src/config.py 에서 두 단계 위)
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml을 읽어서 사전(dict)으로 돌려준다. 꼭 필요한 항목이 빠졌으면 에러."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("markets", "data", "strategy", "risk", "backtest", "report"):
        if key not in cfg:
            raise KeyError(f"config.yaml에 '{key}' 섹션이 없습니다")
    return cfg


def resolve(path: str | Path) -> Path:
    """config에 적힌 상대경로(예: data/cache)를 프로젝트 폴더 기준 전체 경로로 바꾼다."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def setup_logging(cfg: dict, name: str) -> Path:
    """로그를 '화면'과 'logs/이름_날짜_시간.log 파일' 두 곳에 동시에 남기도록 설정한다."""
    log_cfg = cfg.get("logging", {})
    log_dir = resolve(log_cfg.get("dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"   # 시간 / 등급 / 어디서 / 내용
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=fmt,
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # 외부 라이브러리가 남기는 자잘한 기록은 줄인다
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file
