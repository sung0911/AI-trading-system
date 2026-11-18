"""config.yaml 로드와 로깅 설정."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("markets", "data", "strategy", "risk", "backtest", "report"):
        if key not in cfg:
            raise KeyError(f"config.yaml에 '{key}' 섹션이 없습니다")
    return cfg


def resolve(path: str | Path) -> Path:
    """config의 상대경로를 프로젝트 루트 기준으로 바꾼다."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def setup_logging(cfg: dict, name: str) -> Path:
    log_cfg = cfg.get("logging", {})
    log_dir = resolve(log_cfg.get("dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=fmt,
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # 외부 라이브러리 잡음 줄이기
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file
