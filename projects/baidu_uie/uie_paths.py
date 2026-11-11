"""项目路径约定：对齐文档里的 UIE_ROOT / datasets / models。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def uie_root() -> Path:
    """优先环境变量 UIE_ROOT，否则用本项目目录。"""
    env = os.environ.get("UIE_ROOT")
    if env:
        return Path(env).resolve()
    return PROJECT_ROOT


def datasets_dir() -> Path:
    return uie_root() / "datasets" / "dataset_versions"


def models_dir() -> Path:
    return uie_root() / "models"


def cache_dir() -> Path:
    return uie_root() / "cache"


def ensure_layout() -> None:
    for p in (datasets_dir(), models_dir(), cache_dir(), uie_root() / "logs"):
        p.mkdir(parents=True, exist_ok=True)
