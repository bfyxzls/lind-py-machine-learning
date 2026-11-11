"""UIE 微调入口（对齐文档 scripts/02_train_uie.py）。

用法:
  cd projects/baidu_uie
  python scripts/00_gen_synthetic_data.py
  python scripts/02_train_uie.py --config configs/train_cpu_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uie_paths import cache_dir, datasets_dir, ensure_layout, models_dir  # noqa: E402
from vendor.uie.finetune import train_and_export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="训练配置 JSON")
    args = parser.parse_args()

    ensure_layout()
    os.environ.setdefault("PADDLE_HOME", str(cache_dir()))

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    version = cfg["dataset_version"]
    data_dir = datasets_dir() / version
    train_path = data_dir / "train.txt"
    dev_path = data_dir / "dev.txt"
    if not train_path.exists() or not dev_path.exists():
        raise SystemExit(
            f"找不到数据: {train_path}\n请先运行: python scripts/00_gen_synthetic_data.py "
            f"--dataset-version {version}"
        )

    data_type = cfg.get("data_type", "review")
    run_id = f"{data_type}-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = models_dir() / data_type / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"run_dir={run_dir}")
    print(f"pretrained={cfg['pretrained_model']} device={cfg.get('device')}")
    export_dir = train_and_export(cfg, train_path, dev_path, run_dir)
    print("TRAIN_OK")
    print(f"checkpoint_export={export_dir}")


if __name__ == "__main__":
    main()
