"""加载微调后的 export 目录做单条推理验收。

用法:
  python scripts/05_infer_checkpoint.py
  python scripts/05_infer_checkpoint.py --model-path models/review/<run>/checkpoint/export \\
      --schema-config configs/schema_review.json \\
      --text "物流很快，包装完好，感觉满意。【情感极性：正向】"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uie_paths import models_dir  # noqa: E402


def _latest_export() -> Path | None:
    root = models_dir() / "review"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    for run in runs:
        export = run / "checkpoint" / "export"
        if export.exists():
            return export
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="export 目录；默认取最新 run")
    parser.add_argument("--schema-config", default="configs/schema_review.json")
    parser.add_argument(
        "--text",
        default="这款手机屏幕很清晰，整体满意。【情感极性：正向】",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    schema_path = Path(args.schema_config)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["schema"]

    model_path = Path(args.model_path) if args.model_path else _latest_export()
    if model_path is None or not Path(model_path).exists():
        raise SystemExit(
            "未找到 export 模型。请先训练，或用 scripts/06_zero_shot_infer.py 做零样本推理。"
        )
    model_path = Path(model_path)

    try:
        from paddlenlp import Taskflow
    except ImportError as e:
        raise SystemExit(f"需要 paddlenlp: {e}") from e

    ie = Taskflow(
        "information_extraction",
        schema=schema,
        task_path=str(model_path),
        device_id=-1 if args.device == "cpu" else 0,
    )
    result = ie(args.text)
    print("INFER_OK")
    print(f"model={model_path}")
    print(f"text={args.text}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
