"""零样本抽取：不训练，直接用官方 UIE 基座 + schema。

适合先验证「领域 schema 是否合理」，再决定是否少样本微调。

用法:
  python scripts/06_zero_shot_infer.py
  python scripts/06_zero_shot_infer.py --model uie-base --text "客服回复很敷衍，沟通体验糟心。"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-config", default="configs/schema_review.json")
    parser.add_argument("--model", default="uie-mini", help="uie-mini / uie-base / uie-medium ...")
    parser.add_argument(
        "--text",
        default="物流太慢了，等了很久才到，有点失望。【情感极性：负向】",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    schema_path = Path(args.schema_config)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["schema"]

    try:
        from paddlenlp import Taskflow
    except ImportError as e:
        raise SystemExit(
            "未安装 paddlenlp。CPU 安装示例:\n"
            "  pip install paddlepaddle==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
            "  pip install -r requirements.txt\n"
            f"原始错误: {e}"
        ) from e

    ie = Taskflow(
        "information_extraction",
        schema=schema,
        model=args.model,
        device_id=-1 if args.device == "cpu" else 0,
    )
    result = ie(args.text)
    print("ZERO_SHOT_OK")
    print(f"model={args.model}")
    print(f"schema={schema}")
    print(f"text={args.text}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
