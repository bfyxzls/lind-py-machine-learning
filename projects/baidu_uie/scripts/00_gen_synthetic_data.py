"""生成电商评论观点抽取的合成 UIE 标注数据（JSONL）。

情感极性按「原规则」：在文本末尾标注 【情感极性：正向/负向】，
由 UIE 作为 span 字段直接学习抽取。

用法:
  python scripts/00_gen_synthetic_data.py --dataset-version review/smoke-v0.1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uie_paths import datasets_dir, ensure_layout  # noqa: E402

SCHEMA = ["商品属性", "评价词", "情感极性"]

# (属性, 评价词, 极性, 模板)
TEMPLATES = [
    ("屏幕", "清晰", "正向", "这款手机{attr}很{opin}，整体{pol_word}。"),
    ("屏幕", "发糊", "负向", "失望，{attr}有点{opin}，体验{pol_word}。"),
    ("电池", "耐用", "正向", "{attr}非常{opin}，续航让人{pol_word}。"),
    ("电池", "不耐用", "负向", "{attr}太{opin}了，用一天就没电，很{pol_word}。"),
    ("物流", "很快", "正向", "{attr}{opin}，包装完好，感觉{pol_word}。"),
    ("物流", "太慢", "负向", "{attr}{opin}，等了很久才到，有点{pol_word}。"),
    ("客服", "耐心", "正向", "{attr}态度{opin}，问题解决得及时，{pol_word}。"),
    ("客服", "敷衍", "负向", "{attr}回复很{opin}，沟通体验{pol_word}。"),
    ("音质", "出色", "正向", "耳机{attr}{opin}，听歌很{pol_word}。"),
    ("做工", "粗糙", "负向", "机身{attr}偏{opin}，细节让人{pol_word}。"),
    ("价格", "实惠", "正向", "{attr}比较{opin}，性价比高，很{pol_word}。"),
    ("散热", "发热严重", "负向", "玩游戏时{attr}{opin}，体验{pol_word}。"),
]

POL_WORDS = {
    "正向": ["满意", "开心", "不错"],
    "负向": ["失望", "糟心", "难受"],
}


def _span(content: str, piece: str) -> dict:
    start = content.find(piece)
    if start < 0:
        raise ValueError(f"span not found: {piece!r} in {content!r}")
    return {"text": piece, "start": start, "end": start + len(piece)}


def make_example(rng: random.Random) -> list[dict]:
    attr, opin, polarity, tmpl = rng.choice(TEMPLATES)
    pol_word = rng.choice(POL_WORDS[polarity])
    content = tmpl.format(attr=attr, opin=opin, pol_word=pol_word)
    # 原规则：末尾显式挂上「正向/负向」，供 UIE 抽情感极性
    if polarity not in content:
        content = f"{content}【情感极性：{polarity}】"
    return [
        {"content": content, "prompt": "商品属性", "result_list": [_span(content, attr)]},
        {"content": content, "prompt": "评价词", "result_list": [_span(content, opin)]},
        {"content": content, "prompt": "情感极性", "result_list": [_span(content, polarity)]},
    ]


def write_split(path: Path, n_texts: int, seed: int) -> int:
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(n_texts):
        for row in make_example(rng):
            lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 UIE 合成数据")
    parser.add_argument("--dataset-version", default="review/smoke-v0.1.0")
    parser.add_argument("--train-size", type=int, default=40, help="训练「评论条数」（每条扩成 3 行 prompt）")
    parser.add_argument("--dev-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_layout()
    out = datasets_dir() / args.dataset_version
    out.mkdir(parents=True, exist_ok=True)

    n_train = write_split(out / "train.txt", args.train_size, args.seed)
    n_dev = write_split(out / "dev.txt", args.dev_size, args.seed + 1)
    n_test = write_split(out / "test.txt", args.test_size, args.seed + 2)

    manifest = {
        "dataset_version": args.dataset_version,
        "schema": SCHEMA,
        "domain": "ecommerce_review",
        "polarity": "uie_span_with_tag",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {"train_lines": n_train, "dev_lines": n_dev, "test_lines": n_test},
        "format": "uie_jsonl",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DATA_OK")
    print(f"path={out}")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
