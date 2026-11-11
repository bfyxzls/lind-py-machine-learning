"""评价词 → 情感极性（规则后处理）。

UIE 只负责从原文抽出「商品属性 / 评价词」；
「正向 / 负向」通常不在原文里，不宜当成 span 硬抽。
"""

from __future__ import annotations

from typing import Any

# 可按业务继续扩充
POS_OPINIONS = {
    "清晰",
    "耐用",
    "很快",
    "耐心",
    "出色",
    "实惠",
    "满意",
    "不错",
    "好用",
    "流畅",
    "精致",
}
NEG_OPINIONS = {
    "发糊",
    "不耐用",
    "太慢",
    "敷衍",
    "粗糙",
    "发热严重",
    "失望",
    "糟心",
    "卡顿",
    "差劲",
    "模糊",
    "太差",
}


def polarity_of_opinion(text: str) -> str | None:
    t = text.strip()
    if t in POS_OPINIONS:
        return "正向"
    if t in NEG_OPINIONS:
        return "负向"
    # 简单启发：带「不」「太慢」「差」等
    if any(k in t for k in ("不", "差", "慢", "糊", "敷衍", "粗糙", "发热", "糟")):
        return "负向"
    if any(k in t for k in ("清", "好", "快", "耐", "棒", "优", "精")):
        return "正向"
    return None


def enrich_uie_result(raw: Any) -> dict:
    """把 Taskflow 原始结果整理成更易读的结构，并推断情感极性。"""
    items: list[dict] = []
    blocks = raw if isinstance(raw, list) else [raw]
    attrs: list[dict] = []
    opinions: list[dict] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        for hit in block.get("商品属性") or []:
            attrs.append(hit)
        for hit in block.get("评价词") or []:
            opinions.append(hit)

    # 按出现顺序粗配对：同句多属性时，评价词与最近属性关联
    for op in opinions:
        pol = polarity_of_opinion(op.get("text", ""))
        items.append(
            {
                "评价词": op.get("text"),
                "span": {"start": op.get("start"), "end": op.get("end")},
                "probability": op.get("probability"),
                "情感极性": pol,
            }
        )

    return {
        "商品属性": attrs,
        "评价细节": items,
        "说明": "情感极性由评价词规则推断，不是 UIE 从原文直接抽出的 span",
    }
