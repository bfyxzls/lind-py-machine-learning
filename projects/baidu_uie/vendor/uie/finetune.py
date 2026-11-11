"""UIE 少样本微调（简化可运行版，对齐 PaddleNLP UIE 数据格式）。

依赖: paddlepaddle + paddlenlp==2.8.x
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def _require_paddle():
    try:
        import paddle
        import paddlenlp  # noqa: F401
        # paddlenlp 2.8+ 类名是 UIE（旧文档/示例里的 UIEModel 已移除）
        from paddlenlp.transformers import AutoTokenizer, UIE
    except ImportError as e:
        raise SystemExit(
            "未安装 paddle/paddlenlp，或版本不匹配。请先阅读 projects/baidu_uie/PROJECT.md。\n"
            f"原始错误: {e}"
        ) from e
    return paddle, AutoTokenizer, UIE


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _char_to_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    """把字符级 [start, end) 映射到 token 下标；offsets 为 tokenizer offset_mapping。"""
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e == 0:
            continue  # special token
        if tok_start is None and s <= start < e:
            tok_start = i
        if s < end <= e:
            tok_end = i
            break
        if tok_start is not None and e >= end:
            tok_end = i
            break
    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end


def convert_example(example: dict, tokenizer, max_seq_length: int) -> dict[str, Any] | None:
    content = example["content"]
    prompt = example["prompt"]
    result_list = example.get("result_list") or []

    encoded = tokenizer(
        text=prompt,
        text_pair=content,
        max_length=max_seq_length,
        truncation=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
    )

    seq_len = len(encoded["input_ids"])
    start_ids = [0.0] * seq_len
    end_ids = [0.0] * seq_len
    offsets = encoded["offset_mapping"]

    # text_pair 场景下，部分 tokenizer 的 offset 只相对各自片段；
    # paddlenlp UIE 习惯：prompt + SEP + content，content 字符偏移需加上 prompt 长度 + 1 的策略不稳定。
    # 更稳妥：在完整解码串里用原文 span 对齐 —— 这里用「在 content 上找 span，再映射到 pair 段」。
    # 若 offset_mapping 对 text/text_pair 分别从 0 计，则对 content span 直接用 content 侧 offsets。
    pair_offsets = []
    # 找到 token_type_ids==1 的区间作为 content（若无 token_type，则 SEP 后全部）
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        for i, (tt, off) in enumerate(zip(token_type_ids, offsets)):
            if tt == 1:
                pair_offsets.append((i, off))
    else:
        # fallback: 全部非空 offset
        pair_offsets = [(i, off) for i, off in enumerate(offsets) if off != (0, 0)]

    for item in result_list:
        span = _char_to_token_span([o for _, o in pair_offsets] if pair_offsets else offsets, item["start"], item["end"])
        if span is None:
            # 再试：用全局 offsets + 原文在 content 中的位置
            # 构造「仅 content 字符」映射：遍历 offsets，跳过 prompt 段
            if token_type_ids is not None:
                content_offs = [(0, 0)] * seq_len
                for i, off in pair_offsets:
                    content_offs[i] = off
                span2 = _char_to_token_span(content_offs, item["start"], item["end"])
                if span2 is None:
                    continue
                s, e = span2
            else:
                continue
        else:
            if token_type_ids is not None and pair_offsets:
                # span 是相对 pair_offsets 列表的下标，需映回真实 token index
                s = pair_offsets[span[0]][0]
                e = pair_offsets[span[1]][0]
            else:
                s, e = span
        if 0 <= s < seq_len and 0 <= e < seq_len:
            start_ids[s] = 1.0
            end_ids[e] = 1.0

    return {
        "input_ids": encoded["input_ids"],
        "token_type_ids": encoded.get("token_type_ids") or [0] * seq_len,
        "position_ids": encoded.get("position_ids") or list(range(seq_len)),
        "attention_mask": encoded.get("attention_mask") or [1] * seq_len,
        "start_positions": start_ids,
        "end_positions": end_ids,
    }


def _pad_batch(features: list[dict], pad_id: int = 0) -> dict:
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {
        "input_ids": [],
        "token_type_ids": [],
        "position_ids": [],
        "attention_mask": [],
        "start_positions": [],
        "end_positions": [],
    }
    for f in features:
        pad = max_len - len(f["input_ids"])
        for k in batch:
            if k in ("start_positions", "end_positions"):
                batch[k].append(f[k] + [0.0] * pad)
            elif k == "attention_mask":
                batch[k].append(f[k] + [0] * pad)
            else:
                batch[k].append(f[k] + [pad_id] * pad)
    return batch


def _batch_to_tensor(paddle, batch: dict):
    out = {}
    for k, v in batch.items():
        dtype = "float32" if k.endswith("positions") else "int64"
        out[k] = paddle.to_tensor(v, dtype=dtype)
    return out


def evaluate(model, paddle, tokenizer, examples: list[dict], max_seq_length: int, batch_size: int = 8) -> dict:
    model.eval()
    # span-level 粗评：预测 start/end argmax 与金标集合 IoU 近似用 exact match 率
    total = 0
    hit = 0
    feats = []
    golds = []
    for ex in examples:
        f = convert_example(ex, tokenizer, max_seq_length)
        if f is None:
            continue
        feats.append(f)
        golds.append(ex)

    for i in range(0, len(feats), batch_size):
        chunk = feats[i : i + batch_size]
        gold_chunk = golds[i : i + batch_size]
        batch = _batch_to_tensor(paddle, _pad_batch(chunk))
        with paddle.no_grad():
            start_prob, end_prob = model(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"],
                position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"],
            )
        start_ids = start_prob.numpy() >= 0.5
        end_ids = end_prob.numpy() >= 0.5
        for j in range(len(chunk)):
            total += 1
            g_s = np.array(chunk[j]["start_positions"]) >= 0.5
            g_e = np.array(chunk[j]["end_positions"]) >= 0.5
            # 截断到实际长度
            L = len(chunk[j]["input_ids"])
            if np.array_equal(start_ids[j, :L], g_s[:L]) and np.array_equal(end_ids[j, :L], g_e[:L]):
                hit += 1
            _ = gold_chunk  # reserved
    acc = hit / total if total else 0.0
    model.train()
    return {"exact_match": acc, "n": total}


def train_and_export(cfg: dict, train_path: Path, dev_path: Path, run_dir: Path) -> Path:
    paddle, AutoTokenizer, UIE = _require_paddle()

    device = cfg.get("device", "cpu")
    if device.startswith("gpu"):
        paddle.device.set_device("gpu:0" if device == "gpu" else device)
    else:
        paddle.device.set_device("cpu")

    model_name = cfg["pretrained_model"]
    max_seq_length = int(cfg.get("max_seq_length", 512))
    batch_size = int(cfg.get("batch_size", 4))
    epochs = int(cfg.get("num_epochs", 2))
    lr = float(cfg.get("learning_rate", 1e-5))
    seed = int(cfg.get("seed", 42))
    logging_steps = int(cfg.get("logging_steps", 5))
    eval_steps = int(cfg.get("eval_steps", 50))

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = UIE.from_pretrained(model_name)
    model.train()

    train_raw = load_jsonl(train_path)
    dev_raw = load_jsonl(dev_path)
    train_feats = [f for f in (convert_example(x, tokenizer, max_seq_length) for x in train_raw) if f]
    if not train_feats:
        raise RuntimeError("训练集转换后为空，请检查标注 span 与 tokenizer 映射")

    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters())
    criterion = paddle.nn.BCELoss()

    ckpt_dir = run_dir / "checkpoint"
    export_dir = ckpt_dir / "export"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_em = -1.0
    log_lines = []

    for epoch in range(epochs):
        random.shuffle(train_feats)
        n_batches = math.ceil(len(train_feats) / batch_size)
        for bi in range(n_batches):
            chunk = train_feats[bi * batch_size : (bi + 1) * batch_size]
            batch = _batch_to_tensor(paddle, _pad_batch(chunk))
            start_prob, end_prob = model(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"],
                position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss_s = criterion(start_prob, batch["start_positions"])
            loss_e = criterion(end_prob, batch["end_positions"])
            loss = loss_s + loss_e
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            global_step += 1

            if global_step % logging_steps == 0:
                msg = f"epoch={epoch+1} step={global_step} loss={float(loss):.4f}"
                print(msg)
                log_lines.append(msg)

            if global_step % eval_steps == 0:
                metrics = evaluate(model, paddle, tokenizer, dev_raw, max_seq_length, batch_size)
                msg = f"EVAL step={global_step} exact_match={metrics['exact_match']:.4f} n={metrics['n']}"
                print(msg)
                log_lines.append(msg)
                if metrics["exact_match"] >= best_em:
                    best_em = metrics["exact_match"]
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

    # final save + export
    metrics = evaluate(model, paddle, tokenizer, dev_raw, max_seq_length, batch_size)
    print(f"FINAL_EVAL exact_match={metrics['exact_match']:.4f}")
    log_lines.append(f"FINAL_EVAL exact_match={metrics['exact_match']:.4f}")
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    (run_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (run_dir / "RUN_OK").write_text(run_dir.name + "\n", encoding="utf-8")
    print(f"export={export_dir}")
    return export_dir
