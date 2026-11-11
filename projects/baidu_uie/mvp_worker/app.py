"""简易抽取 HTTP 服务（对齐文档 mvp_worker 用法）。

启动:
  set UIE_MODEL_PATH=models\\review\\<run>\\checkpoint\\export
  python -m mvp_worker --config configs/mvp_worker.json

接口:
  GET  /health
  POST /extract   {"text": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from review_postprocess import enrich_uie_result


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)


def build_app(
    schema: list[str],
    model_path: str | None,
    fallback_model: str,
    device: str,
    enrich_polarity: bool = True,
) -> FastAPI:
    app = FastAPI(title="UIE Review Extractor", version="0.2.0")
    state: dict[str, Any] = {
        "ie": None,
        "schema": schema,
        "model": None,
        "enrich_polarity": enrich_polarity,
    }

    @app.on_event("startup")
    def _load():
        from paddlenlp import Taskflow

        device_id = -1 if device == "cpu" else 0
        if model_path and Path(model_path).exists():
            state["ie"] = Taskflow(
                "information_extraction",
                schema=schema,
                task_path=model_path,
                device_id=device_id,
            )
            state["model"] = model_path
        else:
            state["ie"] = Taskflow(
                "information_extraction",
                schema=schema,
                model=fallback_model,
                device_id=device_id,
            )
            state["model"] = fallback_model
        print(f"loaded model={state['model']}")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": state["model"], "schema": state["schema"]}

    @app.post("/extract")
    def extract(body: ExtractRequest):
        if state["ie"] is None:
            raise HTTPException(503, "model not ready")
        try:
            result = state["ie"](body.text)
        except Exception as e:
            raise HTTPException(500, str(e)) from e
        payload = {"text": body.text, "schema": state["schema"], "result": result}
        if state.get("enrich_polarity"):
            payload["enriched"] = enrich_uie_result(result)
        return payload

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp_worker.json")
    parser.add_argument("--no-load", action="store_true", help="仅打印配置，不启动（占位兼容文档）")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model_path = os.environ.get(cfg.get("model_path_env", "UIE_MODEL_PATH"))
    if args.no_load:
        print(json.dumps({**cfg, "model_path": model_path}, ensure_ascii=False, indent=2))
        return

    app = build_app(
        schema=cfg["schema"],
        model_path=model_path,
        fallback_model=cfg.get("fallback_model", "uie-mini"),
        device=cfg.get("device", "cpu"),
        enrich_polarity=bool(cfg.get("enrich_polarity", True)),
    )
    import uvicorn

    uvicorn.run(app, host=cfg.get("host", "127.0.0.1"), port=int(cfg.get("port", 8088)))


if __name__ == "__main__":
    main()
